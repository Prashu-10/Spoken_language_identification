# coding=utf-8 
# copyright by speechflow  2023/03/17

import argparse
import tensorflow as tf

# Modern GPU memory growth configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

import datetime
import time
import os
from shutil import copyfile
import matplotlib.pyplot as plt
from vocab.vocab import Vocab
from configs.config import Config
from models.model import MulSpeechLR as Model
from termcolor import colored
from featurizers.speech_featurizers import NumpySpeechFeaturizer
from dataset import create_dataset
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, recall_score, precision_score

# Use MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()

def train(config_file):
    config = Config(config_file)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_log_root = "./saved_weights/"
    if not os.path.exists(dir_log_root):
        os.makedirs(dir_log_root)
    dir_current = os.path.join(dir_log_root, current_time)
    if not os.path.exists(dir_current):
        os.makedirs(dir_current)
        os.makedirs(os.path.join(dir_current, 'best'))
        os.makedirs(os.path.join(dir_current, 'last'))
    
    copyfile(config_file, os.path.join(dir_current, 'config.yml'))
    log_file = open(os.path.join(dir_current, 'log.txt'), 'w')
    copyfile(config.dataset_config['vocabulary'], os.path.join(dir_current, 'vocab.txt'))
    
    config.print()
    log_file.write(config.toString())
    log_file.flush()

    vocab = Vocab(config.dataset_config['vocabulary'])
    batch_size = config.running_config['batch_size']
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    speech_featurizer = NumpySpeechFeaturizer(config.speech_config)

    with strategy.scope():
        model = Model(**config.model_config, vocab_size=len(vocab.token_list))
        if config.running_config['load_weights'] is not None:
            model.load_weights(config.running_config['load_weights'])
        model.add_featurizers(speech_featurizer)
        model.init_build([None, config.speech_config['num_feature_bins']])
        model.summary()

        # Use modern optimizers with learning rate schedules
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            config.optimizer_config['max_lr'],
            decay_steps=config.running_config['num_epochs'] * config.running_config['train_steps']
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01
        )
        
        # Modern loss functions
        loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
            from_logits=True,
            alpha=0.25,
            gamma=2.0,
            reduction=tf.keras.losses.Reduction.NONE
        )
        loss_fn_smooth = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1,
            reduction=tf.keras.losses.Reduction.NONE
        )

    # Create datasets
    train_dataset = create_dataset(
        batch_size=global_batch_size,
        load_type=config.dataset_config['load_type'],
        data_type=config.dataset_config['train'],
        speech_featurizer=speech_featurizer,
        config=config,
        vocab=vocab
    )
    eval_dataset = create_dataset(
        batch_size=global_batch_size,
        load_type=config.dataset_config['load_type'],
        data_type=config.dataset_config['dev'],
        speech_featurizer=speech_featurizer,
        config=config,
        vocab=vocab
    )
    test_dataset = create_dataset(
        batch_size=global_batch_size,
        load_type=config.dataset_config['load_type'],
        data_type=config.dataset_config['test'],
        speech_featurizer=speech_featurizer,
        config=config,
        vocab=vocab
    )

    # Distribute datasets
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    eval_dist_dataset = strategy.experimental_distribute_dataset(eval_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    def compute_loss(labels, predictions, smooth=False):
        per_example_loss = loss_fn_smooth(tf.one_hot(labels, len(vocab.token_list)), predictions) if smooth else \
                          loss_fn(tf.one_hot(labels, len(vocab.token_list)), predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    @tf.function
    def train_step(inputs):
        x, x_len, y = inputs
        
        with tf.GradientTape() as tape:
            predictions = model([x, x_len], training=True)
            loss = compute_loss(y, predictions, smooth=True)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss.update_state(loss)
        train_accuracy.update_state(y, predictions)
        return loss

    @tf.function
    def test_step(inputs):
        x, x_len, y = inputs
        predictions = model([x, x_len], training=False)
        loss = compute_loss(y, predictions, smooth=True)
        
        val_loss.update_state(loss)
        val_accuracy.update_state(y, predictions)
        return predictions, y

    @tf.function
    def distributed_train_step(inputs):
        per_replica_losses = strategy.run(train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(inputs):
        return strategy.run(test_step, args=(inputs,))

    # Training loop
    best_accuracy = 0
    for epoch in range(1, config.running_config['num_epochs'] + 1):
        start_time = time.time()
        
        # Reset metrics
        train_loss.reset_states()
        val_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()

        # Training
        for step, inputs in enumerate(train_dist_dataset):
            loss = distributed_train_step(inputs)
            if step % 10 == 0:
                template = 'Epoch {}, Step {}, Loss: {:.4f}, Accuracy: {:.4f}'
                print(colored(template.format(
                    epoch, step + 1,
                    train_loss.result(),
                    train_accuracy.result()
                ), 'green'))

        # Validation
        all_predictions = []
        all_labels = []
        for inputs in eval_dist_dataset:
            predictions, labels = distributed_test_step(inputs)
            all_predictions.extend(tf.argmax(predictions, axis=-1).numpy())
            all_labels.extend(labels.numpy())

        # Calculate metrics
        val_f1 = f1_score(y_true=all_labels, y_pred=all_predictions, average='macro')
        val_precision = precision_score(y_true=all_labels, y_pred=all_predictions, average='macro', zero_division=1)
        val_recall = recall_score(y_true=all_labels, y_pred=all_predictions, average='macro')

        # Save best model
        if val_precision > best_accuracy:
            best_accuracy = val_precision
            model.save_weights(os.path.join(dir_current, 'best', 'model'))
        model.save_weights(os.path.join(dir_current, 'last', 'model'))

        # Log results
        template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Time: {:.2f}s'
        print(template.format(
            epoch,
            train_loss.result(),
            train_accuracy.result(),
            val_loss.result(),
            val_accuracy.result(),
            val_f1,
            val_precision,
            val_recall,
            time.time() - start_time
        ))
        log_file.write(template.format(
            epoch,
            train_loss.result(),
            train_accuracy.result(),
            val_loss.result(),
            val_accuracy.result(),
            val_f1,
            val_precision,
            val_recall,
            time.time() - start_time
        ) + '\n')
        log_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spoken_language_identification Model training")
    parser.add_argument("--config_file", type=str, default='./configs/config.yml', help="Config File Path")
    args = parser.parse_args()
    kwargs = vars(args)
    with strategy.scope():
        train(**kwargs)