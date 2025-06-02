import os
import json
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from multilingual_dataset import MultilingualDataset
from featurizers.speech_featurizers import NumpySpeechFeaturizer
from configs.config import Config
from vocab.vocab import Vocab

def setup_mixed_precision():
    """Setup mixed precision for better performance on ARM32"""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

class ExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

class SqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

def create_model(config, num_languages):
    """Create the model with support for multiple languages"""
    inputs = tf.keras.Input(shape=(None, config.speech_config['num_feature_bins']))
    x = inputs
    
    # CNN layers
    for filters, kernel in zip(config.model_config['filters'], config.model_config['kernel_size']):
        x = ExpandDimsLayer(axis=-1)(x)
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            padding='same',
            activation='relu'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = SqueezeLayer(axis=-1)(x)
    
    # BiLSTM layers
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config.model_config['rnn_cell'],
            return_sequences=True
        )
    )(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_languages, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # Load configuration
    config = Config("configs/config.yml")
    
    # Setup mixed precision if enabled
    if config.optimizer_config.get('use_mixed_precision', False):
        setup_mixed_precision()
    
    # Load language configuration
    with open(config.dataset_config['languages_file'], 'r') as f:
        languages_config = json.load(f)
    languages = languages_config['supported_languages']
    
    # Initialize components
    vocab = Vocab(config.dataset_config['vocabulary'])
    speech_featurizer = NumpySpeechFeaturizer(config.speech_config)
    
    # Create datasets
    train_dataset = MultilingualDataset(
        config=config,
        languages=languages,
        vocab=vocab,
        speech_featurizer=speech_featurizer,
        data_type='train',
        max_samples_per_language=config.dataset_config['max_samples_per_language']
    )
    
    val_dataset = MultilingualDataset(
        config=config,
        languages=languages,
        vocab=vocab,
        speech_featurizer=speech_featurizer,
        data_type='validation',
        max_samples_per_language=config.dataset_config['max_samples_per_language'] // 10
    )
    
    # Create model
    model = create_model(config, len(languages))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.optimizer_config['max_lr'],
        beta_1=config.optimizer_config['beta1'],
        beta_2=config.optimizer_config['beta2'],
        epsilon=config.optimizer_config['epsilon']
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join('saved_weights', 'multilingual', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, 'epoch_{epoch:02d}'),
            save_weights_only=True,
            save_freq='epoch'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('logs', timestamp),
            update_freq='batch'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    train_generator = train_dataset.get_batch_generator(config.running_config['batch_size'])
    val_generator = val_dataset.get_batch_generator(config.running_config['batch_size'])
    
    steps_per_epoch = config.running_config['train_steps']
    validation_steps = config.running_config['dev_steps']
    
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=config.running_config['num_epochs'],
        callbacks=callbacks
    )
    
    # Save final model
    model.save_weights(os.path.join(save_dir, 'final'))
    
    # Save language mapping
    train_dataset.save_language_mapping(os.path.join(save_dir, 'language_mapping.json'))

if __name__ == '__main__':
    main() 