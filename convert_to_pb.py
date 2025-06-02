from vocab.vocab import Vocab
from numpy import genfromtxt
import tensorflow as tf
from configs.config import Config
from models.model import MulSpeechLR as Model
from featurizers.speech_featurizers import TFSpeechFeaturizer

import numpy as np


weights_dir = './saved_weights/lang14/20230228-084356/'
config_file = weights_dir + 'config.yml'
vocabulary =  weights_dir + 'vocab.txt'

config = Config(config_file)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)
librosa_mel_filter = genfromtxt('librosa_mel_filter.csv', delimiter=',')
librosa_mel_filter = np.asarray(librosa_mel_filter, dtype=np.float32)
speech_featurizer.set_mel_filter(librosa_mel_filter)

vocab = Vocab(vocabulary)

# Build model
model = Model(**config.model_config, vocab_size=len(vocab.token_list))
model.init_build([None, config.speech_config['num_feature_bins']]) 
model.load_weights(weights_dir + "last/model")
model.add_featurizers(speech_featurizer)

version = 2

# Convert to SavedModel format with signatures
@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
def predict_fn(signal):
    output, prob = model.predict_pb(signal)
    return {"output_0": output, "output_1": prob}

# Save model with proper signatures
tf.saved_model.save(
    model,
    f"saved_models/lang14/pb/{version}",
    signatures={
        "serving_default": predict_fn,
        "predict_pb": model.predict_pb
    }
)
print('Model converted to SavedModel format successfully')

# Save model for TensorFlow Serving
tf.saved_model.save(
    model,
    f"saved_models/lang14/serving/{version}",
    signatures={
        "serving_default": predict_fn,
        "predict_pb": model.predict_pb
    }
)
print('Model converted for TensorFlow Serving successfully')
