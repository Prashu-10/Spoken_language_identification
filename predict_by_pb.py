import os
# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Must import tensorflow after setting environment variables
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Using CPU only")

from vocab.vocab import Vocab
import librosa
import numpy as np
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def load_model(model_path):
    try:
        # Load model in CPU mode
        with tf.device('/CPU:0'):
            return tf.saved_model.load(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_wav(wav_path, model, vocab):
    try:
        # Load and preprocess audio
        signal, _ = librosa.load(wav_path, sr=16000)
        
        # Convert to tensor and ensure CPU operation
        with tf.device('/CPU:0'):
            signal = tf.convert_to_tensor(signal, dtype=tf.float32)
            
            # Make prediction
            if hasattr(model, 'predict_pb'):
                output = model.predict_pb(signal)
            else:
                output = model(signal)
            
            if isinstance(output, dict):
                pred = output.get("output_0", None)
                prob = output.get("output_1", None)
            elif isinstance(output, (list, tuple)) and len(output) == 2:
                pred, prob = output
            else:
                print("Unexpected model output format")
                return None, None
            
            # Get prediction
            pred_idx = tf.argmax(pred).numpy()
            probability = tf.reduce_max(tf.nn.softmax(prob)).numpy()
        
        language = vocab.token_list[pred_idx]
        print(f"Detected language: {language} (confidence: {probability*100:.2f}%)")
        
        return pred_idx, probability
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

if __name__ == '__main__':
    try:
        # Initialize vocabulary
        vocab = Vocab("vocab/vocab.txt")
        
        # Load model
        print("Loading model...")
        model = load_model('saved_models/lang14/pb/2/')
        
        if model is None:
            print("Failed to load model. Exiting.")
            sys.exit(1)
            
        # Make prediction
        print("Making prediction...")
        predict_wav("test_audios/french.wav", model, vocab)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the model file exists in saved_models/lang14/pb/2/")
        print("2. Make sure the vocabulary file exists in vocab/vocab.txt")
        print("3. Make sure the audio file exists in test_audios/french.wav")
