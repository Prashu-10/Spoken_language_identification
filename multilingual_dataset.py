import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor
from featurizers.speech_featurizers import SpeechFeaturizer
from configs.config import Config
from vocab.vocab import Vocab

class MultilingualDataset:
    def __init__(
        self,
        config: Config,
        languages: List[str],
        vocab: Vocab,
        speech_featurizer: SpeechFeaturizer,
        data_type: str = "train",
        max_samples_per_language: Optional[int] = None
    ):
        self.config = config
        self.languages = languages
        self.vocab = vocab
        self.speech_featurizer = speech_featurizer
        self.data_type = data_type
        self.max_samples_per_language = max_samples_per_language
        self.dataset_cache = {}
        
        # Initialize language mapping
        self.language_to_id = {lang: idx for idx, lang in enumerate(languages)}
        self.id_to_language = {idx: lang for lang, idx in self.language_to_id.items()}
        
        # Load FLEURS dataset for multiple languages
        self.load_datasets()

    def load_datasets(self):
        """Load datasets for all specified languages"""
        for lang in tqdm(self.languages, desc="Loading languages"):
            try:
                # Load FLEURS dataset for the language
                dataset = load_dataset("google/fleurs", lang, split=self.data_type)
                
                # Apply sampling if specified
                if self.max_samples_per_language:
                    dataset = dataset.select(range(min(len(dataset), self.max_samples_per_language)))
                
                self.dataset_cache[lang] = dataset
                print(f"Loaded {len(dataset)} samples for {lang}")
            except Exception as e:
                print(f"Error loading dataset for {lang}: {str(e)}")

    def prepare_audio(self, audio_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Process audio data to extract features"""
        if sampling_rate != self.config.speech_config['sample_rate']:
            # Resample if necessary
            import librosa
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sampling_rate, 
                target_sr=self.config.speech_config['sample_rate']
            )
        
        # Extract features using the speech featurizer
        features = self.speech_featurizer.extract(audio_data)
        return features

    def get_batch_generator(self, batch_size: int):
        """Generate batches of data"""
        while True:
            for lang in self.languages:
                dataset = self.dataset_cache[lang]
                
                for i in range(0, len(dataset), batch_size):
                    batch_data = dataset[i:i + batch_size]
                    
                    features_list = []
                    labels = []
                    
                    for item in batch_data:
                        # Process audio
                        audio_data = item['audio']['array']
                        sampling_rate = item['audio']['sampling_rate']
                        features = self.prepare_audio(audio_data, sampling_rate)
                        features_list.append(features)
                        
                        # Get language label
                        labels.append(self.language_to_id[lang])
                    
                    # Pad features to same length
                    max_len = max(feat.shape[0] for feat in features_list)
                    padded_features = np.zeros((len(features_list), max_len, features_list[0].shape[1]))
                    
                    for j, feat in enumerate(features_list):
                        padded_features[j, :feat.shape[0], :] = feat
                    
                    yield {
                        'features': padded_features,
                        'input_lengths': np.array([len(feat) for feat in features_list]),
                        'labels': np.array(labels)
                    }

    def save_language_mapping(self, file_path: str):
        """Save language to ID mapping"""
        with open(file_path, 'w') as f:
            json.dump({
                'language_to_id': self.language_to_id,
                'id_to_language': self.id_to_language
            }, f, indent=2)

    @classmethod
    def load_language_mapping(cls, file_path: str) -> Dict:
        """Load language to ID mapping"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def get_num_languages(self) -> int:
        """Get total number of languages"""
        return len(self.languages)

    def get_language_list(self) -> List[str]:
        """Get list of all languages"""
        return self.languages.copy() 