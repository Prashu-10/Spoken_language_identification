import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from datasets import load_dataset, Dataset, load_from_disk
from huggingface_hub import HfFileSystem
from featurizers.speech_featurizers import NumpySpeechFeaturizer
from configs.config import Config
from vocab.vocab import Vocab

class MultilingualDataset:
    def __init__(
        self,
        config: Config,
        languages: List[str],
        vocab: Vocab,
        speech_featurizer: NumpySpeechFeaturizer,
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

    def find_dataset_path(self, lang: str) -> Optional[str]:
        """Find the dataset path in local hub/datasets structure or HuggingFace cache"""
        # First try local data directory
        local_dir = os.path.join("data", "fleurs", "hub", "datasets--google--fluers")
        if os.path.exists(local_dir):
            # Look for blob directories
            blob_dirs = [d for d in os.listdir(local_dir) if d.startswith("blobs")]
            if blob_dirs:
                # Use the first blob directory found
                dataset_path = os.path.join(local_dir, blob_dirs[0], lang)
                if os.path.exists(dataset_path):
                    print(f"Found local dataset for {lang} at {dataset_path}")
                    return dataset_path

        # Fallback to HuggingFace cache
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        dataset_dir = os.path.join(cache_dir, "google-fleurs", lang)
        
        if not os.path.exists(dataset_dir):
            print(f"Dataset directory not found for language {lang}")
            return None
            
        # Look for the downloaded version
        versions = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        if not versions:
            print(f"No dataset versions found for language {lang}")
            return None
            
        # Use the latest version
        latest_version = sorted(versions)[-1]
        dataset_path = os.path.join(dataset_dir, latest_version)
        
        return dataset_path if os.path.exists(dataset_path) else None

    def load_local_dataset(self, lang: str) -> Optional[Dataset]:
        """Load dataset from HuggingFace cache"""
        try:
            # Find the dataset path
            dataset_path = self.find_dataset_path(lang)
            if dataset_path is None:
                return None
                
            # Load the dataset
            dataset = load_from_disk(dataset_path)
            
            # Get the appropriate split
            if self.data_type in dataset:
                return dataset[self.data_type]
            else:
                print(f"Split {self.data_type} not found for language {lang}")
                return None
                
        except Exception as e:
            print(f"Error loading dataset for language {lang}: {str(e)}")
            return None

    def load_datasets(self):
        """Load datasets for all specified languages"""
        for lang in tqdm(self.languages, desc="Loading languages"):
            try:
                # Load local dataset for the language
                dataset = self.load_local_dataset(lang)
                
                if dataset is None:
                    continue
                
                # Apply sampling if specified
                if self.max_samples_per_language:
                    dataset = dataset.select(range(min(len(dataset), self.max_samples_per_language)))
                
                self.dataset_cache[lang] = dataset
                print(f"Loaded {len(dataset)} samples for {lang}")
            except Exception as e:
                print(f"Error loading dataset for {lang}: {str(e)}")

    def prepare_audio(self, audio_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Process audio data to extract features using NumPy-based extraction"""
        if sampling_rate != self.config.speech_config['sample_rate']:
            # Resample if necessary
            import librosa
            audio_data = librosa.resample(
                audio_data, 
                orig_sr=sampling_rate, 
                target_sr=self.config.speech_config['sample_rate']
            )
        
        # Extract features using NumPy-based feature extraction
        features = self.speech_featurizer.extract(audio_data)
        return features

    def get_batch_generator(self, batch_size: int):
        """Generate batches of data"""
        while True:
            for lang in self.languages:
                if lang not in self.dataset_cache:
                    continue
                    
                dataset = self.dataset_cache[lang]
                indices = list(range(len(dataset)))
                
                if self.config.dataset_config.get('shuffle', True):
                    np.random.shuffle(indices)
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_data = dataset.select(batch_indices)
                    
                    features_list = []
                    labels = []
                    
                    for item in batch_data:
                        try:
                            # Process audio
                            audio_data = item['audio']['array']
                            sampling_rate = item['audio']['sampling_rate']
                            features = self.prepare_audio(audio_data, sampling_rate)
                            features_list.append(features)
                            
                            # Get language label
                            labels.append(self.language_to_id[lang])
                        except Exception as e:
                            print(f"Error processing item in {lang}: {str(e)}")
                            continue
                    
                    if not features_list:
                        continue
                    
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