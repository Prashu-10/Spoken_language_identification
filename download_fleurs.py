import os
import json
import argparse
import shutil
import time
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
import soundfile as sf
import numpy as np
from pathlib import Path

# All FLEURS languages
ALL_LANGUAGES = [
    'af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 
    'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'ga', 'gl', 'gu', 'ha', 'he', 
    'hi', 'hr', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 
    'ko', 'ky', 'lb', 'lg', 'ln', 'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 
    'my', 'ne', 'nl', 'no', 'ny', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'rw', 'sd', 
    'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 
    'tk', 'tr', 'uk', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zu'
]

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_audio(audio_data, sample_rate, output_path):
    """Save audio data to WAV file"""
    sf.write(output_path, audio_data, sample_rate)

def download_language(lang, output_dir, splits=None, retry_count=3, retry_delay=5):
    """Download and organize dataset for a specific language with retries"""
    if splits is None:
        splits = ['train', 'validation', 'test']
    
    lang_dir = os.path.join(output_dir, lang)
    print(f"\nProcessing language: {lang}")
    
    for split in splits:
        print(f"\nDownloading {split} split...")
        split_dir = os.path.join(lang_dir, split)
        audio_dir = os.path.join(split_dir, 'audio')
        
        # Skip if already downloaded
        metadata_path = os.path.join(split_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            print(f"Skipping {lang} {split} - already downloaded")
            continue
            
        ensure_dir(audio_dir)
        
        # Load dataset with retries
        dataset = None
        for attempt in range(retry_count):
            try:
                dataset = load_dataset("google/fleurs", lang, split=split)
                break
            except Exception as e:
                if attempt < retry_count - 1:
                    print(f"Attempt {attempt + 1} failed for {lang} {split}: {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Error downloading {lang} {split} after {retry_count} attempts: {str(e)}")
                    return False
        
        if dataset is None:
            continue
        
        # Prepare metadata
        metadata = {
            'data': [],
            'lang': lang,
            'split': split
        }
        
        # Process each example
        for idx, item in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            try:
                # Extract audio
                audio_data = item['audio']['array']
                sample_rate = item['audio']['sampling_rate']
                
                # Generate ID
                item_id = f"{lang}_{split}_{idx:06d}"
                
                # Save audio file
                audio_path = os.path.join(audio_dir, f"{item_id}.wav")
                save_audio(audio_data, sample_rate, audio_path)
                
                # Add to metadata
                metadata['data'].append({
                    'id': item_id,
                    'transcription': item.get('transcription', ''),
                    'raw_transcription': item.get('raw_transcription', ''),
                    'language': item.get('language', lang),
                    'gender': item.get('gender', ''),
                    'lang_id': item.get('lang_id', -1)
                })
            
            except Exception as e:
                print(f"Error processing item {idx} in {lang} {split}: {str(e)}")
                continue
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(metadata['data'])} examples for {lang} {split}")
    
    return True

def download_languages_in_batches(languages, output_dir, batch_size=5, splits=None):
    """Download languages in batches to manage memory usage"""
    total_languages = len(languages)
    successful = []
    failed = []
    
    for i in range(0, total_languages, batch_size):
        batch = languages[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} of {(total_languages + batch_size - 1)//batch_size}")
        print(f"Languages in this batch: {', '.join(batch)}")
        
        for lang in batch:
            try:
                if download_language(lang, output_dir, splits):
                    successful.append(lang)
                else:
                    failed.append(lang)
            except Exception as e:
                print(f"Failed to download {lang}: {str(e)}")
                failed.append(lang)
        
        # Clear some memory
        if i + batch_size < total_languages:
            print("\nClearing memory before next batch...")
            time.sleep(5)  # Give some time for memory cleanup
    
    return successful, failed

def main():
    parser = argparse.ArgumentParser(description='Download and organize FLEURS dataset')
    parser.add_argument('--output_dir', type=str, default='./data/fleurs',
                        help='Output directory for the dataset')
    parser.add_argument('--languages', type=str, nargs='+',
                        help='List of language codes to download (default: all languages)')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'validation', 'test'],
                        help='Dataset splits to download')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Number of languages to download in parallel')
    args = parser.parse_args()
    
    # Use all languages if none specified
    languages = args.languages if args.languages else ALL_LANGUAGES
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Download languages in batches
    print(f"Starting download of {len(languages)} languages in batches of {args.batch_size}")
    successful, failed = download_languages_in_batches(
        languages, args.output_dir, args.batch_size, args.splits
    )
    
    # Print summary
    print("\n=== Download Summary ===")
    print(f"Successfully downloaded: {len(successful)} languages")
    print(f"Failed to download: {len(failed)} languages")
    
    if failed:
        print("\nFailed languages:")
        print(", ".join(failed))
        
        # Save failed languages to file for retry
        failed_file = os.path.join(args.output_dir, "failed_languages.txt")
        with open(failed_file, 'w') as f:
            f.write("\n".join(failed))
        print(f"\nFailed languages list saved to: {failed_file}")
        print("You can retry failed languages using:")
        print(f"python download_fleurs.py --languages {' '.join(failed)}")

if __name__ == '__main__':
    main() 