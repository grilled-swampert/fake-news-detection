# prepare_data_muril..py

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

# --- IMPORTANT: UPDATE THESE PATHS ---
DATASET_FOLDER = Path(r"E:\F & R News Dataset")  # The main folder containing language folders
OUTPUT_CSV_FILE = Path(r"E:\F & R News Dataset\all_news_data.csv") # Where to save the final CSV

def read_txt_file(file_path):
    """Reads a text file with multiple encoding fallbacks."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except Exception:
            continue
    print(f"Warning: Could not decode {file_path}")
    return None

def get_folder_pairs(dataset_folder):
    """Finds pairs of fake/real news folders for each language."""
    language_folders = {}
    for folder in dataset_folder.iterdir():
        if folder.is_dir() and folder.name != "__MACOSX" and "features" not in folder.name:
            try:
                language = folder.name.split('_')[0].capitalize()
                if language not in language_folders:
                    language_folders[language] = {}
                
                if 'fake' in folder.name.lower():
                    language_folders[language]['fake'] = folder
                elif 'real' in folder.name.lower():
                    language_folders[language]['real'] = folder
            except IndexError:
                print(f"Skipping folder with unexpected name: {folder.name}")
                continue
    
    # Return only languages that have both fake and real folders
    return {lang: paths for lang, paths in language_folders.items() if 'fake' in paths and 'real' in paths}

def main():
    """Main function to process files and create the CSV."""
    folder_pairs = get_folder_pairs(DATASET_FOLDER)
    
    if not folder_pairs:
        print("Error: No valid fake/real folder pairs found. Check your folder names.")
        return

    print("Found the following language pairs:")
    for lang, paths in folder_pairs.items():
        print(f" - {lang}: {paths['fake'].name} & {paths['real'].name}")
        
    all_data = []

    for language, paths in folder_pairs.items():
        # Process fake news (label=0)
        fake_folder = paths['fake']
        print(f"\nProcessing {language} Fake News...")
        for txt_file in tqdm(list(fake_folder.glob('*.txt')), desc=f"{language} Fake"):
            text = read_txt_file(txt_file)
            if text:
                all_data.append({
                    'text': text,
                    'label': 0, # 0 for Fake
                    'language': language.lower()
                })

        # Process real news (label=1)
        real_folder = paths['real']
        print(f"\nProcessing {language} Real News...")
        for txt_file in tqdm(list(real_folder.glob('*.txt')), desc=f"{language} Real"):
            text = read_txt_file(txt_file)
            if text:
                all_data.append({
                    'text': text,
                    'label': 1, # 1 for Real
                    'language': language.lower()
                })

    # Create and save DataFrame
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1).reset_index(drop=True) # Shuffle the dataset

    print(f"\nCreated a DataFrame with {len(df)} total samples.")
    print("Label distribution:")
    print(df['label'].value_counts())
    
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
    print(f"\n✓ Successfully saved all data to: {OUTPUT_CSV_FILE}")

if __name__ == "__main__":
    main()