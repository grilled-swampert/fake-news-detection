# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split

class FakeNewsDataset(TorchDataset):
    """Custom PyTorch Dataset for our news text and labels."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_and_split_data(csv_path, test_size=0.2, random_state=42):
    """Loads CSV, cleans, and splits data."""
    print(f"Loading and splitting the dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Ensure 'text' column is string type and handle potential NaNs
    df['text'] = df['text'].astype(str)
    df.dropna(subset=['text', 'label'], inplace=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )
    print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")
    return train_df, test_df