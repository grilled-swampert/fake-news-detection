# train_muril.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os 
import torch
from torch.utils.data import Dataset as TorchDataset # Added PyTorch Dataset

# --- Hardware & Resource Configuration ---
# Allow PyTorch to use TF32 on Ampere GPUs (like your A5000) for a speed boost
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# <<< NOTE: NUM_PROC is no longer used for tokenization, 
# but dataloader_num_workers (in TrainingArguments) is still used.
print(f"Using {os.cpu_count()} total CPU threads.")

# --- Configuration ---
MODEL_CHECKPOINT = "google/muril-base-cased"
CSV_FILE_PATH = r"D:\fake-news-detection\dataset\all_news_data.csv" # Path to the CSV from Step 1
OUTPUT_MODEL_DIR = "./muril-finetuned-fake-news" # Where to save the final model

# 1. Load and Split the Dataset
print("Loading and splitting the dataset...")
df = pd.read_csv(CSV_FILE_PATH)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Training samples: {len(train_df)}, Testing samples: {len(test_df)}")

# 2. Load Tokenizer and Preprocess Data
print("Loading tokenizer and preprocessing data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# <<< MODIFIED: Tokenize DataFrames directly (This is now single-threaded)
print("Tokenizing training data...")
train_encodings = tokenizer(
    train_df['text'].tolist(), 
    truncation=True, 
    padding="max_length", 
    max_length=512
)
print("Tokenizing testing data...")
test_encodings = tokenizer(
    test_df['text'].tolist(), 
    truncation=True, 
    padding="max_length", 
    max_length=512
)

# Get labels
train_labels = train_df['label'].tolist()
test_labels = test_df['label'].tolist()


# <<< MODIFIED: Define a custom PyTorch Dataset class
class FakeNewsDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Create item dictionary
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# <<< MODIFIED: Create instances of our new Dataset class
train_dataset = FakeNewsDataset(train_encodings, train_labels)
test_dataset = FakeNewsDataset(test_encodings, test_labels)

# 3. Define the Model and Metrics
print("Defining model and metrics...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 4. Define Training Arguments
# (Arguments are unchanged, dataloader_num_workers is still beneficial)
training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=32, 
    per_device_eval_batch_size=64,
    dataloader_num_workers=8, 
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    
    # <<< MODIFIED: Renamed argument for older transformers version
    eval_strategy="epoch",
    
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
)

# <<< MODIFIED: Define the data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Initialize and Run Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator  # <<< MODIFIED: Added the collator
)

print("\nStarting model training...")
trainer.train()

# 6. Save the final model
print("\nTraining complete. Saving the best model...")
trainer.save_model(OUTPUT_MODEL_DIR)
print(f"✓ Model saved to {OUTPUT_MODEL_DIR}")

# 7. Evaluate the final model
print("\nEvaluating the final model on the test set...")
eval_results = trainer.evaluate()
print(f"Final Evaluation Results: {eval_results}")