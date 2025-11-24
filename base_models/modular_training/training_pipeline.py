# training_pipeline.py
import os
import numpy as np
import torch
import datetime  # <-- NEW: To timestamp our logs
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback  # <-- NEW: Import the base class for our callback
)
import dataset # Our custom dataset.py file

# --- NEW: Our Custom File Logger Callback ---
class FileLoggerCallback(TrainerCallback):
    """
    A custom callback that logs metrics to a text file.
    This is triggered every time the Trainer calls `log()`.
    """
    def __init__(self, log_file_path):
        # Make sure the directory for the log file exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        self.log_file_path = log_file_path
        # Write a header to the file when we start
        with open(self.log_file_path, "w") as f:
            f.write(f"--- Training Log Initialized at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the Trainer logs metrics (e.g., every 60 seconds).
        """
        if logs is not None:
            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format the log message
            log_message = f"[{timestamp}] - Step {state.global_step}\n"
            for key, value in logs.items():
                # Format numbers to be readable
                if isinstance(value, float):
                    log_message += f"  {key}: {value:.6f}\n"
                else:
                    log_message += f"  {key}: {value}\n"
            log_message += "-" * 30 + "\n"

            # Append this message to our log file
            try:
                with open(self.log_file_path, "a") as f:
                    f.write(log_message)
            except Exception as e:
                print(f"Error: Could not write to log file {self.log_file_path}. Details: {e}")
# --- END of new Callback ---


# Put the common compute_metrics function here
def compute_metrics(pred):
    """Computes metrics for evaluation."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def run_training(model_name, model_module, csv_path):
    """
    A generic training pipeline that runs by using a 'model_module'.
    ...
    """
    print(f"\n--- 🚀 Starting Training Pipeline for: {model_name} ---")
    
    # --- Configuration ---
    OUTPUT_MODEL_DIR = f"./{model_name}-finetuned-fake-news"
    LOGGING_DIR = f"./logs_{model_name}"
    
    # <-- NEW: Define the path for our text log file
    LOG_TEXT_FILE = f"{LOGGING_DIR}/training_progress.txt"
    print(f"Metrics will be logged to: {LOG_TEXT_FILE}")

    # 1. Load and Split Data (from dataset.py)
    train_df, test_df = dataset.load_and_split_data(csv_path)

    # 2. Get Tokenizer (from the specific model module)
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = model_module.get_tokenizer()

    # 3. Tokenize Data (Shared logic)
    print("Tokenizing training data...")
    train_encodings = tokenizer(
        train_df["text"].tolist(), truncation=True, padding="max_length", max_length=512
    )
    print("Tokenizing testing data...")
    test_encodings = tokenizer(
        test_df["text"].tolist(), truncation=True, padding="max_length", max_length=512
    )

    # 4. Create Datasets (from dataset.py)
    train_dataset = dataset.FakeNewsDataset(train_encodings, train_df["label"].tolist())
    test_dataset = dataset.FakeNewsDataset(test_encodings, test_df["label"].tolist())
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Get Model (from the specific model module)
    print(f"Loading model for {model_name}...")
    model = model_module.get_model(tokenizer=tokenizer) 

    # 6. Get Training Args (from the specific model module)
    print("Loading training arguments...")
    training_args = model_module.get_training_args(OUTPUT_MODEL_DIR, LOGGING_DIR)

    # <-- NEW: Create an instance of our file logger
    my_file_logger = FileLoggerCallback(log_file_path=LOG_TEXT_FILE)

    # 7. Create Trainer (Shared logic)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[my_file_logger]  # <-- NEW: Pass our callback to the Trainer
    )

    # 8. Train! (Shared logic)
    print("\nStarting model training...")
    trainer.train()

    # 9. Save and Evaluate (Shared logic)
    print("\nTraining complete. Saving the best model...")
    trainer.save_model(OUTPUT_MODEL_DIR)
    print(f"✓ Model saved to {OUTPUT_MODEL_DIR}")

    print("\nEvaluating the final model on the test set...")
    eval_results = trainer.evaluate()
    print(f"Final Evaluation Results: {eval_results}")
    
    # <-- NEW: Write final results to the log file too
    with open(LOG_TEXT_FILE, "a") as f:
        f.write("\n--- FINAL EVALUATION ---\n")
        f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(str(eval_results))
        f.write("\n--------------------------\n")
        
    print(f"--- ✅ Pipeline for {model_name} Finished ---")