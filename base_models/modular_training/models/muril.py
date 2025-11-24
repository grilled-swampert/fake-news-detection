# models/muril.py
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments
)

MODEL_CHECKPOINT = "google/muril-base-cased"

def get_tokenizer():
    """Returns the tokenizer for MuRIL."""
    return AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def get_model(tokenizer):
    """Returns the model for MuRIL."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2
    )

def get_training_args(output_dir, logging_dir):
    """Returns the OPTIMIZED training arguments for MuRIL."""
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        dataloader_num_workers=4,
        warmup_steps=500,
        weight_decay=0.01,

        # --- LOGGING UPDATES (Matching XLM-RoBERTa) ---
        logging_strategy="epoch",       # Change: Log once per epoch (instead of steps)
        logging_first_step=True,        # Change: Log the very first step immediately
        report_to="tensorboard",        # Change: Enabled Tensorboard (was "none")
        # logging_steps=100,            # Removed: Ignored when strategy is "epoch"
        # ----------------------------------------------

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,                      # Optimized for A5000
        torch_compile=False              # Kept as True (faster, if it works for you)
    )