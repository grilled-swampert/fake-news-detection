# models/sarvam_1.py
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments
)

MODEL_CHECKPOINT = "sarvamai/sarvam-1"

def get_tokenizer():
    """Returns the tokenizer for Sarvam-1."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    # --- Llama-specific setup ---
    tokenizer.pad_token = tokenizer.eos_token
    # ---
    return tokenizer

def get_model(tokenizer):
    """Returns the model for Sarvam-1."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # --- Llama-specific setup ---
    model.config.pad_token_id = tokenizer.eos_token_id
    # ---
    return model

def get_training_args(output_dir, logging_dir):
    """Returns the OPTIMIZED training arguments for Sarvam-1 (2B model)."""
    # These settings are already a good fit for a 2B model on 24GB VRAM
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        bf16=True,
        dataloader_num_workers=4,         # <-- ADDED
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        torch_compile=True                # <-- ADDED FOR SPEED
    )