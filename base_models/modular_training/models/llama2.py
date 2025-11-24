# models/llama2.py
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments
)

MODEL_CHECKPOINT = "meta-llama/Llama-2-7b-hf"

def get_tokenizer():
    """Returns the tokenizer for Llama 2."""
    print("NOTE: You must be logged in via 'huggingface-cli login' to download Llama 2.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # --- Llama-specific setup ---
    tokenizer.pad_token = tokenizer.eos_token
    # ---
    return tokenizer

def get_model(tokenizer):
    """Returns the model for Llama 2, configured with the tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, 
        num_labels=2,
        torch_dtype=torch.bfloat16, # Use bfloat16 for large models
        device_map="auto"           # Automatically distribute model
    )
    
    # --- Llama-specific setup ---
    model.config.pad_token_id = tokenizer.eos_token_id
    # ---
    return model

def get_training_args(output_dir, logging_dir):
    """Returns the OPTIMIZED training arguments for Llama 2 (7B model)."""
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,    # <-- INCREASED
        per_device_eval_batch_size=8,   # <-- Increased eval batch
        gradient_accumulation_steps=4,  # <-- DECREASED
        bf16=True,                      # Use bf16 for Llama
        dataloader_num_workers=4,         # <-- ADDED
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        torch_compile=True                # <-- ADDED FOR SPEED
    )