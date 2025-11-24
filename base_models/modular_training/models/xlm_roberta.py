from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments
)

MODEL_CHECKPOINT = "xlm-roberta-base"

def get_tokenizer():
    """Returns the tokenizer for XLM-RoBERTa."""
    return AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def get_model(tokenizer):
    """Returns the model for XLM-RoBERTa."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2
    )

def get_training_args(output_dir, logging_dir):
    """Returns the OPTIMIZED training arguments for XLM-RoBERTa."""
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=3,
        per_device_train_batch_size=64,  # Good for your A5000
        per_device_eval_batch_size=64,
        dataloader_num_workers=4,
        warmup_steps=500,
        weight_decay=0.01,
        
        # --- KEY FIX ---
        logging_strategy="epoch",       # <-- FIX: Changed from "seconds" to "epoch"
        # logging_steps=60,             # This is now ignored (which is fine)
        logging_first_step=True,        # Log the metrics at the very first step
        report_to="tensorboard",
        # --- END OF KEY FIX ---
        
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,                      # Perfect for your A5000
        torch_compile=False              # Great for speed
    )