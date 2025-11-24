# models/malayalam_bert.py
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments
)

MODEL_CHECKPOINT = "l3cube-pune/malayalam-bert"

def get_tokenizer():
    """Returns the tokenizer for Malayalam-BERT."""
    return AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def get_model(tokenizer):
    """Returns the model for Malayalam-BERT."""
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=2
    )

def get_training_args(output_dir, logging_dir):
    """Returns the OPTIMIZED training arguments for Malayalam-BERT."""
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        num_train_epochs=3,
        per_device_train_batch_size=64,   # <-- INCREASED
        per_device_eval_batch_size=64,
        dataloader_num_workers=4,         # <-- ADDED
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        report_to="none",
        torch_compile=True                # <-- ADDED FOR SPEED
    )