"""
Model loading utilities
"""

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)
from config import MODEL_CONFIGS
from device_utils import move_model_to_device


def load_mlm_model(device: str):
    """
    Load MuRIL MLM model and tokenizer

    Args:
        device: Target device

    Returns:
        Tuple of (tokenizer, model, actual_device)
    """
    print("📦 Loading MuRIL (MLM) model...")

    config = MODEL_CONFIGS['mlm']

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForMaskedLM.from_pretrained(
        config['model_name'],
        ignore_mismatched_sizes=True
    )

    model, actual_device = move_model_to_device(model, device, "MLM model")
    model.eval()

    print(f"   ✅ MuRIL loaded on {actual_device}")
    return tokenizer, model, actual_device


def load_indic_to_en_model(device: str):
    """
    Load IndicTrans2 Indic→English model

    Args:
        device: Target device

    Returns:
        Tuple of (tokenizer, model)
    """
    print("📦 Loading IndicTrans2 (Indic→En) model...")

    config = MODEL_CONFIGS['indic_to_en']

    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config['model_name'],
        trust_remote_code=True
    )

    model = model.to(device)
    model.eval()

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   ✅ Indic→En model loaded")
    return tokenizer, model


def load_en_to_indic_model(device: str):
    """
    Load IndicTrans2 English→Indic model

    Args:
        device: Target device

    Returns:
        Tuple of (tokenizer, model)
    """
    print("📦 Loading IndicTrans2 (En→Indic) model...")

    config = MODEL_CONFIGS['en_to_indic']

    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config['model_name'],
        trust_remote_code=True
    )

    model = model.to(device)
    model.eval()

    print(f"   ✅ En→Indic model loaded")
    return tokenizer, model


def load_indic_processor():
    """
    Load IndicProcessor with fallback

    Returns:
        IndicProcessor instance or None
    """
    try:
        from IndicTransToolkit.processor import IndicProcessor
        print("📦 Initializing IndicProcessor...")
        ip = IndicProcessor(inference=True)
        print("   ✅ IndicProcessor initialized")
        return ip
    except ImportError:
        print("⚠️  IndicTransToolkit not available")
        print("   Back translation will use fallback preprocessing")
        return None
    except Exception as e:
        print(f"⚠️  IndicProcessor initialization failed: {e}")
        print("   Back translation will use fallback preprocessing")
        return None