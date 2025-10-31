"""
Configuration file for Indic Language Augmentation Pipeline
"""

# Language code mappings for IndicTrans2
LANG_CODE_MAP = {
    'hindi': 'hin_Deva',
    'marathi': 'mar_Deva',
    'gujarati': 'guj_Gujr',
    'telugu': 'tel_Telu',
    'bengali': 'ben_Beng',
    'tamil': 'tam_Taml',
    'kannada': 'kan_Knda',
    'malayalam': 'mal_Mlym',
    'punjabi': 'pan_Guru',
    'odia': 'ory_Orya'
}

# Model configurations
MODEL_CONFIGS = {
    'mlm': {
        'model_name': 'google/muril-base-cased',
        'max_length': 512
    },
    'indic_to_en': {
        'model_name': 'ai4bharat/indictrans2-indic-en-1B',
        'max_length': 256
    },
    'en_to_indic': {
        'model_name': 'ai4bharat/indictrans2-en-indic-1B',
        'max_length': 256
    }
}

# Default augmentation parameters
DEFAULT_MLM_PARAMS = {
    'mask_ratio': 0.15,
    'top_k': 5,
    'min_word_length': 2
}

DEFAULT_TRANSLATION_PARAMS = {
    'num_beams': 5,
    'num_return_sequences': 1,
    'max_length': 256
}

# Dataset augmentation defaults
DEFAULT_DATASET_PARAMS = {
    'num_aug_per_row': 1,
    'techniques': ['mlm', 'back_translation'],
    'text_column': 'text',
    'label_column': 'label',
    'language_column': 'language',
    'balance_labels': False
}