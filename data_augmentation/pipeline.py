"""
Main pipeline class that ties everything together
"""

import pandas as pd
import warnings
from typing import List, Optional

from device_utils import detect_device, print_device_info
from model_loader import (
    load_mlm_model,
    load_indic_to_en_model,
    load_en_to_indic_model,
    load_indic_processor
)
from mlm_augmenter import MLMAugmenter
from translation_augmenter import TranslationAugmenter
from dataset_augmenter import DatasetAugmenter
from config import LANG_CODE_MAP

warnings.filterwarnings('ignore')


class IndicAugmentationPipeline:
    """
    Main pipeline for Indic language data augmentation
    Combines MLM and back-translation techniques
    """

    def __init__(self, device: str = None):
        """
        Initialize augmentation pipeline

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Detect and setup device
        self.device = detect_device(device)
        print_device_info(self.device)

        print("\n" + "=" * 70)
        print("Loading Models")
        print("=" * 70 + "\n")

        # Load all models and components
        self._load_all_models()

        # Initialize augmenters
        self._initialize_augmenters()

        print("\n✅ Pipeline initialized successfully!\n")

    def _load_all_models(self):
        """Load all required models"""
        try:
            # Load MLM model
            (self.mlm_tokenizer,
             self.mlm_model,
             self.device) = load_mlm_model(self.device)

            # Load IndicProcessor
            self.ip = load_indic_processor()

            # Load translation models
            (self.trans_tokenizer_in_en,
             self.trans_model_in_en) = load_indic_to_en_model(self.device)

            (self.trans_tokenizer_en_in,
             self.trans_model_en_in) = load_en_to_indic_model(self.device)

        except Exception as e:
            print(f"\n❌ Error loading models: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Check internet connection (models download on first run)")
            print("   2. Ensure you have ~8GB free disk space")
            print("   3. Try: pip install --upgrade transformers torch")
            raise

    def _initialize_augmenters(self):
        """Initialize augmenter components"""
        # MLM augmenter
        self.mlm_augmenter = MLMAugmenter(
            self.mlm_tokenizer,
            self.mlm_model,
            self.device
        )

        # Translation augmenter (if processor available)
        if self.ip is not None:
            self.translation_augmenter = TranslationAugmenter(
                self.trans_tokenizer_in_en,
                self.trans_model_in_en,
                self.trans_tokenizer_en_in,
                self.trans_model_en_in,
                self.device,
                self.ip
            )
        else:
            self.translation_augmenter = None
            print("⚠️  Back translation will be disabled")

        # Dataset augmenter
        self.dataset_augmenter = DatasetAugmenter(
            self.mlm_augmenter,
            self.translation_augmenter
        )

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def augment_with_mlm(self, text: str, mask_ratio: float = 0.15,
                         top_k: int = 5) -> str:
        """
        Augment text using Masked Language Modeling

        Args:
            text: Input text
            mask_ratio: Ratio of words to mask (0.15 = 15%)
            top_k: Number of top predictions to consider

        Returns:
            Augmented text
        """
        return self.mlm_augmenter.augment(text, mask_ratio, top_k)

    def augment_with_back_translation(self, text: str, lang_code: str,
                                      verbose: bool = False) -> str:
        """
        Augment text using back translation (Indic → English → Indic)

        Args:
            text: Input text
            lang_code: Language code (e.g., 'hin_Deva')
            verbose: Print intermediate translation steps

        Returns:
            Back-translated text
        """
        if self.translation_augmenter is None:
            print("⚠️  Back translation not available (no IndicProcessor)")
            return text

        return self.translation_augmenter.augment(text, lang_code, verbose)

    def map_language(self, lang: str) -> Optional[str]:
        """
        Map dataset language name to model language code

        Args:
            lang: Language name (e.g., 'hindi', 'marathi')

        Returns:
            Language code (e.g., 'hin_Deva') or None
        """
        return self.dataset_augmenter.map_language(lang)

    def create_augmented_dataset(self,
                                 df: pd.DataFrame,
                                 num_aug_per_row: int = 1,
                                 techniques: List[str] = None,
                                 text_column: str = 'text',
                                 label_column: str = 'label',
                                 language_column: str = 'language',
                                 balance_labels: bool = False) -> pd.DataFrame:
        """
        Create augmented dataset with multiple techniques

        Args:
            df: Input DataFrame
            num_aug_per_row: Number of augmentations per sample
            techniques: List of techniques ['mlm', 'back_translation']
                       or None (auto-detect)
            text_column: Name of text column
            label_column: Name of label column
            language_column: Name of language column
            balance_labels: Augment minority classes more

        Returns:
            Augmented DataFrame
        """
        return self.dataset_augmenter.augment_dataset(
            df=df,
            num_aug_per_row=num_aug_per_row,
            techniques=techniques,
            text_column=text_column,
            label_column=label_column,
            language_column=language_column,
            balance_labels=balance_labels
        )

    def test_augmentation(self, text: str, language: str,
                          n_samples: int = 3):
        """
        Test augmentation on a single text

        Args:
            text: Sample text
            language: Language name (e.g., 'hindi', 'marathi')
            n_samples: Number of augmented samples to generate
        """
        lang_code = self.map_language(language)
        if not lang_code:
            print(f"❌ Unknown language: {language}")
            print(f"   Supported: {', '.join(LANG_CODE_MAP.keys())}")
            return

        print(f"🔬 Testing augmentation on: '{text}'\n")
        print(f"Language: {language} ({lang_code})\n")

        print("=" * 70)
        print("MLM Augmentation:")
        print("=" * 70)
        for i in range(n_samples):
            aug = self.augment_with_mlm(text)
            print(f"{i + 1}. {aug}")

        if self.translation_augmenter is not None:
            print("\n" + "=" * 70)
            print("Back Translation:")
            print("=" * 70)
            for i in range(n_samples):
                aug = self.augment_with_back_translation(
                    text, lang_code, verbose=True
                )
                print(f"{i + 1}. {aug}\n")
        else:
            print("\n⚠️  Back translation not available")

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(LANG_CODE_MAP.keys())

    def get_available_techniques(self) -> List[str]:
        """Get list of available augmentation techniques"""
        techniques = ["mlm"]
        if self.translation_augmenter is not None:
            techniques.append("back_translation")
        return techniques