"""
Advanced Indic Language Data Augmentation Pipeline with IndicTrans2
Fixed version with alternative preprocessing approach

INSTALLATION:
    pip install torch transformers sentencepiece pandas numpy
    pip install IndicTransToolkit  # CRITICAL: Required for IndicTrans2

    For GPU support (optional, 10-20x faster):
    pip install torch --index-url https://download.pytorch.org/whl/cu118
"""

import torch
import random
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import IndicProcessor, but have fallback
try:
    from IndicTransToolkit.processor import IndicProcessor
    INDIC_PROCESSOR_AVAILABLE = True
except ImportError:
    print("⚠️  IndicTransToolkit not available, back translation will be disabled")
    INDIC_PROCESSOR_AVAILABLE = False


class IndicAugmentationPipeline:
    def __init__(self, device: str = None):
        """
        Initialize all models for Indic language augmentation

        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Smart device detection
        if device:
            self.device = device
        else:
            try:
                if torch.cuda.is_available():
                    torch.zeros(1).cuda()
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            except (AssertionError, RuntimeError):
                print("⚠️  CUDA not available, falling back to CPU")
                self.device = 'cpu'

        print(f"🚀 Initializing models on device: {self.device}")
        if self.device == 'cpu':
            print("💡 Tip: Install CUDA-enabled PyTorch for 10-20x speedup")

        # Language mapping
        self.lang_code_map = {
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

        self._load_models()

    def _load_models(self):
        """Load all required models"""
        try:
            print("Loading MuRIL (MLM) model...")
            self.mlm_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
            self.mlm_model = AutoModelForMaskedLM.from_pretrained(
                "google/muril-base-cased",
                ignore_mismatched_sizes=True
            )

            if self.device == 'cuda':
                try:
                    self.mlm_model = self.mlm_model.to(self.device)
                except (AssertionError, RuntimeError) as e:
                    print(f"Could not move MLM model to CUDA: {e}")
                    print("   Falling back to CPU...")
                    self.device = 'cpu'
                    self.mlm_model = self.mlm_model.to('cpu')
            else:
                self.mlm_model = self.mlm_model.to(self.device)

            self.mlm_model.eval()

            # Initialize IndicProcessor if available
            if INDIC_PROCESSOR_AVAILABLE:
                print("📦 Initializing IndicProcessor...")
                try:
                    self.ip = IndicProcessor(inference=True)
                    print("✅ IndicProcessor initialized successfully")
                except Exception as e:
                    print(f"⚠️  IndicProcessor initialization failed: {e}")
                    print("   Back translation will be disabled")
                    self.ip = None
            else:
                self.ip = None

            print("📦 Loading IndicTrans2 (Indic→En) model...")
            self.trans_tokenizer_in_en = AutoTokenizer.from_pretrained(
                "ai4bharat/indictrans2-indic-en-1B",
                trust_remote_code=True
            )
            self.trans_model_in_en = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/indictrans2-indic-en-1B",
                trust_remote_code=True
            )
            self.trans_model_in_en.to(self.device)
            self.trans_model_in_en.eval()

            if self.trans_tokenizer_in_en.pad_token is None:
                self.trans_tokenizer_in_en.pad_token = self.trans_tokenizer_in_en.eos_token

            print("📦 Loading IndicTrans2 (En→Indic) model...")
            self.trans_tokenizer_en_in = AutoTokenizer.from_pretrained(
                "ai4bharat/indictrans2-en-indic-1B",
                trust_remote_code=True
            )
            self.trans_model_en_in = AutoModelForSeq2SeqLM.from_pretrained(
                "ai4bharat/indictrans2-en-indic-1B",
                trust_remote_code=True
            )
            self.trans_model_en_in.to(self.device)
            self.trans_model_en_in.eval()

            print("✅ All models loaded successfully!\n")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Check internet connection (models download on first run)")
            print("   2. Ensure you have ~8GB free disk space")
            print("   3. Try: pip install --upgrade transformers torch IndicTransToolkit")
            raise

    # =========================================================================
    # MLM-BASED AUGMENTATION
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
        try:
            words = text.split()
            if len(words) < 3:
                return text

            # Calculate number of words to mask
            num_to_mask = max(1, int(len(words) * mask_ratio))

            # Randomly select words to mask (avoid very short words)
            maskable_indices = [i for i, w in enumerate(words) if len(w) > 2]
            if not maskable_indices:
                return text

            mask_indices = random.sample(
                maskable_indices,
                min(num_to_mask, len(maskable_indices))
            )

            # Create masked text
            masked_words = words.copy()
            for idx in mask_indices:
                masked_words[idx] = self.mlm_tokenizer.mask_token

            masked_text = ' '.join(masked_words)

            # Tokenize
            inputs = self.mlm_tokenizer(
                masked_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.mlm_model(**inputs)
                predictions = outputs.logits

            # Replace masked tokens with predictions
            result_words = words.copy()
            mask_token_id = self.mlm_tokenizer.mask_token_id

            for i, input_id in enumerate(inputs['input_ids'][0]):
                if input_id == mask_token_id:
                    # Get top-k predictions
                    top_k_tokens = torch.topk(predictions[0, i], k=top_k).indices

                    # Randomly select from top-k
                    predicted_token_id = random.choice(top_k_tokens.tolist())
                    predicted_token = self.mlm_tokenizer.decode([predicted_token_id]).strip()

                    # Find corresponding word index
                    mask_count = list(inputs['input_ids'][0][:i+1]).count(mask_token_id)
                    if mask_count <= len(mask_indices):
                        original_idx = mask_indices[mask_count - 1]
                        result_words[original_idx] = predicted_token

            augmented_text = ' '.join(result_words)

            # Return original if augmentation failed or resulted in same text
            return augmented_text if augmented_text != text else text

        except Exception as e:
            print(f"⚠️  MLM augmentation failed: {e}")
            return text

    # =========================================================================
    # BACK TRANSLATION (WITH MULTIPLE FALLBACK STRATEGIES)
    # =========================================================================

    def _preprocess_for_translation(self, text: str, src_lang: str, tgt_lang: str) -> Optional[str]:
        """
        Preprocess text for translation with multiple fallback strategies

        Args:
            text: Input text
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            Preprocessed text or None if failed
        """
        if not self.ip:
            # Fallback: Add language tags directly (works with some IndicTrans2 versions)
            return f"{src_lang} {text}"

        try:
            # Try IndicProcessor first
            batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            if batch and batch[0] and batch[0].strip():
                return batch[0]
        except Exception as e:
            print(f"   IndicProcessor failed: {e}")

        # Fallback: Add language tags directly
        return f"{src_lang} {text}"

    def _postprocess_translation(self, text: str, lang: str) -> str:
        """
        Postprocess translated text

        Args:
            text: Translated text
            lang: Language code

        Returns:
            Postprocessed text
        """
        if not self.ip:
            # Remove language tag if present
            parts = text.split(maxsplit=1)
            if len(parts) > 1 and parts[0] in self.lang_code_map.values():
                return parts[1]
            return text

        try:
            batch = self.ip.postprocess_batch([text], lang=lang)
            if batch and batch[0]:
                return batch[0]
        except Exception as e:
            print(f"   Postprocessing failed: {e}")

        return text

    def augment_with_back_translation(self, text: str, lang_code: str) -> str:
        """
        Augment text using back translation (Indic → English → Indic)
        Fixed version with proper IndicTrans2 preprocessing
        """
        try:
            # Validate input
            if not text or not text.strip():
                return text

            print(f"   🔄 Original: {text[:50]}...")

            # ====================================================================
            # STEP 1: Indic → English
            # ====================================================================

            # Preprocess with IndicProcessor
            if self.ip:
                try:
                    batch = self.ip.preprocess_batch(
                        [text],
                        src_lang=lang_code,
                        tgt_lang='eng_Latn'
                    )
                    preprocessed_text = batch[0] if batch and batch[0] else None
                except Exception as e:
                    print(f"   ⚠️ IndicProcessor failed: {e}")
                    preprocessed_text = None
            else:
                preprocessed_text = None

            # Fallback: Manual preprocessing
            if not preprocessed_text or not preprocessed_text.strip():
                # IndicTrans2 expects format: "source_lang target_lang text"
                preprocessed_text = text
                print(f"   Using fallback preprocessing")

            print(f"   Preprocessed: {preprocessed_text[:50]}...")

            # Tokenize for Indic→En
            try:
                inputs_to_en = self.trans_tokenizer_in_en(
                    [preprocessed_text],  # Pass as list
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
            except Exception as e:
                print(f"   ⚠️ Tokenization failed: {e}")
                return text

            # Validate tokenization
            if not inputs_to_en or 'input_ids' not in inputs_to_en:
                print(f"   ⚠️ Invalid tokenization result")
                return text

            if inputs_to_en['input_ids'].shape[1] == 0:
                print(f"   ⚠️ Empty tokenization")
                return text

            # Move to device
            inputs_to_en = {k: v.to(self.device) for k, v in inputs_to_en.items()}

            print(f"   Input shape: {inputs_to_en['input_ids'].shape}")

            # Generate English translation
            # Try different generation strategies to work around model bugs
            with torch.no_grad():
                try:
                    # Strategy 1: Try with use_cache=False (disables past_key_values)
                    generated_tokens = self.trans_model_in_en.generate(
                        **inputs_to_en,
                        num_beams=5,
                        num_return_sequences=1,
                        max_length=256,
                        use_cache=False
                    )
                except (AttributeError, TypeError) as e:
                    print(f"   Beam search failed, trying greedy decoding: {e}")
                    try:
                        # Strategy 2: Use greedy decoding (no beam search)
                        generated_tokens = self.trans_model_in_en.generate(
                            **inputs_to_en,
                            max_length=256,
                            do_sample=False,
                            use_cache=False
                        )
                    except Exception as e:
                        print(f"   Greedy decoding failed: {e}")
                        return text

            # Decode
            english_translations = self.trans_tokenizer_in_en.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Postprocess
            if self.ip:
                try:
                    english_batch = self.ip.postprocess_batch(
                        english_translations,
                        lang='eng_Latn'
                    )
                    english_text = english_batch[0] if english_batch else english_translations[0]
                except:
                    english_text = english_translations[0]
            else:
                english_text = english_translations[0]

            english_text = english_text.strip()

            if not english_text:
                print(f"   ⚠️ Empty English translation")
                return text

            print(f"   → English: {english_text[:50]}...")

            # ====================================================================
            # STEP 2: English → Indic
            # ====================================================================

            # Preprocess English text
            if self.ip:
                try:
                    batch = self.ip.preprocess_batch(
                        [english_text],
                        src_lang='eng_Latn',
                        tgt_lang=lang_code
                    )
                    preprocessed_english = batch[0] if batch and batch[0] else None
                except Exception as e:
                    print(f"   ⚠️ IndicProcessor failed: {e}")
                    preprocessed_english = None
            else:
                preprocessed_english = None

            # Fallback
            if not preprocessed_english or not preprocessed_english.strip():
                preprocessed_english = english_text
                print(f"   Using fallback preprocessing for English")

            # Tokenize for En→Indic
            try:
                inputs_to_indic = self.trans_tokenizer_en_in(
                    [preprocessed_english],  # Pass as list
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
            except Exception as e:
                print(f"   ⚠️ Tokenization failed: {e}")
                return text

            # Validate
            if not inputs_to_indic or 'input_ids' not in inputs_to_indic:
                print(f"   ⚠️ Invalid tokenization result")
                return text

            if inputs_to_indic['input_ids'].shape[1] == 0:
                print(f"   ⚠️ Empty tokenization")
                return text

            # Move to device
            inputs_to_indic = {k: v.to(self.device) for k, v in inputs_to_indic.items()}

            # Generate Indic translation
            with torch.no_grad():
                generated_tokens = self.trans_model_en_in.generate(
                    **inputs_to_indic,
                    num_beams=5,
                    num_return_sequences=1,
                    max_length=256
                )

            # Decode
            indic_translations = self.trans_tokenizer_en_in.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Postprocess
            if self.ip:
                try:
                    indic_batch = self.ip.postprocess_batch(
                        indic_translations,
                        lang=lang_code
                    )
                    back_translated = indic_batch[0] if indic_batch else indic_translations[0]
                except:
                    back_translated = indic_translations[0]
            else:
                back_translated = indic_translations[0]

            back_translated = back_translated.strip()

            if not back_translated:
                print(f"   ⚠️ Empty back translation")
                return text

            print(f"   → Back translated: {back_translated[:50]}...")

            return back_translated

        except Exception as e:
            print(f"   ⚠️ Back translation failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return text
        
    # =========================================================================
    # DATASET AUGMENTATION
    # =========================================================================

    def map_language(self, lang: str) -> Optional[str]:
        """Map dataset language name to model language code"""
        lang_lower = lang.lower().strip()
        return self.lang_code_map.get(lang_lower)

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
            techniques: List of techniques ['mlm', 'back_translation'] or None (use both)
            text_column: Name of text column
            label_column: Name of label column
            language_column: Name of language column
            balance_labels: Augment minority classes more

        Returns:
            Augmented DataFrame
        """
        print(f"🔄 Starting augmentation pipeline...")
        print(f"   Original samples: {len(df)}")
        print(f"   Augmentations per sample: {num_aug_per_row}")

        # Default to both techniques, but check if back translation is available
        if techniques is None:
            techniques = ["mlm"]
            if self.ip is not None:
                techniques.append("back_translation")
            else:
                print("   ⚠️  Back translation disabled (IndicProcessor not available)")

        # Calculate augmentation strategy if balancing
        aug_strategy = {}
        if balance_labels:
            label_counts = df[label_column].value_counts()
            max_count = label_counts.max()

            for label in label_counts.index:
                count = label_counts[label]
                ratio = max_count / count
                aug_strategy[label] = min(int(num_aug_per_row * ratio), 5)

            print(f"   ⚖️  Balancing strategy: {aug_strategy}")

        new_rows = []
        successful_aug = {'mlm': 0, 'back_translation': 0}
        failed_aug = 0

        for idx, row in df.iterrows():
            original_text = row[text_column]
            original_label = row[label_column]
            original_lang = row[language_column]

            # Add original row
            new_rows.append(row.to_dict())

            # Map language
            lang_code = self.map_language(original_lang)
            if not lang_code:
                print(f"⚠️  Unknown language: {original_lang}, skipping augmentation")
                continue

            # Determine number of augmentations
            n_aug = aug_strategy.get(original_label, num_aug_per_row) if balance_labels else num_aug_per_row

            # Generate augmentations
            for _ in range(n_aug):
                # Randomly choose technique
                technique = random.choice(techniques)

                new_text = original_text

                try:
                    if technique == "mlm":
                        new_text = self.augment_with_mlm(
                            original_text,
                            mask_ratio=0.15,
                            top_k=5
                        )
                        if new_text != original_text:
                            successful_aug['mlm'] += 1

                    elif technique == "back_translation":
                        new_text = self.augment_with_back_translation(
                            original_text,
                            lang_code
                        )
                        if new_text != original_text:
                            successful_aug['back_translation'] += 1

                except Exception as e:
                    print(f"⚠️  Augmentation failed for: '{original_text[:50]}...' | Error: {e}")
                    failed_aug += 1
                    continue

                # Add augmented sample (only if different from original)
                if new_text != original_text and new_text.strip():
                    new_rows.append({
                        text_column: new_text,
                        label_column: original_label,
                        language_column: original_lang
                    })

            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"   Processed: {idx + 1}/{len(df)} samples...")

        augmented_df = pd.DataFrame(new_rows)

        # Print statistics
        print(f"\n✅ Augmentation complete!")
        print(f"   Original: {len(df):,} samples")
        print(f"   Augmented: {len(augmented_df):,} samples")
        print(f"   Increase: {len(augmented_df) - len(df):,} (+{((len(augmented_df)/len(df)-1)*100):.1f}%)")
        print(f"\n📊 Augmentation breakdown:")
        print(f"   MLM: {successful_aug['mlm']:,}")
        print(f"   Back Translation: {successful_aug['back_translation']:,}")
        print(f"   Failed: {failed_aug:,}")

        if label_column in df.columns:
            print(f"\n📈 Label distribution:")
            for label in sorted(augmented_df[label_column].unique()):
                orig = len(df[df[label_column] == label])
                aug = len(augmented_df[augmented_df[label_column] == label])
                print(f"   Label {label}: {orig:,} → {aug:,} (+{aug-orig:,})")

        return augmented_df

    def test_augmentation(self, text: str, language: str, n_samples: int = 3):
        """
        Test augmentation on a single text

        Args:
            text: Sample text
            language: Language name
            n_samples: Number of augmented samples to generate
        """
        lang_code = self.map_language(language)
        if not lang_code:
            print(f"❌ Unknown language: {language}")
            return

        print(f"🔬 Testing augmentation on: '{text}'\n")
        print(f"Language: {language} ({lang_code})\n")

        print("="*70)
        print("MLM Augmentation:")
        print("="*70)
        for i in range(n_samples):
            aug = self.augment_with_mlm(text)
            print(f"{i+1}. {aug}")

        if self.ip is not None:
            print("\n" + "="*70)
            print("Back Translation:")
            print("="*70)
            for i in range(n_samples):
                aug = self.augment_with_back_translation(text, lang_code)
                print(f"{i+1}. {aug}")
        else:
            print("\n⚠️  Back translation disabled (IndicProcessor not available)")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Initializing Indic Augmentation Pipeline...\n")

    try:
        pipeline = IndicAugmentationPipeline()
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        print("\n💡 Quick fix:")
        print("   pip install torch transformers sentencepiece IndicTransToolkit")
        exit(1)

    # Test on single text
    print("\n" + "="*70)
    print("Testing Single Text")
    print("="*70 + "\n")

    sample_text = "पश्चिम बंगालमध्ये फटाक्यांच्या कारखान्यात भीषण स्फोट"
    pipeline.test_augmentation(sample_text, language='marathi', n_samples=3)