"""
Dataset augmentation utilities
"""

import random
import pandas as pd
from typing import List, Dict
from config import LANG_CODE_MAP, DEFAULT_DATASET_PARAMS


class DatasetAugmenter:
    """Handles dataset-level augmentation operations"""

    def __init__(self, mlm_augmenter, translation_augmenter=None):
        """
        Initialize dataset augmenter

        Args:
            mlm_augmenter: MLMAugmenter instance
            translation_augmenter: TranslationAugmenter instance (optional)
        """
        self.mlm_augmenter = mlm_augmenter
        self.translation_augmenter = translation_augmenter

    def map_language(self, lang: str) -> str:
        """Map dataset language name to model language code"""
        lang_lower = lang.lower().strip()
        return LANG_CODE_MAP.get(lang_lower)

    def _calculate_augmentation_strategy(self, df: pd.DataFrame,
                                         label_column: str,
                                         num_aug_per_row: int) -> Dict:
        """Calculate how many augmentations per label for balancing"""
        label_counts = df[label_column].value_counts()
        max_count = label_counts.max()

        strategy = {}
        for label in label_counts.index:
            count = label_counts[label]
            ratio = max_count / count
            strategy[label] = min(int(num_aug_per_row * ratio), 5)

        return strategy

    def augment_dataset(self,
                        df: pd.DataFrame,
                        num_aug_per_row: int = None,
                        techniques: List[str] = None,
                        text_column: str = None,
                        label_column: str = None,
                        language_column: str = None,
                        balance_labels: bool = None) -> pd.DataFrame:
        """
        Create augmented dataset with multiple techniques

        Args:
            df: Input DataFrame
            num_aug_per_row: Augmentations per sample (default from config)
            techniques: List of techniques or None (auto-detect available)
            text_column: Name of text column (default from config)
            label_column: Name of label column (default from config)
            language_column: Name of language column (default from config)
            balance_labels: Augment minority classes more (default from config)

        Returns:
            Augmented DataFrame
        """
        # Use defaults if not specified
        if num_aug_per_row is None:
            num_aug_per_row = DEFAULT_DATASET_PARAMS['num_aug_per_row']
        if text_column is None:
            text_column = DEFAULT_DATASET_PARAMS['text_column']
        if label_column is None:
            label_column = DEFAULT_DATASET_PARAMS['label_column']
        if language_column is None:
            language_column = DEFAULT_DATASET_PARAMS['language_column']
        if balance_labels is None:
            balance_labels = DEFAULT_DATASET_PARAMS['balance_labels']

        # Determine available techniques
        if techniques is None:
            techniques = ["mlm"]
            if self.translation_augmenter is not None:
                techniques.append("back_translation")
            else:
                print("   ℹ️  Back translation not available")

        print(f"🔄 Starting augmentation pipeline...")
        print(f"   Original samples: {len(df)}")
        print(f"   Augmentations per sample: {num_aug_per_row}")
        print(f"   Techniques: {', '.join(techniques)}")

        # Calculate augmentation strategy if balancing
        aug_strategy = {}
        if balance_labels and label_column in df.columns:
            aug_strategy = self._calculate_augmentation_strategy(
                df, label_column, num_aug_per_row
            )
            print(f"   ⚖️  Balancing strategy: {aug_strategy}")

        new_rows = []
        stats = {
            'mlm': 0,
            'back_translation': 0,
            'failed': 0
        }

        for idx, row in df.iterrows():
            original_text = row[text_column]
            original_label = row.get(label_column)
            original_lang = row[language_column]

            # Add original row
            new_rows.append(row.to_dict())

            # Map language
            lang_code = self.map_language(original_lang)
            if not lang_code:
                print(f"⚠️  Unknown language: {original_lang}, skipping")
                continue

            # Determine number of augmentations
            n_aug = (aug_strategy.get(original_label, num_aug_per_row)
                     if balance_labels and original_label
                     else num_aug_per_row)

            # Generate augmentations
            for _ in range(n_aug):
                technique = random.choice(techniques)
                new_text = original_text

                try:
                    if technique == "mlm":
                        new_text = self.mlm_augmenter.augment(original_text)
                        if new_text != original_text:
                            stats['mlm'] += 1

                    elif technique == "back_translation":
                        if self.translation_augmenter:
                            new_text = self.translation_augmenter.augment(
                                original_text, lang_code
                            )
                            if new_text != original_text:
                                stats['back_translation'] += 1

                except Exception as e:
                    print(f"⚠️  Aug failed: '{original_text[:30]}...' | {e}")
                    stats['failed'] += 1
                    continue

                # Add augmented sample (only if different)
                if new_text != original_text and new_text.strip():
                    aug_row = {
                        text_column: new_text,
                        language_column: original_lang
                    }
                    if label_column in row:
                        aug_row[label_column] = original_label
                    new_rows.append(aug_row)

            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"   Processed: {idx + 1}/{len(df)} samples...")

        augmented_df = pd.DataFrame(new_rows)

        # Print statistics
        self._print_augmentation_stats(
            df, augmented_df, stats, label_column
        )

        return augmented_df

    def _print_augmentation_stats(self, original_df: pd.DataFrame,
                                  augmented_df: pd.DataFrame,
                                  stats: Dict, label_column: str):
        """Print augmentation statistics"""
        print(f"\n✅ Augmentation complete!")
        print(f"   Original: {len(original_df):,} samples")
        print(f"   Augmented: {len(augmented_df):,} samples")
        increase = len(augmented_df) - len(original_df)
        pct = ((len(augmented_df) / len(original_df) - 1) * 100)
        print(f"   Increase: {increase:,} (+{pct:.1f}%)")

        print(f"\n📊 Augmentation breakdown:")
        print(f"   MLM: {stats['mlm']:,}")
        print(f"   Back Translation: {stats['back_translation']:,}")
        print(f"   Failed: {stats['failed']:,}")

        if label_column in original_df.columns:
            print(f"\n📈 Label distribution:")
            for label in sorted(augmented_df[label_column].unique()):
                orig = len(original_df[original_df[label_column] == label])
                aug = len(augmented_df[augmented_df[label_column] == label])
                print(f"   Label {label}: {orig:,} → {aug:,} (+{aug - orig:,})")