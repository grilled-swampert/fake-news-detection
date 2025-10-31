"""
Masked Language Model (MLM) based augmentation
"""

import torch
import random
from config import DEFAULT_MLM_PARAMS


class MLMAugmenter:
    """Handles MLM-based text augmentation using MuRIL"""

    def __init__(self, tokenizer, model, device: str):
        """
        Initialize MLM augmenter

        Args:
            tokenizer: MLM tokenizer
            model: MLM model
            device: Device to run on
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def augment(self, text: str, mask_ratio: float = None,
                top_k: int = None) -> str:
        """
        Augment text using Masked Language Modeling

        Args:
            text: Input text
            mask_ratio: Ratio of words to mask (default from config)
            top_k: Number of top predictions to consider (default from config)

        Returns:
            Augmented text
        """
        # Use defaults if not specified
        if mask_ratio is None:
            mask_ratio = DEFAULT_MLM_PARAMS['mask_ratio']
        if top_k is None:
            top_k = DEFAULT_MLM_PARAMS['top_k']

        try:
            words = text.split()
            if len(words) < 3:
                return text

            # Calculate number of words to mask
            num_to_mask = max(1, int(len(words) * mask_ratio))

            # Select maskable words (avoid very short words)
            min_word_len = DEFAULT_MLM_PARAMS['min_word_length']
            maskable_indices = [
                i for i, w in enumerate(words)
                if len(w) > min_word_len
            ]

            if not maskable_indices:
                return text

            # Randomly select words to mask
            mask_indices = random.sample(
                maskable_indices,
                min(num_to_mask, len(maskable_indices))
            )

            # Create masked text
            masked_words = words.copy()
            for idx in mask_indices:
                masked_words[idx] = self.tokenizer.mask_token

            masked_text = ' '.join(masked_words)

            # Tokenize
            inputs = self.tokenizer(
                masked_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits

            # Replace masked tokens with predictions
            result_words = words.copy()
            mask_token_id = self.tokenizer.mask_token_id

            for i, input_id in enumerate(inputs['input_ids'][0]):
                if input_id == mask_token_id:
                    # Get top-k predictions
                    top_k_tokens = torch.topk(
                        predictions[0, i], k=top_k
                    ).indices

                    # Randomly select from top-k
                    predicted_token_id = random.choice(top_k_tokens.tolist())
                    predicted_token = self.tokenizer.decode(
                        [predicted_token_id]
                    ).strip()

                    # Find corresponding word index
                    mask_count = list(inputs['input_ids'][0][:i + 1]).count(
                        mask_token_id
                    )
                    if mask_count <= len(mask_indices):
                        original_idx = mask_indices[mask_count - 1]
                        result_words[original_idx] = predicted_token

            augmented_text = ' '.join(result_words)

            # Return original if augmentation failed or resulted in same text
            return augmented_text if augmented_text != text else text

        except Exception as e:
            print(f"⚠️  MLM augmentation failed: {e}")
            return text