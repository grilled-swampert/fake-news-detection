"""
Back translation based augmentation
"""

import torch
from typing import Optional
from config import DEFAULT_TRANSLATION_PARAMS


class TranslationAugmenter:
    """Handles back-translation augmentation using IndicTrans2"""

    def __init__(self, tokenizer_in_en, model_in_en,
                 tokenizer_en_in, model_en_in, device: str,
                 indic_processor=None):
        """
        Initialize translation augmenter

        Args:
            tokenizer_in_en: Indic→En tokenizer
            model_in_en: Indic→En model
            tokenizer_en_in: En→Indic tokenizer
            model_en_in: En→Indic model
            device: Device to run on
            indic_processor: IndicProcessor instance (optional)
        """
        self.tokenizer_in_en = tokenizer_in_en
        self.model_in_en = model_in_en
        self.tokenizer_en_in = tokenizer_en_in
        self.model_en_in = model_en_in
        self.device = device
        self.ip = indic_processor

    def _preprocess_text(self, text: str, src_lang: str,
                         tgt_lang: str) -> Optional[str]:
        """Preprocess text for translation with fallback"""
        if not self.ip:
            return text

        try:
            batch = self.ip.preprocess_batch(
                [text], src_lang=src_lang, tgt_lang=tgt_lang
            )
            if batch and batch[0] and batch[0].strip():
                return batch[0]
        except Exception as e:
            print(f"   IndicProcessor preprocessing failed: {e}")

        return text

    def _postprocess_text(self, text: str, lang: str) -> str:
        """Postprocess translated text"""
        if not self.ip:
            # Remove language tag if present
            parts = text.split(maxsplit=1)
            if len(parts) > 1:
                return parts[1]
            return text

        try:
            batch = self.ip.postprocess_batch([text], lang=lang)
            if batch and batch[0]:
                return batch[0]
        except Exception as e:
            print(f"   Postprocessing failed: {e}")

        return text

    def _translate_indic_to_english(self, text: str,
                                    lang_code: str) -> Optional[str]:
        """Translate from Indic language to English"""
        try:
            # Preprocess
            preprocessed = self._preprocess_text(
                text, src_lang=lang_code, tgt_lang='eng_Latn'
            )

            if not preprocessed or not preprocessed.strip():
                return None

            # Tokenize
            inputs = self.tokenizer_in_en(
                [preprocessed],
                padding=True,
                truncation=True,
                max_length=DEFAULT_TRANSLATION_PARAMS['max_length'],
                return_tensors='pt'
            )

            if not inputs or 'input_ids' not in inputs:
                return None

            if inputs['input_ids'].shape[1] == 0:
                return None

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                try:
                    # Try with use_cache=False to avoid model bugs
                    generated = self.model_in_en.generate(
                        **inputs,
                        num_beams=DEFAULT_TRANSLATION_PARAMS['num_beams'],
                        num_return_sequences=DEFAULT_TRANSLATION_PARAMS['num_return_sequences'],
                        max_length=DEFAULT_TRANSLATION_PARAMS['max_length'],
                        use_cache=False
                    )
                except (AttributeError, TypeError) as e:
                    print(f"   Beam search failed, using greedy: {e}")
                    # Fallback to greedy decoding
                    generated = self.model_in_en.generate(
                        **inputs,
                        max_length=DEFAULT_TRANSLATION_PARAMS['max_length'],
                        do_sample=False,
                        use_cache=False
                    )

            # Decode
            translations = self.tokenizer_in_en.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Postprocess
            english_text = self._postprocess_text(translations[0], 'eng_Latn')
            return english_text.strip() if english_text else None

        except Exception as e:
            print(f"   Indic→English translation failed: {e}")
            return None

    def _translate_english_to_indic(self, text: str,
                                    lang_code: str) -> Optional[str]:
        """Translate from English to Indic language"""
        try:
            # Preprocess
            preprocessed = self._preprocess_text(
                text, src_lang='eng_Latn', tgt_lang=lang_code
            )

            if not preprocessed or not preprocessed.strip():
                return None

            # Tokenize
            inputs = self.tokenizer_en_in(
                [preprocessed],
                padding=True,
                truncation=True,
                max_length=DEFAULT_TRANSLATION_PARAMS['max_length'],
                return_tensors='pt'
            )

            if not inputs or 'input_ids' not in inputs:
                return None

            if inputs['input_ids'].shape[1] == 0:
                return None

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated = self.model_en_in.generate(
                    **inputs,
                    num_beams=DEFAULT_TRANSLATION_PARAMS['num_beams'],
                    num_return_sequences=DEFAULT_TRANSLATION_PARAMS['num_return_sequences'],
                    max_length=DEFAULT_TRANSLATION_PARAMS['max_length']
                )

            # Decode
            translations = self.tokenizer_en_in.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Postprocess
            indic_text = self._postprocess_text(translations[0], lang_code)
            return indic_text.strip() if indic_text else None

        except Exception as e:
            print(f"   English→Indic translation failed: {e}")
            return None

    def augment(self, text: str, lang_code: str,
                verbose: bool = False) -> str:
        """
        Augment text using back translation (Indic → English → Indic)

        Args:
            text: Input text
            lang_code: Language code (e.g., 'hin_Deva')
            verbose: Print intermediate steps

        Returns:
            Back-translated text (or original if failed)
        """
        try:
            if not text or not text.strip():
                return text

            if verbose:
                print(f"   🔄 Original: {text[:50]}...")

            # Step 1: Indic → English
            english_text = self._translate_indic_to_english(text, lang_code)

            if not english_text:
                if verbose:
                    print(f"   ⚠️ Failed to translate to English")
                return text

            if verbose:
                print(f"   → English: {english_text[:50]}...")

            # Step 2: English → Indic
            back_translated = self._translate_english_to_indic(
                english_text, lang_code
            )

            if not back_translated:
                if verbose:
                    print(f"   ⚠️ Failed to translate back to Indic")
                return text

            if verbose:
                print(f"   → Back translated: {back_translated[:50]}...")

            return back_translated

        except Exception as e:
            print(f"   ⚠️ Back translation failed: {type(e).__name__}: {e}")
            return text