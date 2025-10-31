"""
Main script demonstrating usage of the Indic Augmentation Pipeline

INSTALLATION:
    pip install torch transformers sentencepiece pandas numpy
    pip install IndicTransToolkit  # For back translation support

    For GPU support (optional, 10-20x faster):
    pip install torch --index-url https://download.pytorch.org/whl/cu118
"""

import pandas as pd
from pipeline import IndicAugmentationPipeline


def test_single_text():
    """Test augmentation on a single text sample"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Text Augmentation")
    print("=" * 70 + "\n")

    sample_text = "पश्चिम बंगालमध्ये फटाक्यांच्या कारखान्यात भीषण स्फोट"

    pipeline = IndicAugmentationPipeline()
    pipeline.test_augmentation(
        text=sample_text,
        language='marathi',
        n_samples=3
    )


def test_dataset_augmentation():
    """Test augmentation on a sample dataset"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Dataset Augmentation")
    print("=" * 70 + "\n")

    # Create sample dataset
    data = {
        'text': [
            'यह एक अच्छी खबर है',
            'बुरी खबर आई है',
            'मौसम बहुत अच्छा है',
            'आज बारिश हो रही है',
        ],
        'label': [1, 0, 1, 0],  # 1=positive, 0=negative
        'language': ['hindi', 'hindi', 'hindi', 'hindi']
    }

    df = pd.DataFrame(data)

    print("Original Dataset:")
    print(df)
    print(f"\nShape: {df.shape}")

    # Initialize pipeline
    pipeline = IndicAugmentationPipeline()

    # Augment dataset
    augmented_df = pipeline.create_augmented_dataset(
        df=df,
        num_aug_per_row=2,
        techniques=['mlm'],  # Use only MLM for faster testing
        text_column='text',
        label_column='label',
        language_column='language',
        balance_labels=False
    )

    print("\n" + "=" * 70)
    print("Augmented Dataset:")
    print("=" * 70)
    print(augmented_df)
    print(f"\nShape: {augmented_df.shape}")

    # Save to CSV (optional)
    # augmented_df.to_csv('augmented_data.csv', index=False)
    # print("\n✅ Saved to 'augmented_data.csv'")


def test_multilingual_dataset():
    """Test augmentation on multilingual dataset"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multilingual Dataset Augmentation")
    print("=" * 70 + "\n")

    # Create multilingual sample dataset
    data = {
        'text': [
            'यह एक अच्छी फिल्म है',  # Hindi
            'हे एक चांगले चित्रपट आहे',  # Marathi
            'இது ஒரு நல்ல திரைப்படம்',  # Tamil
            'यह बुरी खबर है',  # Hindi
        ],
        'label': [1, 1, 1, 0],
        'language': ['hindi', 'marathi', 'tamil', 'hindi']
    }

    df = pd.DataFrame(data)

    print("Original Multilingual Dataset:")
    print(df)

    # Initialize pipeline
    pipeline = IndicAugmentationPipeline()

    # Show supported languages
    print(f"\n📚 Supported languages: {', '.join(pipeline.get_supported_languages())}")
    print(f"🔧 Available techniques: {', '.join(pipeline.get_available_techniques())}")

    # Augment with label balancing
    augmented_df = pipeline.create_augmented_dataset(
        df=df,
        num_aug_per_row=1,
        techniques=['mlm'],
        balance_labels=True  # Balance minority classes
    )

    print("\n" + "=" * 70)
    print("Augmented Multilingual Dataset:")
    print("=" * 70)
    print(augmented_df)


def test_all_techniques():
    """Test all available augmentation techniques"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Testing All Techniques")
    print("=" * 70 + "\n")

    sample_text = "यह एक परीक्षण वाक्य है"

    pipeline = IndicAugmentationPipeline()

    print(f"Original: {sample_text}\n")

    # Test MLM
    print("MLM Augmentation:")
    for i in range(3):
        aug = pipeline.augment_with_mlm(sample_text, mask_ratio=0.2)
        print(f"  {i + 1}. {aug}")

    # Test back translation (if available)
    if 'back_translation' in pipeline.get_available_techniques():
        print("\nBack Translation:")
        lang_code = pipeline.map_language('hindi')
        for i in range(2):
            aug = pipeline.augment_with_back_translation(
                sample_text,
                lang_code,
                verbose=False
            )
            print(f"  {i + 1}. {aug}")


if __name__ == "__main__":
    print("=" * 70)
    print("INDIC LANGUAGE DATA AUGMENTATION PIPELINE - EXAMPLES")
    print("=" * 70)

    try:
        # Run examples
        # Uncomment the examples you want to run:

        test_single_text()
        # test_dataset_augmentation()
        # test_multilingual_dataset()
        # test_all_techniques()

        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        print("\n💡 Check that all dependencies are installed:")
        print("   pip install torch transformers sentencepiece pandas numpy")
        print("   pip install IndicTransToolkit")