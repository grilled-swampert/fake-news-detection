# predict_muril.py

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
# Path to your saved model from Step 2
MODEL_PATH = "./muril-finetuned-fake-news"

# 1. Load the fine-tuned model and tokenizer
print(f"Loading model from: {MODEL_PATH}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 2. Create a text classification pipeline
# This pipeline handles all the tokenization and inference steps for you.
# device=0 means use the first GPU, device=-1 means use CPU.
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

# 3. Test with new examples
hindi_news = "पीएम मोदी ने वाराणसी में नए क्रिकेट स्टेडियम का शिलान्यास किया, सचिन तेंदुलकर भी रहे मौजूद"
gujarati_news = "ગુજરાતમાં ચોમાસાની વિદાય બાદ પણ વરસાદી માહોલ, જાણો હવામાન વિભાગે શું કરી આગાહી"
fake_news_example = "व्हाट्सएप अब गुलाबी रंग में उपलब्ध है, इसे अपडेट करने के लिए इस लिंक पर क्लिक करें"

news_articles = [
    hindi_news,
    gujarati_news,
    fake_news_example
]

print("\nClassifying news articles...")
results = classifier(news_articles)

# The model outputs 'LABEL_0' (fake) and 'LABEL_1' (real). Let's make it readable.
label_map = {"LABEL_0": "Fake", "LABEL_1": "Real"}

for article, result in zip(news_articles, results):
    prediction = label_map[result['label']]
    confidence = result['score']
    print("-" * 50)
    print(f"Article: {article[:70]}...")
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")
    print("-" * 50)