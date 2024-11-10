from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

sentiment_pipeline = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

result = sentiment_pipeline("I love Hugging Face!")
print(result)

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-french-europeana-cased")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-base-french-europeana-cased")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


text = "Le chevalier a trouvé l'épée magique dans la grotte sombre."
result = ner_pipeline(text)
if result:
    for entity in result:
        print(f"Word: {entity['word']}, Entity: {entity['entity']}, Score: {entity['score']:.2f}")
else:
    print("No found.")

