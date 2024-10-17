from transformers import AutoTokenizer, AutoModel
import json
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Добавление специального токена для заполнения (если не установлен)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Используем eos_token как pad_token
model = AutoModel.from_pretrained(model_name)

def generate_embedding(text):
    # Токенизация текста с учетом заполнения и усечения
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

with open("db.json", "rb") as f:
    documents = dict(json.load(f))["documents"]

document_embeddings = [generate_embedding(doc['text']) for doc in documents]

