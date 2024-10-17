from Emb_generation import generate_embedding, documents
import emb_index as ei
def find_relevant_documents(query):
    query_embedding = generate_embedding(query)
    distances, indices = ei.index.search(query_embedding, k=3)

    relevant_docs = [documents[idx] for idx in indices[0]]
    return relevant_docs

def generate_prompt(query, relevant_docs):
    if "инструкция" in query.lower():
        prompt = f"ИНСТУКЦИЯ {relevant_docs[0]['text']}"
    else:
        prompt = f"доки {', '.join([doc['text'] for doc in relevant_docs])}"

    return prompt

