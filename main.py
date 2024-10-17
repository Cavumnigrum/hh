import cohere
from config import token
from Query_prompt import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from lg import chat_with_bot
co = cohere.Client(token)

def generate_llm_response(prompt):
    response = co.generate(
        model="command-r-08-2024",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )
    return response.generations[0].text

query = "Explain artificial intelligence"
relevant_docs = find_relevant_documents(query)
prompt = generate_prompt(query, relevant_docs)
llm_response = generate_llm_response(prompt)
print("#1")
print(llm_response)
print("with langchain")
while True:
    user_input = input("You> ")
    if user_input.lower() == "q":
        break
    response = chat_with_bot(user_input)
    print("Bot> ", response)

