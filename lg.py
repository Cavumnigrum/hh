from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory


class GPTNeoModel:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs, max_length=max_tokens, do_sample=True, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
model_name = "EleutherAI/gpt-neo-1.3B"
gpt_neo_model = GPTNeoModel(model_name=model_name)

prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="User: {user_input}\nAI:"
)

memory = ConversationBufferMemory(memory_key="chat_history")

class GPTNeoLangChain:
    def __init__(self, gpt_model, prompt_template, memory):
        self.gpt_model = gpt_model
        self.prompt_template = prompt_template
        self.memory = memory

    def run(self, user_input):
        prompt = self.prompt_template.format(user_input=user_input)
        response = self.gpt_model.generate(prompt)
        self.memory.save_context({"input": user_input}, {"output": response})
        return response

llm_chain = GPTNeoLangChain(
    gpt_model=gpt_neo_model,
    prompt_template=prompt_template,
    memory=memory
)

def chat_with_bot(user_input):
    response = llm_chain.run(user_input=user_input)
    return response


