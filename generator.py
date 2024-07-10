import os,requests,json
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

load_dotenv()
token = os.getenv('TOKEN')
print(token)
headers = {"Authorization": f"Bearer {token}"}

def res_gen(payload,model):
    response = requests.post(model, headers=headers, json=payload)
    return response.json()

def prompt_gen(formatted_docs,prompt):
    prompt = {"inputs": {"question": f"{prompt}","context": f"{formatted_docs})"}}
    return prompt

def res_gen_contextual(prompt):
    model = "https://api-inference.huggingface.co/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    response = res_gen(prompt,model)
    print(response)
    return response['answer']

def res_gen_normal(prompt):
    model = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
    response = res_gen({"inputs":f"{prompt}"},model)
    print(response)
    return(response[0]['generated_text'])
