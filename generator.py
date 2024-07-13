import os,requests,json
from dotenv import load_dotenv
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

load_dotenv()
token = os.getenv('TOKEN')
print(token)
headers = {"Authorization": f"Bearer {token}"}

def res_gen(prompt,model='google/gemma-2b-it'):
    model = f"https://api-inference.huggingface.co/models/{model}"
    response = requests.post(model, headers=headers, json=prompt)
    check = validator(prompt,response[0]['generated_text'].replace(prompt,'').strip())
    if check == 'YES':
        return response.json()
    else:
        return 'Answer is not comprehensible. Please re-phrase the query correctly.'

def prompt_gen(formatted_docs,prompt):
    context = re.sub(r'[\r\n!@#$%^&*(),.?":{}|<>]', '', formatted_docs)
    prompt = f'''
Answer the question based on the given context.

CONTEXT: {context}
Question: {prompt}
'''
    return prompt

def res_gen_contextual(prompt,model='google/gemma-2b-it'):
    model = f"https://api-inference.huggingface.co/models/{model}"
    response = res_gen({"inputs":f"{prompt}"},model)
    check = validator(prompt,response[0]['generated_text'].replace(prompt,'').strip())
    if check == 'YES':
        return response.json()
    else:
        return 'Answer is not comprehensible. Please re-phrase the query correctly.'

def res_gen_normal(prompt,model='google/gemma-2b-it'):
    model = f"https://api-inference.huggingface.co/models/{model}"
    response = res_gen({"inputs":f"{prompt}"},model)
    check = validator(prompt,response[0]['generated_text'].replace(prompt,'').strip())    
    if check == 'YES':
        return response.json()
    else:
        return 'Answer is not comprehensible. Please re-phrase the query correctly.'

def validator(prompt,response):
    model='google/flan-t5-large'
    query = f'''
Do you think the below question answer is correct? Just reply one word answer - YES or NO.

Question: {prompt}
Answer: {response}
'''
    response = res_gen(query,model)
    return response[0]['generated_text'].replace(prompt,'').strip()
    
