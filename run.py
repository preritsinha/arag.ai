from flask import Flask,request
import embedding,generator


app = Flask(__name__)

formatted_docs = None
combined_context = None
contextual = False

@app.route("/extract-embeddings",methods=['POST'])
def extract_embeddings():
    input_json = request.get_json()
    url = input_json['url']
    retriever = embedding.embedding_extractor(url)
    global formatted_docs 
    formatted_docs = embedding.doc_format(retriever)
    global contextual 
    contextual = True 
    return "Embeddings created successfully"

@app.route("/result_generator",methods=['POST'])
def result_generator():
    input_json = request.get_json()
    prompt = input_json['prompt']
    global contextual
    if contextual:
        global formatted_docs
        combined_context = generator.prompt_gen(formatted_docs,prompt)
        return generator.res_gen_contextual(combined_context)
    else:
        return generator.res_gen_normal(prompt)

if __name__=='__main__':
    app.run(debug=True, port=5001)