from flask import Flask,request,render_template,jsonify
import embedding,generator
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

formatted_docs = None
combined_context = None
contextual = False

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file_post():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if f:
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)
            
            retriever = embedding.embedding_extractor(file_path)
            global formatted_docs 
            formatted_docs = embedding.doc_format(retriever)
            global contextual 
            contextual = True 
            return jsonify({'message': 'SUCCESS: File Processed and Embeddings Created'}), 200
    else:
        return render_template('upload.html')

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