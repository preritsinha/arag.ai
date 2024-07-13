from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# Load a compatible model from sentence-transformers
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")

# Wrapper class for embeddings
class EmbeddingsWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=True).tolist()[0]

# Instantiate the wrapper
embedding_wrapper = EmbeddingsWrapper(model)

def embedding_extractor(file_path):
    # Load Docs
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    # Embed and create vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_wrapper)
    retriever = vectorstore.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def doc_format(retriever):
    # Retrieve documents and format them
    retrieved_docs = retriever.get_relevant_documents("What is Task Decomposition?")
    formatted_docs = format_docs(retrieved_docs)
    return (formatted_docs)

