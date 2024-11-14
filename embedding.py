from flask import Flask, request, render_template
from transformers import CanineTokenizer, CanineModel
import torch
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

# Load Canine model and tokenizer
model = CanineModel.from_pretrained('google/canine-c')
tokenizer = CanineTokenizer.from_pretrained('google/canine-c')

def embed_texts(texts):
    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    return outputs.last_hidden_state.mean(dim=1).tolist()

def embed_query(text):
    encoding = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
    return outputs.last_hidden_state.mean(dim=1).tolist()[0]

def embedding_extractor(file_path):
    # Load Docs
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)

    # Embed and create vectorstore
    embeddings = embed_texts([doc.page_content for doc in splits])
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def doc_format(retriever):
    # Retrieve documents and format them
    retrieved_docs = retriever.get_relevant_documents("What is Task Decomposition?")
    formatted_docs = format_docs(retrieved_docs)
    return formatted_docs