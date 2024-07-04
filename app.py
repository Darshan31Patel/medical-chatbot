from src.helper import download_hugging_face_embeddings,save_embeddings
from langchain_community.llms import Ollama
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
import joblib
import os
import streamlit as st


PATH = 'model/embedding.joblib'

if os.path.exists(PATH):
    embeddings = joblib.load(PATH)
else: 
    embeddings = download_hugging_face_embeddings()
    save_embeddings(embeddings)

# Load Existing vector database
docsearch = PineconeVectorStore.from_existing_index(index_name="mchatbot",embedding=embeddings)


prompt = ChatPromptTemplate.from_template(prompt_template)
llm = Ollama(model="llama2")

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = docsearch.as_retriever()

retrieval_chain = create_retrieval_chain(retriever,document_chain)


st.title("Medical ChatBot")
input = st.text_input("Ask anything related to medicine or diseases : ")
if input:
    print("start")
    response = retrieval_chain.invoke({"input" : input})
    print(response['answer'])
    print("end")
    st.write(response['answer'])