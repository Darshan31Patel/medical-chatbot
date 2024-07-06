from src.helper import download_hugging_face_embeddings,save_embeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
import joblib
import os
import streamlit as st


PATH = 'model/embedding.joblib'
GROQ_API_KEY = os.environ['GROQ_API_KEY']

if os.path.exists(PATH):
    embeddings = joblib.load(PATH)
    print("Model loaded")
else: 
    embeddings = download_hugging_face_embeddings()
    save_embeddings(embeddings)
    print("Model saved")

# Load Existing vector database
docsearch = PineconeVectorStore.from_existing_index(index_name="mchatbot",embedding=embeddings)


prompt = ChatPromptTemplate.from_template(prompt_template)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model='llama3-70b-8192')

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = docsearch.as_retriever()

retrieval_chain = create_retrieval_chain(retriever,document_chain)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


st.title("Medical ChatBot")
input = st.text_input("Ask anything related to medicine or diseases : ")

if input:
    st.session_state['chat_history'].append(('You',input))
    print("start")
    response = retrieval_chain.invoke({"input" : input})
    # print(response['answer'])
    print("end")
    st.session_state['chat_history'].append(('Bot',response['answer']))
    st.write(response['answer'])

if st.session_state['chat_history']:
    st.subheader("Chat history:")

for role,text in st.session_state['chat_history']:
    st.write(f'{role} : {text}')