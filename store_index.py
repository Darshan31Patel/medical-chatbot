from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Create new vector database
vectorstore = PineconeVectorStore(index='mchatbot', pinecone_api_key= os.getenv("PINECONE_API_KEY") ,embedding=embeddings)

docsearch = vectorstore.from_texts(texts=[t.page_content for t in text_chunks],embedding=embeddings,index_name='mchatbot')