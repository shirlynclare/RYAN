from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
index_name="clare"
pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index("clare")

#Creating Embeddings for Each of The Text Chunks & storing
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
