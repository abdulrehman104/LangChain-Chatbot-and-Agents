import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Init embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("findify")

# Init vectorstore
vectorstore = PineconeVectorStore(index=index,embedding=embedding_model,text_key="Description")

