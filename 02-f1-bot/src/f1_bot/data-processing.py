import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Load Environment Variables
load_dotenv()

# Qdrant credentials
qdrant_api_key = os.environ.get("QDRANT_API_KEY")
qdrant_url = os.environ.get("QDRANT_URL")
gemini_api_key = os.environ.get("GEMINI_API_KEY")


# F1 URLs for various resources
F1_URLS = [
    "https://en.wikipedia.org/wiki/Formula_One",
    "https://www.formula1.com/en/latest/all",
    "https://www.forbes.com/sites/brettknight/2023/11/29/formula-1s-highest-paid-drivers-2023/?sh=12bdb942463f",
    "https://www.autosport.com/f1/news/history-of-female-f1-drivers-including-grand-prix-starters-and-test-drivers/10584871/",
    "https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship",
    "https://en.wikipedia.org/wiki/2022_Formula_One_World_Championship",
    "https://en.wikipedia.org/wiki/List_of_Formula_One_World_Drivers_Champions",
    "https://en.wikipedia.org/wiki/2024_Formula_One_World_Championship",
    "https://www.formula1.com/en/results.html/2024/races.html",
    "https://www.formula1.com/en/racing/2024.html",
]


# Load all the Web Pages
def load_documents(urls):
    """Loads documents from a list of URLs."""
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(web_path=url)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading URL {url}: {e}")
    return documents


# Splits this docs into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Splits documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = splitter.split_documents(documents)
    print(f"Split the documents into {len(docs)} chunks.")
    return docs


# Setup Qdrant Vector Database
def initialize_qdrant_collection(collection_name, vector_size):
    """Initializes or recreates a Qdrant collection."""
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Check if the collection exists
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting it...")
        client.delete_collection(collection_name)

    # Create a new collection with the correct dimensions
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' initialized successfully.")


# Create Embedding and Add this embedding to Vector Store
def embed_and_store_documents(documents, collection_name):
    """Generates embeddings for documents and stores them in Qdrant."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=gemini_api_key)

    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        force_recreate=True  # Ensure the collection is recreated if it exists
    )
    print("Documents successfully stored in Qdrant.")


# Main function to run the pipeline
def main():
    # Step 1: Load documents
    print("Loading documents...")
    documents = load_documents(F1_URLS)
    print(f"Loaded {len(documents)} documents from the URLs.")

    # Step 2: Split documents into chunks
    print("Splitting documents into chunks...")
    docs = split_documents(documents)
    print(f"Split the documents into {len(docs)} chunks.")

    # Step 3: Initialize Qdrant collection
    collection_name = "f1_data"
    vector_size = 768  # GoogleGenerativeAIEmbeddings has 768 dimensions
    print(f"Initializing Qdrant collection '{collection_name}'...")
    initialize_qdrant_collection(collection_name, vector_size)

    # Step 4: Embed and store documents in Qdrant
    print("Embedding and storing documents in Qdrant...")
    embed_and_store_documents(docs, collection_name)
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
