import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 Purpose: Build and index a PDF into Qdrant for RAG-based querying
# ─────────────────────────────────────────────────────────────────────────────

# Load environment variables from .env into os.environ
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

print(f"GEMINI_API_KEY: GEMINI_API_KEY is set: {bool(GEMINI_API_KEY)}")
print(f"QDRANT_API_KEY: QDRANT_API_KEY is set: {bool(QDRANT_API_KEY)}")
print(f"QDRANT_URL: QDRANT_URL is set: {bool(QDRANT_URL)}")


# ─────────────────────────────────────────────────────────────────────────────
# 📄 PDF Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_pdf(file_path: str):
    """
    Load a PDF from a local path or URL and return a list of Page Documents.
    Each document represents one page of the PDF.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDF: {file_path}")
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# 🔨 Text Splitting
# ─────────────────────────────────────────────────────────────────────────────
def split_text_into_chunks(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split the list of page Documents into smaller text chunks suitable for embedding.
    Uses overlapping windows to preserve context across splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(
        f"✅ Split into {len(chunks)} text chunks ({chunk_size=} / {chunk_overlap=} overlap)")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 🚀 Qdrant Collection Initialization
# ─────────────────────────────────────────────────────────────────────────────
def initialize_qdrant_collection(collection_name: str, vector_size: int):
    """
    Initialize (or recreate) a Qdrant collection with cosine similarity.
    Deletes existing collection to start fresh.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # If exists, delete and recreate for clean state
    if client.collection_exists(collection_name):
        print(f"⚠️ Collection '{collection_name}' exists. Deleting... 🎬")
        client.delete_collection(collection_name)

    # Create a new collection with specified vector dimension & cosine distance
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(
        f"✅ Qdrant collection '{collection_name}' initialized ({vector_size=} dims)")


# ─────────────────────────────────────────────────────────────────────────────
# ✨ Embedding & Upsert into Qdrant
# ─────────────────────────────────────────────────────────────────────────────
def create_vector_store(docs, collection_name: str):
    """
    Embed each text chunk with Gemini embeddings and push to Qdrant.
    Uses force_recreate to ensure a fresh index when re-running.
    """
    # Initialize embedding model (Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    # Upsert documents into Qdrant collection
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=True
    )
    print(
        f"✅ Indexed {len(docs)} chunks into Qdrant collection '{collection_name}'")


# ─────────────────────────────────────────────────────────────────────────────
# 🎬 Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    End-to-end PDF → Vector pipeline:
      1. Load PDF
      2. Split into chunks
      3. Initialize Qdrant
      4. Embed & store chunks
    """
    print("▶️ Starting PDF processing pipeline...")
    file_path = ("https://guidinglightcharity.org.uk/wp-content/uploads/2024/09/Generative-AI-Foundations-in-Python-Discover-key-techniques-and-navigate-modern-challenges-in-LLMs-Carlos-Rodriguez-Z-Library.pdf")

    # 1️⃣ Load PDF pages
    pages = load_pdf(file_path)
    print(f"Loaded {len(pages)} pages from PDF.")

    # 2️⃣ Split text into manageable chunks
    docs = split_text_into_chunks(pages)

    # 3️⃣ Setup Qdrant collection
    collection = "chat_with_pdf"
    vector_dims = 768  # dimensions for Gemini embeddings
    initialize_qdrant_collection(collection, vector_dims)
 
    # 4️⃣ Embed & upsert to Qdrant
    create_vector_store(docs, collection)

    print("✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
