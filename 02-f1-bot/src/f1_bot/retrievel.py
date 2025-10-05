import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# Load environment variables
load_dotenv()

# Retrieve API keys and URLs from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GEMINI_API_KEY)


# Connect to Qdrant vector store
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="f1_data",
    retrieval_mode=RetrievalMode.DENSE,
)



# Create a retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 2})



# Define the prompt template
PROMPT_TEMPLATE = """
You are an AI Assistant with extensive knowledge of Formula One racing.
Use the context below to enhance your responses with the latest information from Wikipedia, the official F1 website, and other sources.
If the context does not include the required information, respond based on your existing knowledge.
Do not mention the sources of your information or whether the context includes or excludes any details.
Format responses using markdown where applicable and do not return images.

---------
**Context:**
{context}
---------

**Question:** {question}
**Answer:**
""" 




# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY)



# Create the RAG pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    | llm
    | StrOutputParser()
)



# Function to get the response
def get_response(question):
    return rag_chain.invoke(question)


get_response("Who won the 2023 Formula One World Championship?") 