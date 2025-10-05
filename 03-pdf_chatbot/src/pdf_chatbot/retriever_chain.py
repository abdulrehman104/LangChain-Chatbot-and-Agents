import os
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 Purpose: Build a PDF-chat RAG chain using Qdrant + Gemini for QA
# ─────────────────────────────────────────────────────────────────────────────

# Load and validate environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

print(f"GEMINI_API_KEY: GEMINI_API_KEY is set: {bool(GEMINI_API_KEY)}")
print(f"QDRANT_API_KEY: QDRANT_API_KEY is set: {bool(QDRANT_API_KEY)}")
print(f"QDRANT_URL: QDRANT_URL is set: {bool(QDRANT_URL)}")         


# ─────────────────────────────────────────────────────────────────────────────
# 📦 Embeddings Initialization
# ─────────────────────────────────────────────────────────────────────────────
# Google Generative AI embeddings (Gemini) for document indexing
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# 🔍 Qdrant Retriever Setup
# ─────────────────────────────────────────────────────────────────────────────
# Connect to an existing Qdrant collection and enable retrieval
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name="chat_with_pdf",
    retrieval_mode=RetrievalMode.DENSE  # use dense (ANN) search
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# ─────────────────────────────────────────────────────────────────────────────
# 📝 Prompt Template for PDF QA
# ─────────────────────────────────────────────────────────────────────────────
# Designed to answer solely from the provided PDF context
PROMPT_TEMPLATE = """
You are an expert AI assistant that answers questions based entirely on the contents of an uploaded PDF document.
Use the provided context to craft a detailed, informative response.

If the answer cannot be found in the context, respond clearly:
"I’m sorry, but I couldn’t find information about that in the PDF you provided."

Do NOT reference external sources, metadata, or your own knowledge beyond the PDF.

---
Context:
{context}
---
Question: {question}
Answer:
"""


# ─────────────────────────────────────────────────────────────────────────────
# 🤖 LLM Initialization
# ─────────────────────────────────────────────────────────────────────────────
# Chat model (Gemini) for generating answers
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", google_api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# 🔗 Build the RAG Chain
# ─────────────────────────────────────────────────────────────────────────────
# 1) Retriever fetches PDF chunks
# 2) Template merges context + question
# 3) LLM generates answer
# 4) Parser ensures string output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    | llm
    | StrOutputParser()
)


# ─────────────────────────────────────────────────────────────────────────────
# 🎛️ Query Function
# ─────────────────────────────────────────────────────────────────────────────

def query_pdf_qa(question: str) -> str:
    """
    Run a RAG query: retrieve from PDF + generate an answer.
    :param question: User's question about the PDF
    :return: Generated answer text
    """
    try:
        # Invoke the chain, auto-fetching context under the hood
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        # Catch and report any runtime errors
        return f"Error during RAG query: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Example Execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_q = "Tell me about traditional methods in NLP"
    print(f"🗣️ Question: {sample_q}")
    print(f"💬 Answer: {query_pdf_qa(sample_q)}")


# def query_pdf_qa(question: str) -> str:
#     """
#     Run a RAG query: retrieve from PDF + generate an answer with step logs.
#     :param question: User's question about the PDF
#     :return: Generated answer text
#     """
#     try:
#         print("Retrieving relevant PDF context...")

#         start_retrieval = time.time()
#         # Manually perform retrieval for logging
#         context_docs = retriever.get_relevant_documents(question)
#         elapsed_retrieval = time.time() - start_retrieval
#         print(f"Retrieved {len(context_docs)} chunks in {elapsed_retrieval:.2f}s")

#         print("Formatting prompt for LLM...")
#         prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
#             context="\n".join([doc.page_content for doc in context_docs]), question=question)

#         print("Sending prompt to LLM...")
#         start_llm = time.time()
#         raw_response = llm.invoke(prompt)
#         elapsed_llm = time.time() - start_llm
#         print(f"Received response from LLM in {elapsed_llm:.2f}s")

#         print("Parsing final output...")
#         answer = StrOutputParser().parse(raw_response)
#         print("Query completed successfully!")
#         return answer

#     except Exception as e:
#         # Catch and report any runtime errors
#         error_msg = f"❌ Error during RAG query: {e}"
#         print(error_msg)
#         return error_msg
