import chainlit as cl
from pdf_chatbot.retriever_chain import query_pdf_qa
from pdf_chatbot.data_processing import (
    load_pdf, 
    split_text_into_chunks, 
    initialize_qdrant_collection, 
    create_vector_store
    )


@cl.on_chat_start
async def start_chat():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(content="Please upload a text file to begin!", accept={"text/plain": [".pdf"]}, max_size_mb=5).send()

    file = files[0]
    pdf_path = file.path

    # Send initial loading message and capture handle for updates
    initial_msg = cl.Message(content="🔄 Processing PDF. Please wait.....")
    await initial_msg.send()

    # 1️⃣ Load PDF pages
    pages = load_pdf(pdf_path)

    # 2️⃣ Split into chunks
    chunks = split_text_into_chunks(pages)

    # 3️⃣ Initialize Qdrant collection
    collection_name = "chat_with_pdf"
    initialize_qdrant_collection(collection_name, vector_size=768)

    # 4️⃣ Embed & store chunks
    create_vector_store(chunks, collection_name)

    # Final update when ready
    initial_msg.content = "✅ Your PDF has been processed—feel free to ask any questions about it."
    await initial_msg.update()


@cl.on_message
async def main(message: cl.Message):
    # Send loading indicator while querying RAG chain
    query_msg = cl.Message(content="🔄 Retrieving answer... Please wait.")
    await query_msg.send()

    # Call retrieval + generation function
    answer = query_pdf_qa(message.content)

    # Update the message with the final answer
    query_msg.content = answer
    await query_msg.update() 
