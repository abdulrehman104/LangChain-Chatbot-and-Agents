import streamlit as st
from f1_bot.retrievel import get_response

# Set page configuration
st.set_page_config(
    page_title="Formula 1 Chatbot",
    page_icon="🏎️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1 {
        color: #e10600;
    }
    .stChatInput input {
        border-radius: 5px;
        padding: 10px;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatMessage.user {
        background-color: #e10600;
        color: white;
    }
    .stChatMessage.assistant {
        background-color: #f0f2f6;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("🏎️ Formula 1 Chatbot")
    st.markdown("""
    Welcome to the Formula 1 Chatbot!  
    Ask any question about Formula 1, and the AI will provide you with detailed answers.
    """)
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Enter your question in the chat box.")
    st.markdown("2. Press Enter or click the send button.")
    st.markdown("3. View the AI's response in the chat.")
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and LangChain.")

# Main content
st.title("Formula 1 Chatbot")
st.markdown("Ask me anything about Formula 1!")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display conversation history
for exchange in st.session_state.conversation:
    if exchange["role"] == "user":
        with st.chat_message("user"):
            st.markdown(exchange["content"])
    elif exchange["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(exchange["content"])

# Chat input for the question
question = st.chat_input("Enter your question about Formula 1:")

# Process the question and generate a response
if question:
    # Add user's question to the conversation history
    st.session_state.conversation.append({"role": "user", "content": question})

    with st.spinner("Fetching your answer..."):
        # Call the backend function to get the response
        response = get_response(question)
        # Add AI's response to the conversation history
        st.session_state.conversation.append(
            {"role": "assistant", "content": response})

    # Rerun the app to update the conversation history display
    st.rerun()
