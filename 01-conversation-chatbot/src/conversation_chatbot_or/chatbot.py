import streamlit as st
from conversation_chatbot_or.app import ask_chatbot

# Page config
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #ffffff !important;
    }
    section.main {
        background-color: #ffffff !important;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #000000;
    }
    .chat-message {
        padding: 0;
        border-radius: 0;
        margin-bottom: 1rem;
        background: none !important;
        color: #000000 !important;
        box-shadow: none !important;
    }
    .chat-message.user, .chat-message.bot {
        background: none !important;
        color: #000000 !important;
        box-shadow: none !important;
    }
    .stChatMessage {
        background: none !important;
        color: #000000 !important;
        box-shadow: none !important;
    }
    .stChatInput {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.markdown("---")
    st.markdown("""
    ### About
    This is an AI-powered chatbot built with Streamlit and OpenRouter API.
    
    ### Features
    - Real-time chat interface
    - Powered by Google's Gemma model
    - Clean and modern UI
    """)

    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask_chatbot(prompt)
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
