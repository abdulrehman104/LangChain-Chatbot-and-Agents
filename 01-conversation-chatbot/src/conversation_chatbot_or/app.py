import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file 
load_dotenv()

# === Configuration ===
BASE_URL = "https://openrouter.ai/api/v1"
MODEL    = "google/gemma-2-9b-it"

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable in your .env file.")

def ask_chatbot(question: str) -> str:
    """Send a single user question to OpenRouter and return the assistant's reply."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/yourusername/your-repo",  # Required by OpenRouter
        "X-Title": "AI Chatbot"  # Optional, but good practice
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,  # Add some creativity to responses
        "max_tokens": 1000   # Limit response length
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if not resp.ok:
            print(f"❌ HTTP {resp.status_code} error from API:")
            print(resp.text)
            raise RuntimeError(f"API call failed with status {resp.status_code}")

        data = resp.json()
        if "choices" not in data or not data["choices"]:
            print("⚠️ Unexpected JSON structure:")
            print(json.dumps(data, indent=2))
            raise RuntimeError("No 'choices' in response")

        return data["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

# This part is only used when running app.py directly
if __name__ == "__main__":
    print("🤖 Chatbot is online! Type your question and press Enter.")
    print("Type exit() to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("🤖 Goodbye!")
            break

        try:
            answer = ask_chatbot(user_input)
            print(f"Bot: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")
            break
