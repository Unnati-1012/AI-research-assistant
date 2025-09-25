import os
from groq import Groq

# Initialize Groq client with API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)

# Create a chat completion
chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models"
        }
    ]
)

# Print the assistant's reply
print(chat_completion.choices[0].message["content"])
