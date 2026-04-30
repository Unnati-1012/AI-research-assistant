import os
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the ASYNCHRONOUS Groq client once
groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

async def answer_with_groq_async(prompt: str, stream: bool = False):
    """
    Generates an answer using the Groq API. This function supports both
    streaming and non-streaming modes.

    Args:
        prompt: The user's prompt.
        stream: If True, returns an async generator yielding response chunks.
                If False, returns the full response string.
    """
    params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama-3.3-70b-versatile",  # Using a fast model for streaming
    }

    if not stream:
        # --- Standard non-streaming behavior ---
        # Await the single API call and return the result directly.
        chat_completion = await groq_client.chat.completions.create(**params)
        return chat_completion.choices[0].message.content
    else:
        # --- Streaming behavior ---
        # This defines a new async generator function that will be returned.
        async def generator():
            # Start the streaming API call
            stream_completion = await groq_client.chat.completions.create(**params, stream=True)
            # Iterate over the async stream of chunks
            async for chunk in stream_completion:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content # Yield each piece of content as it arrives
        
        # Return the generator object itself, NOT the result of calling it.
        return generator()