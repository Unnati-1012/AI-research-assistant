import os
import asyncio
import logging
from groq import Groq
from dotenv import load_dotenv

import os
import asyncio
from groq import Groq

def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)

async def answer_with_groq_async(prompt: str):
    loop = asyncio.get_running_loop()
    def blocking_call():
        client = get_client()
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        if completion.choices and completion.choices[0].message.content:
            return completion.choices[0].message.content
        return "⚠️ Could not generate an answer."
    return await loop.run_in_executor(None, blocking_call)

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment variables")
    return Groq(api_key=api_key)

async def answer_with_groq_async(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Async-safe wrapper to generate an answer using the Groq LLM.
    """
    loop = asyncio.get_running_loop()

    def blocking_call():
        try:
            client = get_client()
            completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model
            )

            logger.info(f"Groq full response: {completion}")

            if completion.choices and completion.choices[0].message.content:
                answer = completion.choices[0].message.content
                logger.info(f"Groq returned answer: {answer}")
                return answer
            else:
                logger.warning("Groq returned no answer.")
                return "I couldn’t generate an answer."
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return f"Error: {e}"

    return await loop.run_in_executor(None, blocking_call)
