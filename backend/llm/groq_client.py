from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

class GroqClient:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # List of available models to try (in order of preference)
        self.available_models = [
            "mixtral-8x7b-32768",
            "llama-3.1-8b-instant", 
            "deepseek-r1-distill-llama-70b",
            "llama-3-8b-8192",
        ]

    def generate(self, messages, model=None):
        # If no model specified, use first available
        if model is None:
            model = self.available_models[0]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            
            # If model is decommissioned, try next one
            if "decommissioned" in error_msg or "model not found" in error_msg.lower():
                for alt_model in self.available_models:
                    if alt_model != model:
                        try:
                            response = self.client.chat.completions.create(
                                model=alt_model,
                                messages=messages,
                                temperature=0.7,
                                max_tokens=1024
                            )
                            return response.choices[0].message.content
                        except:
                            continue
            
            raise Exception(f"All models failed. Last error: {error_msg}")