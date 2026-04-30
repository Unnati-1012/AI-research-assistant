from llm.groq_client import GroqClient

class LLMManager:

    def __init__(self):
        self.client = GroqClient()

    def chat(self, prompt):

        messages = [
            {"role": "user", "content": prompt}
        ]

        return self.client.generate(messages)

    def route(self, query):

        prompt = f"""
        Classify the query:

        CHAT → general explanation
        ORCHESTRATOR → requires database/tools

        Query: {query}

        Answer ONLY: CHAT or ORCHESTRATOR
        """

        return self.chat(prompt)

    def format_response(self, raw_data):

        prompt = f"""
        Convert the following data into a clean, user-friendly answer:

        {raw_data}
        """

        return self.chat(prompt)
    def answer_with_context(self, query, context):

        prompt = f"""
        Use the following context to answer the question clearly and concisely.

        Context:
        {context}

        Question:
        {query}
        """

        return self.chat(prompt)

    def extract_entity(self, query):

        prompt = f"""
        Extract the biological entity from the query.

        Query: {query}

        Return only the entity name.
        """

        return self.chat(prompt)