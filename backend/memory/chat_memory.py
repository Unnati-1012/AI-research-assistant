class ChatMemory:

    def __init__(self):
        self.history = []

    def add(self, query, response):
        self.history.append((query, response))

    def get_context(self):

        return "\n".join(
            [f"User: {q}\nAI: {r}" for q, r in self.history]
        )