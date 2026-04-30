from llm.llm_manager import LLMManager

class QueryRouter:

    def __init__(self):
        self.llm = LLMManager()

    def route(self, query):

        decision = self.llm.route(query)

        if "ORCHESTRATOR" in decision:
            return "ORCHESTRATOR"

        return "CHAT"