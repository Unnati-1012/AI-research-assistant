from llm.llm_manager import LLMManager

class ToolSelector:

    def __init__(self):
        self.llm = LLMManager()

    def select(self, query):

        prompt = f"""
        You are a bioinformatics expert.

        Available tools:
        - uniprot (protein data)
        - pdb (structure)
        - ncbi (gene info)
        - kegg (pathways)
        - ensembl (genome)
        - pubchem (drugs)
        - string (interactions)

        Based on the query, return the BEST tool(s).

        Query: {query}

        Answer ONLY tool names (comma-separated).
        """

        response = self.llm.chat(prompt)

        return [tool.strip().lower() for tool in response.split(",")]