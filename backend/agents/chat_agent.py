from routing.query_router import QueryRouter
from llm.llm_manager import LLMManager
from memory.chat_memory import ChatMemory
from agents.orchestrator_agent import OrchestratorAgent
from agents.tool_selector import ToolSelector
from tools.web_search import WebSearch
from tools.uniprot_tool import UniProtTool
from tools.kegg_tool import KEGGTool
from tools.ncbi_tool import NCBITool
from tools.string_tool import STRINGTool
from tools.pubchem_tool import PubChemTool
from tools.ensembl_tool import EnsemblTool
from tools.pdb_tool import PDBTool

class ChatAgent:

    def __init__(self):

        self.router = QueryRouter()
        self.llm = LLMManager()
        self.memory = ChatMemory()
        self.orchestrator = OrchestratorAgent()
        self.tool_selector = ToolSelector()
        self.web = WebSearch()
        
        # Initialize tools
        self.tools = {
            "uniprot": UniProtTool(),
            "kegg": KEGGTool(),
            "ncbi": NCBITool(),
            "string": STRINGTool(),
            "pubchem": PubChemTool(),
            "ensembl": EnsemblTool(),
            "pdb": PDBTool()
        }

    def needs_web_search(self, query):

        keywords = ["latest", "recent", "current", "news", "update"]

        return any(word in query.lower() for word in keywords)

    def handle(self, query):
        try:
            tools_used = []
            tool_results = {}
            
            # Select appropriate tools for this query
            selected_tools = self.tool_selector.select(query)
            
            # Execute selected tools
            for tool_name in selected_tools:
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name].run(query)
                        tool_results[tool_name] = result
                        tools_used.append(tool_name)
                    except Exception as e:
                        tool_results[tool_name] = f"Error: {str(e)}"
            
            # Build context for LLM
            context = f"Query: {query}\n\n"
            if tool_results:
                context += "Tool Results:\n"
                for tool_name, result in tool_results.items():
                    context += f"\n{tool_name}: {result}\n"
            
            # Get response from LLM with tool data as context
            prompt = f"""You are a bioinformatics expert assistant. Use the following tool results to provide an accurate answer to the user's query.

{context}

Please provide a detailed and accurate response based on the tool results provided above. If you have specific data from the tools, use that data. Do not make assumptions about the gene or protein - use the actual data provided.
"""
            
            response = self.llm.chat(prompt)
            
            # Store in memory
            self.memory.add(query, response)
            
            # Return structured format
            return {
                "response": response,
                "tools_used": tools_used,
                "data": tool_results
            }
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "tools_used": [],
                "data": {}
            }