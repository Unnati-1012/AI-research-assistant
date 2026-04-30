from tools.uniprot_tool import UniProtTool
from tools.pdb_tool import PDBTool
from tools.kegg_tool import KEGGTool
from tools.ncbi_tool import NCBITool
from tools.string_tool import STRINGTool
from tools.pubchem_tool import PubChemTool
from tools.ensembl_tool import EnsemblTool

class OrchestratorAgent:

    def __init__(self):
        self.tools = {
            "uniprot": UniProtTool(),
            "pdb": PDBTool(),
            "kegg": KEGGTool(),
            "ncbi": NCBITool(),
            "string": STRINGTool(),
            "pubchem": PubChemTool(),
            "ensembl": EnsemblTool()
        }

    def handle(self, query):

        tools_used = []
        data = {}

        # 🔥 SIMPLE ROUTING (you can upgrade later)
        if "protein" in query.lower():
            res = self.tools["uniprot"].run(query)
            data["uniprot"] = res
            tools_used.append("uniprot")

        if "structure" in query.lower():
            res = self.tools["pdb"].run(query)
            data["pdb"] = res
            tools_used.append("pdb")

        if "pathway" in query.lower():
            res = self.tools["kegg"].run(query)
            data["kegg"] = res
            tools_used.append("kegg")

        # Add more rules as needed...

        final_text = "Here is the analysis:\n"
        for k, v in data.items():
            final_text += f"\n{k.upper()}:\n{v}"

        return {
            "response": final_text,
            "tools_used": tools_used,
            "data": data
        }