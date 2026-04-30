import requests

class PDBTool:

    def run(self, query):

        protein = query.split()[-1]

        url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={{\"query\":{{\"type\":\"terminal\",\"service\":\"text\",\"parameters\":{{\"value\":\"{protein}\"}}}},\"return_type\":\"entry\"}}"

        response = requests.get(url)

        if response.status_code != 200:
            return "Error fetching PDB data"

        data = response.json()

        try:
            pdb_id = data["result_set"][0]["identifier"]

            return {
                "protein": protein,
                "pdb_id": pdb_id
            }

        except:
            return "No structure found"