import requests

class UniProtTool:

    def run(self, query):

        # Extract protein name (simple version)
        protein = query.split()[-1]

        url = f"https://rest.uniprot.org/uniprotkb/search?query={protein}&format=json"

        response = requests.get(url)

        if response.status_code != 200:
            return "Error fetching UniProt data"

        data = response.json()

        try:
            result = data["results"][0]

            name = result["primaryAccession"]
            description = result["proteinDescription"]["recommendedName"]["fullName"]["value"]

            return {
                "protein_id": name,
                "description": description
            }

        except:
            return "No data found in UniProt"