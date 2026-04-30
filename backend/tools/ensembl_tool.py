import re
import requests


class EnsemblTool:

    def _extract_entity(self, query):
        ensembl_match = re.search(r"\bENS[A-Z]*\d{9,}\b", query.upper())
        if ensembl_match:
            return ensembl_match.group(0), "id"

        tokens = re.findall(r"[A-Za-z0-9_-]+", query)
        ignore = {
            "genomic", "information", "for", "about", "gene", "show", "me", "the",
            "details", "annotation", "annotations", "location", "human", "ensembl"
        }

        candidates = [t for t in tokens if t.lower() not in ignore]
        if not candidates:
            return "", "symbol"
        return candidates[-1], "symbol"

    def _fetch_lookup(self, entity, mode):
        headers = {"Content-Type": "application/json"}
        if mode == "id":
            url = f"https://rest.ensembl.org/lookup/id/{entity}?expand=1;content-type=application/json"
        else:
            url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{entity}?expand=1;content-type=application/json"

        response = requests.get(url, headers=headers, timeout=12)
        if response.status_code != 200:
            return None
        return response.json()

    def run(self, query):
        try:
            entity, mode = self._extract_entity(query)
            if not entity:
                return {"error": "No gene symbol or Ensembl ID found in query"}

            data = self._fetch_lookup(entity, mode)
            if not data and mode == "symbol":
                # Fallback: if symbol lookup fails, try as ID.
                data = self._fetch_lookup(entity, "id")

            if not data:
                return {"error": f"No Ensembl record found for '{entity}'"}

            ensembl_id = data.get("id", "")

            return {
                "query": query,
                "source": "Ensembl",
                "input_entity": entity,
                "ensembl_id": ensembl_id,
                "gene_symbol": data.get("display_name", ""),
                "description": data.get("description", ""),
                "biotype": data.get("biotype", ""),
                "species": data.get("species", ""),
                "assembly_name": data.get("assembly_name", ""),
                "chromosome": data.get("seq_region_name", ""),
                "start": data.get("start", ""),
                "end": data.get("end", ""),
                "strand": data.get("strand", ""),
                "ensembl_url": f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={ensembl_id}" if ensembl_id else ""
            }

        except Exception as e:
            return {"error": f"Ensembl error: {str(e)}"}