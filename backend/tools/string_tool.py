import re
import requests
from urllib.parse import quote_plus


class STRINGTool:

    def _extract_protein(self, query):
        # Prefer explicit gene/protein-like tokens (e.g., TP53, BRCA1) over trailing words.
        tokens = re.findall(r"[A-Za-z0-9_-]+", query)
        ignore = {
            "protein", "proteins", "interaction", "interactions", "network", "of",
            "for", "with", "show", "find", "get", "the", "and", "in"
        }

        for token in reversed(tokens):
            cleaned = token.strip()
            if not cleaned:
                continue
            if cleaned.lower() in ignore:
                continue
            if any(ch.isdigit() for ch in cleaned) or cleaned.isupper():
                return cleaned

        return tokens[-1] if tokens else ""

    def run(self, query):

        protein = self._extract_protein(query)
        if not protein:
            return {"error": "No protein provided for STRING search"}

        url = "https://string-db.org/api/json/network"
        params = {
            "identifiers": protein,
            "species": 9606,
            "required_score": 400,
            "limit": 20,
            "caller_identity": "noviq.ai"
        }

        response = requests.get(url, params=params, timeout=20)

        if response.status_code != 200:
            return {"error": "Error fetching STRING data"}

        edges = response.json()
        if not isinstance(edges, list) or not edges:
            encoded = quote_plus(protein)
            return {
                "query_protein": protein,
                "organism": "Homo sapiens",
                "resolved_identifier": protein,
                "network_url": f"https://string-db.org/cgi/network?identifiers={encoded}&species=9606",
                "network_embed_url": f"https://string-db.org/cgi/network?identifiers={encoded}&species=9606",
                "network_image_url": f"https://string-db.org/api/image/network?identifiers={encoded}&species=9606&required_score=400",
                "interactions": [],
                "source": "STRING"
            }

        interactions = []
        seen = set()
        resolved_identifier = protein

        for edge in edges:
            a = edge.get("preferredName_A", "")
            b = edge.get("preferredName_B", "")
            score = edge.get("score", 0)

            if a.upper() == protein.upper():
                partner = b
                resolved_identifier = edge.get("stringId_A") or resolved_identifier
            elif b.upper() == protein.upper():
                partner = a
                resolved_identifier = edge.get("stringId_B") or resolved_identifier
            else:
                # Fallback: keep edge if query protein matching is ambiguous.
                partner = b

            if not partner:
                continue

            key = (partner, round(float(score), 3))
            if key in seen:
                continue
            seen.add(key)

            interactions.append({
                "partner": partner,
                "string_id": edge.get("stringId_B") if partner == b else edge.get("stringId_A"),
                "score": round(float(score), 3),
                "neighborhood": round(float(edge.get("nscore", 0) or 0), 3),
                "fusion": round(float(edge.get("fscore", 0) or 0), 3),
                "cooccurrence": round(float(edge.get("pscore", 0) or 0), 3),
                "coexpression": round(float(edge.get("ascore", 0) or 0), 3),
                "experiments": round(float(edge.get("escore", 0) or 0), 3),
                "databases": round(float(edge.get("dscore", 0) or 0), 3),
                "textmining": round(float(edge.get("tscore", 0) or 0), 3)
            })

        interactions.sort(key=lambda item: item.get("score", 0), reverse=True)

        encoded_identifier = quote_plus(resolved_identifier)

        return {
            "query_protein": protein,
            "organism": "Homo sapiens",
            "source": "STRING",
            "resolved_identifier": resolved_identifier,
            "network_url": f"https://string-db.org/cgi/network?identifiers={encoded_identifier}&species=9606",
            "network_embed_url": f"https://string-db.org/cgi/network?identifiers={encoded_identifier}&species=9606",
            "network_image_url": f"https://string-db.org/api/image/network?identifiers={encoded_identifier}&species=9606&required_score=400",
            "interactions": interactions[:10]
        }