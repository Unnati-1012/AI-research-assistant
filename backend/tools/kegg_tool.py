import re
import requests


class KEGGTool:
    def _extract_search_term(self, query):
        q = (query or "").lower()
        noise_terms = [
            "kegg", "pathway", "pathways", "image", "images", "diagram", "visual",
            "figure", "of", "show", "me", "please"
        ]

        cleaned = q
        for term in noise_terms:
            cleaned = cleaned.replace(term, " ")

        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or q.strip()

    def _build_image_url(self, pathway_id):
        pid = (pathway_id or "").replace("path:", "").strip()
        if not pid:
            return ""

        if pid.startswith("map"):
            prefix = "map"
        elif len(pid) >= 3:
            prefix = pid[:3]
        else:
            prefix = "map"

        return f"https://www.kegg.jp/kegg/pathway/{prefix}/{pid}.png"

    def run(self, query):
        try:
            search_term = self._extract_search_term(query)
            candidates = [search_term]

            if "-" in search_term:
                candidates.append(search_term.replace("-", " "))
            if " signaling" in search_term:
                candidates.append(search_term.replace(" signaling", ""))

            raw_text = ""
            used_term = search_term

            for term in dict.fromkeys([c for c in candidates if c]):
                url = f"https://rest.kegg.jp/find/pathway/{term}"
                response = requests.get(url, timeout=12)
                if response.status_code == 200 and response.text.strip():
                    raw_text = response.text
                    used_term = term
                    break

            if not raw_text:
                return {
                    "query": query,
                    "search_term": search_term,
                    "entries": [],
                    "error": "No KEGG pathway matches found"
                }

            entries = []
            for line in raw_text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                pid = parts[0].strip().replace("path:", "")
                name = parts[1].strip()
                entries.append({
                    "id": pid,
                    "name": name,
                    "kegg_url": f"https://www.kegg.jp/pathway/{pid}",
                    "image_url": self._build_image_url(pid)
                })

            return {
                "query": query,
                "search_term": used_term,
                "entries": entries[:8]
            }

        except Exception as e:
            return {
                "query": query,
                "entries": [],
                "error": str(e)
            }