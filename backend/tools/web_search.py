import requests
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

class WebSearch:
    def __init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.serper_search_url = "https://google.serper.dev/search"
        self.serper_images_url = "https://google.serper.dev/images"

    def _headers(self):
        return {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }

    def _is_trusted_domain(self, url):
        try:
            host = (urlparse(url).hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]

            trusted_suffixes = (".edu", ".org", ".gov")
            trusted_exact = {"nih.gov", "ncbi.nlm.nih.gov"}

            if host in trusted_exact:
                return True
            return host.endswith(trusted_suffixes)
        except Exception:
            return False

    def _is_scientific_domain(self, url):
        try:
            host = (urlparse(url).hostname or "").lower()
            if host.startswith("www."):
                host = host[4:]

            # Trust academic and scientific domains
            trusted_suffixes = (".edu", ".org", ".gov", ".ac.uk")
            trusted_exact = {"nih.gov", "ncbi.nlm.nih.gov", "kegg.jp", "string-db.org", "ensembl.org", "uniprot.org"}

            if host in trusted_exact:
                return True
            if host.endswith(trusted_suffixes):
                return True
            
            # Also allow major scientific publishers
            scientific_publishers = ["nature", "science", "cell", "plos", "springer", "elsevier"]
            return any(pub in host for pub in scientific_publishers)
        except Exception:
            return False

    def run_images(self, query):
        try:
            if not self.serper_api_key:
                return {"error": "SERPER_API_KEY not found in environment variables"}

            payload = {
                "q": query,
                "num": 10
            }

            response = requests.post(self.serper_images_url, json=payload, headers=self._headers(), timeout=15)
            if response.status_code != 200:
                return {"error": f"Serper Images API error: {response.status_code}"}

            data = response.json()
            images = []
            for item in data.get("images", [])[:8]:
                link = item.get("link", "")
                if not self._is_scientific_domain(link):
                    continue

                images.append({
                    "title": item.get("title", ""),
                    "image_url": item.get("imageUrl", ""),
                    "thumbnail_url": item.get("thumbnailUrl", ""),
                    "source": item.get("source", ""),
                    "link": link
                })

            return {
                "query": query,
                "source": "Google Images (Serper)",
                "source_filter": "scientific-domains (.edu/.org/.gov/.ac.uk/nature/science/plos)",
                "images": images
            }

        except Exception as e:
            return {"error": str(e)}

    def run(self, query):
        try:
            if not self.serper_api_key:
                return {"error": "SERPER_API_KEY not found in environment variables"}

            payload = {
                "q": query,
                "num": 10
            }

            response = requests.post(self.serper_search_url, json=payload, headers=self._headers(), timeout=10)

            if response.status_code != 200:
                return {"error": f"Serper API error: {response.status_code}"}

            data = response.json()

            # Extract results
            results = []
            for item in data.get("organic", [])[:5]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                })

            answer = data.get("answer", {}).get("answer") if data.get("answer") else None

            return {
                "query": query,
                "answer": answer,
                "results": results,
                "knowledge_graph": data.get("knowledgeGraph")
            }

        except Exception as e:
            return {"error": str(e)}