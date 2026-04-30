import re
import requests
import xml.etree.ElementTree as ET


class PapersTool:
    def _extract_topic(self, query):
        words = re.findall(r"[A-Za-z0-9-]+", query)
        ignore = {
            "find", "show", "get", "papers", "paper", "research", "articles",
            "on", "about", "for", "latest", "recent", "review", "reviews",
            "me", "the", "a", "an", "in"
        }
        topic_words = [w for w in words if w.lower() not in ignore]
        return " ".join(topic_words).strip()

    def _search_pmids(self, topic, limit=5):
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={topic}&retmode=json&retmax={limit}&sort=relevance"
        )
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            return []
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _fetch_summaries(self, pmids):
        if not pmids:
            return {}
        ids = ",".join(pmids)
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={ids}&retmode=json"
        )
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            return {}
        data = response.json()
        return data.get("result", {})

    def _fetch_abstracts(self, pmids):
        if not pmids:
            return {}
        ids = ",".join(pmids)
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={ids}&retmode=xml"
        )
        response = requests.get(url, timeout=25)
        if response.status_code != 200:
            return {}

        abstracts = {}
        root = ET.fromstring(response.text)
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                continue
            pmid = pmid_elem.text.strip()

            abstract_texts = []
            for node in article.findall(".//Abstract/AbstractText"):
                if node.text:
                    abstract_texts.append(node.text.strip())
            abstract = " ".join(abstract_texts)
            abstracts[pmid] = abstract

        return abstracts

    def run(self, query):
        topic = self._extract_topic(query) or query.strip()
        pmids = self._search_pmids(topic, limit=6)

        if not pmids:
            return {"topic": topic, "source": "PubMed", "papers": []}

        summaries = self._fetch_summaries(pmids)
        abstracts = self._fetch_abstracts(pmids)

        papers = []
        for pmid in pmids:
            entry = summaries.get(pmid, {})
            title = entry.get("title", "")
            pubdate = entry.get("pubdate", "")
            source = entry.get("source", "")

            authors_list = entry.get("authors", [])
            author_names = [a.get("name", "") for a in authors_list if a.get("name")]
            authors = ", ".join(author_names[:6])

            article_ids = entry.get("articleids", [])
            doi = ""
            for aid in article_ids:
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
                    break

            abstract = abstracts.get(pmid, "")
            brief = abstract[:360].strip()
            if brief and len(abstract) > 360:
                brief += "..."

            papers.append({
                "pmid": pmid,
                "title": title,
                "authors": authors,
                "journal": source,
                "year": pubdate,
                "doi": doi,
                "brief": brief,
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        return {
            "topic": topic,
            "source": "PubMed",
            "papers": papers
        }
