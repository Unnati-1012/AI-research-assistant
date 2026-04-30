import re
import requests


class ClinVarTool:

    def _extract_gene(self, query):
        tokens = re.findall(r"[A-Za-z0-9_-]+", query)
        ignore = {
            "what", "which", "mutations", "mutation", "variant", "variants",
            "exist", "in", "for", "of", "show", "me", "the", "clinvar",
            "pathogenic", "disease", "associated", "known"
        }
        candidates = [t for t in tokens if t.lower() not in ignore]
        if not candidates:
            return ""
        return candidates[-1].upper()

    def _search_ids(self, gene_symbol, limit=10):
        term = f"{gene_symbol}[gene]"
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=clinvar&term={term}&retmode=json&retmax={limit}&sort=relevance"
        )
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            return []
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _summaries(self, ids):
        if not ids:
            return []
        ids_csv = ",".join(ids)
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=clinvar&id={ids_csv}&retmode=json"
        )
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            return []
        data = response.json()
        result = data.get("result", {})
        uids = result.get("uids", [])

        rows = []
        for uid in uids:
            rec = result.get(uid, {})
            germline = rec.get("germline_classification", {}) if isinstance(rec.get("germline_classification"), dict) else {}
            clinical_significance = (
                germline.get("description")
                or rec.get("clinical_significance")
                or rec.get("clinical_significance_description")
                or "Not specified"
            )

            rows.append({
                "variation_id": uid,
                "title": rec.get("title", ""),
                "accession": rec.get("accession", ""),
                "clinical_significance": clinical_significance,
                "review_status": germline.get("review_status") or rec.get("review_status", "Not specified"),
                "last_evaluated": germline.get("last_evaluated") or rec.get("last_evaluated", ""),
                "variant_type": rec.get("obj_type", ""),
                "clinvar_url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{uid}/"
            })

        return rows

    def run(self, query):
        gene = self._extract_gene(query)
        if not gene:
            return {"error": "No gene symbol found for ClinVar search"}

        ids = self._search_ids(gene, limit=10)
        variants = self._summaries(ids)

        return {
            "gene": gene,
            "source": "ClinVar",
            "variants": variants
        }
