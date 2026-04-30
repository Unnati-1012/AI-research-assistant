import re
import requests


class PubChemTool:

    def _extract_compound(self, query):
        words = re.findall(r"[A-Za-z0-9-]+", query)
        ignore = {
            "show", "me", "the", "a", "an", "of", "for", "about", "with", "and",
            "2d", "3d", "structure", "compound", "chemical", "molecule", "drug",
            "display", "give", "find", "what", "is", "conformer", "model", "view"
        }
        candidates = [w for w in words if w.lower() not in ignore]
        if not candidates:
            return ""
        return candidates[-1]

    def _get_cid(self, compound):
        cid_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{compound}/cids/JSON"
        response = requests.get(cid_url, timeout=15)
        if response.status_code != 200:
            return None
        data = response.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None

    def _get_properties(self, cid):
        prop_fields = "Title,MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName"
        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{prop_fields}/JSON"
        response = requests.get(prop_url, timeout=15)
        if response.status_code != 200:
            return {}
        data = response.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        return props[0] if props else {}

    def run(self, query):
        compound = self._extract_compound(query)
        if not compound:
            return {"error": "No compound provided"}

        query_lower = query.lower()
        wants_3d = any(keyword in query_lower for keyword in ["3d", "conformer", "3-d", "3d structure"])

        cid = self._get_cid(compound)
        if not cid:
            return {"error": f"No compound found for '{compound}'"}

        properties = self._get_properties(cid)

        return {
            "compound": properties.get("Title", compound),
            "cid": cid,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "pubchem_3d_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=3D-Conformer",
            "image_url_2d": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=large",
            "sdf_url_2d": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=2d",
            "sdf_url_3d": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d",
            "requested_view": "3d" if wants_3d else "2d",
            "molecular_formula": properties.get("MolecularFormula", ""),
            "molecular_weight": properties.get("MolecularWeight", ""),
            "canonical_smiles": properties.get("CanonicalSMILES", ""),
            "iupac_name": properties.get("IUPACName", "")
        }