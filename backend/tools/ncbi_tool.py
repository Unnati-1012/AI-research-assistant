import requests
import re
import xml.etree.ElementTree as ET

class NCBITool:

    def run(self, query):
        # Try to extract gene ID from query (numbers)
        gene_id_match = re.search(r'\b(\d{6,})\b', query)
        gene_id = gene_id_match.group(1) if gene_id_match else None
        
        # Extract gene name (non-numeric terms, remove common query words)
        query_words = query.split()
        ignore_words = {'sequence', 'information', 'about', 'of', 'for', 'gene', 'genes', 'what', 'is', 'the', 'data', 'details', 'info'}
        gene_name_words = [word for word in query_words if not word.isdigit() and word.lower() not in ignore_words]
        gene_name = ' '.join(gene_name_words) if gene_name_words else None
        
        # If we have a gene ID, use it to fetch gene information
        if gene_id:
            return self._fetch_by_gene_id(gene_id, gene_name)
        
        # Otherwise search by gene name
        if gene_name:
            return self._search_by_gene_name(gene_name)
        
        return {"error": "No gene information provided"}

    def _fetch_by_gene_id(self, gene_id, gene_name):
        """Fetch specific gene information using gene ID"""
        # Use the eutils API to get gene information
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&id={gene_id}&rettype=xml"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return {"error": f"Error fetching NCBI data for gene ID {gene_id}"}
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Extract gene name from the XML
            gene_symbol = None
            gene_description = None
            
            # Try to extract symbol
            for elem in root.iter('Gene-ref_locus'):
                gene_symbol = elem.text
                break
            
            # Try to extract description
            for elem in root.iter('Gene-ref_desc'):
                gene_description = elem.text
                break
            
            # If we found real gene data, return it
            if gene_symbol or gene_description:
                return {
                    "gene_id": gene_id,
                    "gene_name": gene_symbol or gene_name or "Unknown",
                    "description": gene_description or "",
                    "source": "NCBI Gene Database"
                }
            
            # Fallback if we didn't find good data
            if gene_name:
                return {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "source": "NCBI Gene Database"
                }
            
            return {"error": f"No detailed gene information found for ID {gene_id}"}
        
        except Exception as e:
            return {"error": f"Error retrieving gene information: {str(e)}"}

    def _search_by_gene_name(self, gene_name):
        """Search for gene by name"""
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term={gene_name}&retmode=json"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return {"error": "Error searching NCBI database"}
            
            data = response.json()
            
            if data["esearchresult"]["idlist"]:
                gene_id = data["esearchresult"]["idlist"][0]
                # Now fetch detailed info for this gene ID
                return self._fetch_by_gene_id(gene_id, gene_name)
            
            return {"error": f"No genes found matching '{gene_name}'"}
        
        except Exception as e:
            return {"error": f"Error searching NCBI: {str(e)}"}