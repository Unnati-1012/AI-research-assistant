from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq
import json
import os
import requests
import sys
import re

# Add backend directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# Import all tools
from tools.uniprot_tool import UniProtTool
from tools.kegg_tool import KEGGTool
from tools.ncbi_tool import NCBITool
from tools.string_tool import STRINGTool
from tools.pubchem_tool import PubChemTool
from tools.ensembl_tool import EnsemblTool
from tools.papers_tool import PapersTool
from tools.clinvar_tool import ClinVarTool
from tools.web_search import WebSearch

# Load environment variables first
load_dotenv()

app = FastAPI()

# Initialize Groq client with error handling
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    raise

# ── TOOL REGISTRY ────────────────────────────────────────────────────────────
# Define all available tools with descriptions for the agent
TOOLS_REGISTRY = {
    "search_uniprot": {
        "name": "search_uniprot",
        "description": "Search UniProt database for protein information. Use this when user asks about specific proteins, their functions, structures, or sequences.",
        "tool": UniProtTool()
    },
    "search_kegg": {
        "name": "search_kegg",
        "description": "Search KEGG database for metabolic pathways, biological pathways, and genes involved in pathways. Use this for pathway analysis and metabolic information.",
        "tool": KEGGTool()
    },
    "search_ncbi": {
        "name": "search_ncbi",
        "description": "Search NCBI database for gene sequences, genetic information, and genomic data. Use this for gene queries and sequence analysis.",
        "tool": NCBITool()
    },
    "search_string": {
        "name": "search_string",
        "description": "Search STRING database for protein-protein interactions and network analysis. Use this when analyzing protein interactions or building interaction networks.",
        "tool": STRINGTool()
    },
    "search_pubchem": {
        "name": "search_pubchem",
        "description": "Search PubChem database for chemical compounds, drugs, and molecules. Use this for compound and drug information queries.",
        "tool": PubChemTool()
    },
    "search_ensembl": {
        "name": "search_ensembl",
        "description": "Search Ensembl database for gene annotations, genomic locations, and human genes. Use this for human gene information and annotations.",
        "tool": EnsemblTool()
    },
    "search_papers": {
        "name": "search_papers",
        "description": "Search PubMed for research papers. Use this for paper discovery, literature review queries, and citation-style results.",
        "tool": PapersTool()
    },
    "search_clinvar": {
        "name": "search_clinvar",
        "description": "Search ClinVar for clinically relevant variants and mutations for a gene.",
        "tool": ClinVarTool()
    },
    "search_images": {
        "name": "search_images",
        "description": "Search Google Images (Serper) for scientific diagrams and visual references.",
        "tool": WebSearch()
    }
}

# Common protein PDB IDs for structure queries
COMMON_PROTEINS = {
    "hemoglobin": {"pdb_id": "2HHB", "title": "Hemoglobin", "organism": "Homo sapiens"},
    "insulin": {"pdb_id": "4AIL", "title": "Insulin", "organism": "Homo sapiens"},
    "myoglobin": {"pdb_id": "1A6M", "title": "Myoglobin", "organism": "Physeter catodon"},
    "brca1": {"pdb_id": "1JMN", "title": "BRCA1 RING Domain", "organism": "Homo sapiens"},
    "antibody": {"pdb_id": "1A6U", "title": "Antibody", "organism": "Mus musculus"},
    "enzyme": {"pdb_id": "1TIM", "title": "Triose-Phosphate Isomerase", "organism": "Plasmodium falciparum"},
    "tp53": {"pdb_id": "1TSR", "title": "TP53 (p53) Tumor Suppressor Protein", "organism": "Homo sapiens"},
    "egfr": {"pdb_id": "1NQL", "title": "Epidermal Growth Factor Receptor", "organism": "Homo sapiens"},
}

def is_scientific_query(query: str) -> bool:
    """Determine if the query is scientific/biomedical or general conversation"""
    query_lower = query.lower()
    
    # Scientific keywords - these always indicate a scientific query
    scientific_keywords = [
        "protein", "uniprot", "gene", "ncbi", "pathway", "kegg", "interaction", "string",
        "compound", "drug", "pubchem", "chemical", "ensembl", "genomic", "dna", "rna",
        "sequence", "mutation", "amino acid", "enzyme", "receptor", "antibody", "peptide",
        "metabolic", "signaling", "network", "ppi", "structure", "3d", "pdb", "fold",
        "binding", "ligand", "regulation", "transcription", "expression", "chromosome",
        "allele", "snp", "cryo", "x-ray", "nmr", "annotation", "genomic location",
        "clinvar", "variant", "variants", "pathogenic", "mutation", "mutations",
        "ensg", "enst", "ensp", "ensembl id",
        "paper", "papers", "publication", "publications", "journal", "doi", "pubmed",
        "literature", "review", "reviews",
        # Common protein/gene names
        "insulin", "hemoglobin", "myoglobin", "brca1", "tp53", "egfr", "antibody",
        "enzyme", "kinase", "collagen", "hemoglobin", "myosin", "actin", "histone"
    ]
    
    # If ANY scientific keyword is found, it's ALWAYS a scientific query
    for keyword in scientific_keywords:
        if keyword in query_lower:
            return True
    
    # Otherwise, it's conversational
    return False

def analyze_query_intent(query: str) -> list:
    """Analyze user query and determine which tools to use automatically"""
    query_lower = query.lower()
    tools_to_call = []

    compound_structure_keywords = [
        "2d structure", "2d", "chemical structure", "compound structure",
        "molecule structure", "smiles", "inchi", "pubchem", "drug structure",
        "conformer", "3d conformer"
    ]
    compound_keywords = [
        "compound", "drug", "molecule", "chemical", "pubchem",
        "chemical structure", "active ingredient", "ligand", "conformer"
    ]
    is_compound_structure_query = any(keyword in query_lower for keyword in compound_structure_keywords)
    has_ensembl_identifier = bool(re.search(r"\bENS[A-Z]*\d{9,}\b", query.upper()))

    genomic_keywords = [
        "genomic", "genome", "genomic information", "genomic location",
        "chromosome location", "locus", "coordinate", "coordinates"
    ]
    is_genomic_query = any(keyword in query_lower for keyword in genomic_keywords)
    mutation_keywords = [
        "mutation", "mutations", "variant", "variants", "pathogenic",
        "clinvar", "disease-associated", "disease associated"
    ]
    is_mutation_query = any(keyword in query_lower for keyword in mutation_keywords)

    image_keywords = ["image", "images", "diagram", "figure", "visual", "picture", "illustration", "flowchart"]
    is_image_query = any(keyword in query_lower for keyword in image_keywords)

    pathway_keywords = ["pathway", "kegg", "metabolic", "signaling", "regulation", "cascade", "interaction map"]
    is_pathway_query = any(keyword in query_lower for keyword in pathway_keywords)

    # Pathway diagram requests should fetch KEGG + image search together.
    if is_image_query and is_pathway_query:
        return ["search_kegg", "search_images"]

    # Route image requests directly to Google Images search.
    if is_image_query:
        return ["search_images"]
    
    # Protein/UniProt keywords
    protein_keywords = ["protein", "uniprot", "sequence", "amino acid", "enzyme", "receptor", "antibody", "peptide"]
    if any(keyword in query_lower for keyword in protein_keywords):
        tools_to_call.append("search_uniprot")
    
    # Gene/NCBI keywords
    gene_keywords = ["gene", "ncbi", "dna", "sequence", "genetic", "chromosome", "allele", "mutation", "snp"]
    if any(keyword in query_lower for keyword in gene_keywords):
        tools_to_call.append("search_ncbi")

    # Variants / ClinVar queries
    if is_mutation_query:
        tools_to_call.insert(0, "search_clinvar")
        if "clinvar" in query_lower and "ncbi" not in query_lower:
            tools_to_call = [tool for tool in tools_to_call if tool != "search_ncbi"]
            tools_to_call = [tool for tool in tools_to_call if tool != "search_ensembl"]
    
    # Pathway/KEGG keywords
    if any(keyword in query_lower for keyword in pathway_keywords):
        tools_to_call.append("search_kegg")
    
    # Interaction/STRING keywords
    interaction_keywords = ["interaction", "network", "string", "protein-protein", "ppi", "associate", "bind"]
    if any(keyword in query_lower for keyword in interaction_keywords):
        tools_to_call.append("search_string")
    
    # Compound/PubChem keywords
    if any(keyword in query_lower for keyword in compound_keywords) or is_compound_structure_query:
        tools_to_call.append("search_pubchem")

    # For compound structure requests, avoid irrelevant protein defaults.
    if is_compound_structure_query and "search_uniprot" in tools_to_call:
        tools_to_call = [tool for tool in tools_to_call if tool != "search_uniprot"]
    
    # Ensembl/Human gene keywords
    ensembl_keywords = [
        "human gene", "annotation", "genomic location", "ensembl", "transcription", "exon",
        "genomic", "genome", "chromosome location", "locus", "coordinate", "coordinates"
    ]
    if any(keyword in query_lower for keyword in ensembl_keywords):
        tools_to_call.append("search_ensembl")

    if has_ensembl_identifier:
        if "search_ensembl" not in tools_to_call:
            tools_to_call.insert(0, "search_ensembl")
        # Ensembl identifiers should not be routed to NCBI unless user explicitly asks for it.
        if "ncbi" not in query_lower:
            tools_to_call = [tool for tool in tools_to_call if tool != "search_ncbi"]

    # For genomic information requests, prefer Ensembl over NCBI unless user explicitly asked for NCBI.
    if is_genomic_query and "ncbi" not in query_lower:
        tools_to_call = [tool for tool in tools_to_call if tool != "search_ncbi"]
        if "search_ensembl" not in tools_to_call:
            tools_to_call.insert(0, "search_ensembl")

    # Papers / literature search keywords
    paper_keywords = [
        "paper", "papers", "research", "article", "articles", "journal",
        "publication", "publications", "doi", "pubmed", "literature", "review"
    ]
    if any(keyword in query_lower for keyword in paper_keywords):
        tools_to_call.append("search_papers")
    
    # PDB/Structure keywords
    structure_keywords = ["structure", "3d", "pdb", "fold", "conformation", "x-ray", "nmr", "cryo"]
    if any(keyword in query_lower for keyword in structure_keywords):
        # For structures, we'll handle via pdb detection later
        pass
    
    # If no specific tools matched, try general search (multiple tools)
    if not tools_to_call:
        # Default to searching multiple databases for unknown queries
        tools_to_call = ["search_uniprot", "search_ncbi", "search_kegg"]
    
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(tools_to_call))

def build_system_prompt() -> str:
    """Build system prompt for the agent with tool descriptions"""
    tools_desc = "Available tools:\n"
    for tool_id, tool_info in TOOLS_REGISTRY.items():
        tools_desc += f"\n- {tool_info['name']}: {tool_info['description']}"
    
    return f"""You are noviq.ai, an AI assistant specializing in biomedical research and molecular biology. You're honest about being an AI.

{tools_desc}

Guidelines:
- You are noviq.ai - a biomedical research AI
- Be honest that you're an AI (not human)
- Answer questions clearly and accurately
- Don't pretend to be human or make up personal experiences
- For scientific topics, synthesize and cite the database information
- Speak naturally and conversationally
- Be friendly and helpful
- Keep responses easy to understand
- IMPORTANT: Start conversations with varied, innovative opening sentences. Avoid repetitive phrases like "Let's dive into", "Based on the data", "You've gathered", etc. Be creative and unique in how you begin each response."""


def determine_query_type(tool_results: dict) -> str:
    query_type = "general"
    if "search_kegg" in tool_results and "search_images" in tool_results:
        query_type = "pathway_image"
    elif "search_images" in tool_results:
        query_type = "image"
    elif "search_clinvar" in tool_results:
        query_type = "variants"
    elif "search_papers" in tool_results:
        query_type = "papers"
    elif "search_pubchem" in tool_results:
        query_type = "compound"
    elif "pdb" in tool_results:
        query_type = "structure"
    elif "search_ncbi" in tool_results or "search_ensembl" in tool_results:
        query_type = "gene"
    elif "search_kegg" in tool_results:
        query_type = "pathway"
    elif "search_string" in tool_results:
        query_type = "interaction"
    elif "search_uniprot" in tool_results:
        query_type = "protein"
    return query_type


def build_answer_instruction(query_type: str) -> str:
    if query_type == "variants":
        return (
            "Answer in mutation-report style. Start with a one-line direct answer, then list key variants with "
            "clinical significance, review status, and ClinVar variation IDs. Keep it concise and data-first. "
            "Do not include generic background paragraphs unless asked."
        )
    if query_type == "gene":
        return (
            "Answer in genomic-summary style. Start with gene identity and coordinates, then biotype, assembly/species, "
            "and key interpretation. Keep focus on the queried gene. Avoid generic filler."
        )
    if query_type == "papers":
        return (
            "Answer in literature-summary style. Begin with 1-2 sentence synthesis of the evidence, then provide brief "
            "takeaways across the listed papers. Avoid inventing citations."
        )
    if query_type == "interaction":
        return (
            "Answer in network-analysis style. Summarize strongest partners first, mention high-confidence scores and "
            "biological relevance. Keep response structured and specific to retrieved interactions."
        )
    if query_type == "compound":
        return (
            "Answer in compound-profile style. Start with compound identity, then key properties (formula, molecular weight, "
            "SMILES/IUPAC) and requested structure/conformer context."
        )
    if query_type == "pathway":
        return "Answer in pathway-summary style with pathway names, IDs, and biological implications specific to the results."
    if query_type == "pathway_image":
        return (
            "Answer in combined pathway-visual style: briefly explain the KEGG pathway and the visual diagram found, "
            "then emphasize how the image relates to the pathway. Keep it concise and source-specific."
        )
    if query_type == "protein":
        return "Answer in protein-summary style using accession, protein name, and key function from retrieved data."
    if query_type == "image":
        return "Answer in visual-summary style: briefly explain what diagrams/images were found and how they relate to the query."
    return (
        "Provide a clear, query-specific answer using retrieved data only. Avoid repetitive templates and avoid generic "
        "sections when not needed."
    )

def execute_tool(tool_name: str, query: str) -> dict:
    """Execute a tool and return results"""
    if tool_name not in TOOLS_REGISTRY:
        return {"error": f"Tool {tool_name} not found"}
    
    try:
        tool = TOOLS_REGISTRY[tool_name]["tool"]
        print(f"  🔧 Executing {tool_name} with query: {query}")
        if tool_name == "search_images":
            result = tool.run_images(query)
        else:
            result = tool.run(query)
        
        if result:
            print(f"  ✓ {tool_name}: Success")
            return {tool_name: result}
        else:
            print(f"  ⚠️  {tool_name}: No data returned")
            return {tool_name: {"error": "No data found"}}
            
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ {tool_name}: Error - {error_msg[:100]}")
        return {tool_name: {"error": error_msg}}

def search_pdb_database(protein_name: str) -> dict:
    """Search RCSB PDB for protein structures"""
    try:
        print(f"🔍 Searching PDB for: {protein_name}")
        
        # Check common proteins first
        for common_name, pdb_data in COMMON_PROTEINS.items():
            if common_name.lower() in protein_name.lower():
                print(f"✓ Found in common proteins: {common_name}")
                return pdb_data
        
        # Try RCSB PDB API
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query_dict = {
            "query_type": "text",
            "return_type": "entry",
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {"q": protein_name}
            }
        }
        
        response = requests.get(
            search_url,
            params={"q": json.dumps(query_dict)},
            timeout=8
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("result_set", [])
            
            if results and len(results) > 0:
                first_result = results[0]
                pdb_id = first_result.get("identifier", "").strip().upper()
                
                if pdb_id:
                    return {
                        "pdb_id": pdb_id,
                        "title": first_result.get("title", "Unknown Structure"),
                        "organism": ""
                    }
        
        return None
        
    except Exception as e:
        print(f"✗ PDB search error: {e}")
        return None

def extract_tool_calls(response_text: str) -> list:
    """Extract tool calls from LLM response"""
    tool_calls = []
    
    # Look for tool calls in format: [TOOL:tool_name:query]
    pattern = r'\[TOOL:(\w+):([^\]]+)\]'
    matches = re.findall(pattern, response_text)
    
    for tool_name, query in matches:
        if tool_name in TOOLS_REGISTRY:
            tool_calls.append({"tool": tool_name, "query": query.strip()})
    
    return tool_calls

def detect_structure_request(query: str, context: str = "") -> dict:
    """Detect if the response mentions a structure query and fetch PDB data"""
    query_lower = query.lower()
    context_lower = context.lower()

    compound_structure_keywords = [
        "2d structure", "2d", "chemical structure", "compound structure",
        "molecule structure", "smiles", "inchi", "pubchem", "drug",
        "conformer", "3d conformer"
    ]
    # Compound structure queries should be handled by PubChem, not PDB.
    if any(keyword in query_lower for keyword in compound_structure_keywords):
        return {}
    
    # Check for structure request keywords
    structure_keywords = ["structure", "show me", "display", "3d", "pdb", "fold", "conformation"]
    if not any(keyword in query_lower for keyword in structure_keywords):
        return {}
    
    # Try to extract protein name from context
    # Common protein names from COMMON_PROTEINS
    protein_to_search = None
    
    # Check if query itself contains protein name
    for common_name in COMMON_PROTEINS.keys():
        if common_name in query_lower:
            protein_to_search = common_name
            break
    
    # If not found in query, check context (previous response)
    if not protein_to_search:
        for common_name in COMMON_PROTEINS.keys():
            if common_name in context_lower:
                protein_to_search = common_name
                break
    
    # Try to find any protein identifier
    if not protein_to_search:
        # Look for protein IDs like A4GXA9, 2HHB, etc.
        import re
        # PDB ID pattern (4 characters: letters and numbers)
        pdb_pattern = r'\b([A-Z0-9]{4})\b'
        pdb_matches = re.findall(pdb_pattern, query_lower + " " + context_lower)
        if pdb_matches:
            # Assume first match is the PDB ID we want
            pdb_id = pdb_matches[0].upper()
            return {
                "pdb": {
                    "pdb_id": pdb_id,
                    "title": f"Protein Structure {pdb_id}",
                    "organism": ""
                }
            }
    
    # If we found a protein name, search for it
    if protein_to_search:
        pdb_data = search_pdb_database(protein_to_search)
        if pdb_data:
            return {"pdb": pdb_data}
    
    return {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Noviq AI running"}

@app.post("/chat")
async def chat_stream(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "")
        messages = data.get("messages", [])
        
        if not query:
            return {
                "response": "Please provide a query",
                "tools_used": [],
                "data": {},
                "status": "error"
            }
        
        def event_generator():
            try:
                # Prepare messages for the agent
                system_prompt = build_system_prompt()
                
                # Build message history with proper format
                if messages:
                    messages_to_send = [
                        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                        for msg in messages
                    ]
                else:
                    messages_to_send = [{"role": "user", "content": query}]
                
                print(f"\n{'='*60}")
                print(f"🤖 USER QUERY: {query}")
                print(f"{'='*60}")
                
                # ── STAGE 1: DETERMINE QUERY TYPE ──
                is_scientific = is_scientific_query(query)
                print(f"\n🔍 Query Type: {'SCIENTIFIC' if is_scientific else 'CONVERSATIONAL'}")
                
                if not is_scientific:
                    # ── CONVERSATIONAL MODE: Just use LLM ──
                    print(f"\n💬 Stage 1: Conversational mode - using LLM directly...")
                    yield f"data: {json.dumps({'stage': 'thinking', 'message': '💭 Thinking...'})}\n\n"
                    
                    agent_request = [
                        {"role": "system", "content": "You are noviq.ai, a biomedical research AI. Provide detailed, comprehensive, informative responses. Give thorough explanations with good context and depth. Be honest about being an AI."},
                        *messages_to_send
                    ]
                    
                    yield f"data: {json.dumps({'stage': 'response', 'message': '📝 Generating response...'})}\n\n"
                    
                    stream = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=agent_request,
                        stream=True,
                        temperature=0.7
                    )
                    
                    full_text = ""
                    
                    for chunk in stream:
                        try:
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if delta and hasattr(delta, 'content') and delta.content:
                                    token = delta.content
                                    full_text += token
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                        except Exception as chunk_error:
                            print(f"Error processing chunk: {chunk_error}")
                            continue
                    
                    # Final message for conversational
                    final_msg = {'done': True, 'response': full_text, 'tools_used': [], 'query_type': 'conversation', 'data': {}, 'status': 'success'}
                    yield f"data: {json.dumps(final_msg)}\n\n"
                
                else:
                    # ── SCIENTIFIC MODE: Analyze query and call tools ──
                    print(f"\n📊 Stage 1: Analyzing scientific query...")
                    yield f"data: {json.dumps({'stage': 'thinking', 'message': '🔍 Understanding your scientific query...'})}\n\n"
                    
                    tools_to_call = analyze_query_intent(query)
                    tool_results = {}
                    
                    # ── STAGE 2: AUTO-EXECUTE RELEVANT TOOLS ──
                    if tools_to_call:
                        print(f"\n🔧 Stage 2: Auto-executing {len(tools_to_call)} relevant tools...")
                        msg_dict = {'stage': 'tools', 'message': f'🔧 Querying {len(tools_to_call)} data sources...'}
                        yield f"data: {json.dumps(msg_dict)}\n\n"
                        
                        for i, tool_name in enumerate(tools_to_call, 1):
                            clean_name = tool_name.replace("search_", "").upper()
                            msg_dict = {'stage': 'tools', 'message': f'⏳ Querying {clean_name}...', 'progress': i, 'total': len(tools_to_call)}
                            yield f"data: {json.dumps(msg_dict)}\n\n"
                            
                            result = execute_tool(tool_name, query)
                            tool_results.update(result)
                            
                            msg_dict = {'stage': 'tools', 'message': f'✓ {clean_name} retrieved', 'progress': i, 'total': len(tools_to_call)}
                            yield f"data: {json.dumps(msg_dict)}\n\n"
                    
                    # ── STAGE 3: CHECK FOR STRUCTURE REQUESTS ──
                    print(f"\n🔬 Stage 3: Checking for structure requests...")
                    msg_dict = {'stage': 'tools', 'message': '🔬 Loading 3D structures...'}
                    yield f"data: {json.dumps(msg_dict)}\n\n"
                    
                    # Build context from conversation history
                    context = ""
                    for msg in messages_to_send:
                        context += msg.get("content", "") + " "
                    
                    structure_data = detect_structure_request(query, context)
                    if structure_data:
                        tool_results.update(structure_data)
                        yield f"data: {json.dumps({'stage': 'tools', 'message': '✓ Structure loaded'})}\n\n"
                    
                    # ── STAGE 4: AGENT SYNTHESIS ──
                    print(f"\n💭 Stage 4: Agent synthesizing results...")
                    msg_dict = {'stage': 'synthesizing', 'message': '💭 Generating comprehensive answer...'}
                    yield f"data: {json.dumps(msg_dict)}\n\n"

                    query_type = determine_query_type(tool_results)
                    answer_instruction = build_answer_instruction(query_type)
                    
                    # Prepare context with tools data
                    tool_results_text = json.dumps(tool_results, indent=2)
                    if tool_results:
                        context = (
                            f"User query: {query}\n\n"
                            f"Query type: {query_type}\n\n"
                            f"Database results:\n{tool_results_text}\n\n"
                            f"Writing instruction: {answer_instruction}"
                        )
                    else:
                        context = (
                            "No specific data was found in the databases. Provide a brief answer, acknowledge missing "
                            "database hits, and suggest the next best query refinement."
                        )
                    
                    agent_request = [
                        {"role": "system", "content": "You are noviq.ai, a biomedical research AI. Provide comprehensive, detailed answers based on data provided. Be thorough - include mechanisms, significance, implications, and clinical context. Explain concepts clearly with good depth. Honest about being an AI."},
                        *messages_to_send,
                        {"role": "user", "content": context}
                    ]
                    
                    # ── STAGE 5: STREAM FINAL RESPONSE ──
                    yield f"data: {json.dumps({'stage': 'response', 'message': '📝 Generating response...'})}\n\n"
                    
                    stream = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=agent_request,
                        stream=True,
                        temperature=0.7
                    )
                    
                    full_text = ""
                    
                    for chunk in stream:
                        try:
                            if chunk.choices and len(chunk.choices) > 0:
                                delta = chunk.choices[0].delta
                                if delta and hasattr(delta, 'content') and delta.content:
                                    token = delta.content
                                    full_text += token
                                    yield f"data: {json.dumps({'token': token})}\n\n"
                        except Exception as chunk_error:
                            print(f"Error processing chunk: {chunk_error}")
                            continue
                    
                    # Determine query type based on tool results
                    query_type = determine_query_type(tool_results)
                    
                    # Send tools data
                    if tool_results:
                        tools_message = json.dumps({'tools': tool_results, 'query_type': query_type})
                        print(f"📨 Sending tools data: {len(tool_results)} tools")
                        yield f"data: {tools_message}\n\n"
                    
                    # Final message
                    final_msg = {'done': True, 'response': full_text, 'tools_used': list(tool_results.keys()), 'query_type': query_type, 'data': {}, 'status': 'success'}
                    yield f"data: {json.dumps(final_msg)}\n\n"
                
            except Exception as e:
                error_msg = str(e)
                print(f"Event generator error: {error_msg}")
                import traceback
                traceback.print_exc()
                error_response = {'done': True, 'response': f'Error: {error_msg}', 'status': 'error'}
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
        
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response": f"Server error: {str(e)}",
            "status": "error"
        }
