import streamlit as st
import requests
import json
import streamlit.components.v1 as components
import re
import html
from typing import Optional
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="noviq.ai",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000/chat"

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme_param = st.query_params.get("theme")
if isinstance(theme_param, list):
    theme_param = theme_param[0] if theme_param else None
if theme_param in ("dark", "light"):
    st.session_state.theme = theme_param

THEME_VARS = {
    "dark": {
        "bg_deep": "#060b18",
        "bg_mid": "#0b1228",
        "bg_surface": "#0f1a35",
        "text_1": "#e8f4f2",
        "text_2": "#8fa8b0",
        "text_3": "#4a6670",
        "border": "rgba(0, 212, 180, 0.18)",
        "chip_bg": "#0f1a35",
    },
    "light": {
        "bg_deep": "#f7fbff",
        "bg_mid": "#edf4ff",
        "bg_surface": "#ffffff",
        "text_1": "#102033",
        "text_2": "#4f6478",
        "text_3": "#8393a3",
        "border": "rgba(15, 23, 42, 0.12)",
        "chip_bg": "#ffffff",
    },
}

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Montserrat:wght@600;700;800&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
    --accent: #00d4b4;
}

* { box-sizing: border-box; }
html, body, .stApp { 
    background: var(--bg-deep) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text-1);
    margin: 0 !important;
    padding: 0 !important;
}
#MainMenu, header, footer { display: none !important; }

/* SIDEBAR */
.stSidebar {
    background: var(--bg-mid) !important;
    border-right: 1px solid var(--border) !important;
}

.stSidebar [data-testid="stSidebarContent"] {
    background: var(--bg-mid) !important;
}

.stSidebar .stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 12px 16px !important;
    margin: 12px !important;
}

/* MAIN */
.main .block-container {
    max-width: 950px !important;
    padding: 100px 2rem 220px !important;
    margin-bottom: 0 !important;
}

[data-testid="stAppViewContainer"] .main {
    padding-bottom: 180px !important;
}

/* HEADER */
.nv-header {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 70px;
    background: var(--bg-mid);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 30px;
    z-index: 999;
}

.nv-header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nv-logo {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #00d4b4, #007c86);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.nv-title {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 20px;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

.nv-header-right {
    display: flex;
    gap: 12px;
}

.nv-header-actions {
    display: flex;
    gap: 12px;
    align-items: center;
}

.nv-header-btn {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    text-decoration: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.nv-theme-btn {
    width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    border-radius: 6px !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
}

.nv-theme-btn button {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    padding: 0 !important;
    border-radius: 6px !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
}

/* CHAT MESSAGES */
.nv-msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0;
}

.nv-bubble-user {
    background: linear-gradient(135deg, #00866e 0%, #005e4d 100%) !important;
    color: #e8f4f2 !important;
    padding: 12px 18px !important;
    border-radius: 18px 18px 4px 18px !important;
    max-width: 70%;
    word-wrap: break-word;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    line-height: 1.5;
}

.nv-msg-bot {
    display: flex;
    gap: 10px;
    margin: 16px 0;
    align-items: flex-start;
}

.nv-avatar {
    width: 32px;
    height: 32px;
    min-width: 32px;
    background: linear-gradient(135deg, #00d4b4, #007c86);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.nv-bubble-bot {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    padding: 12px 16px !important;
    border-radius: 4px 18px 18px 18px !important;
    max-width: 80%;
    word-wrap: break-word;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    line-height: 1.6;
}

/* EMPTY STATE */
.nv-empty {
    text-align: center;
    padding: 100px 20px 40px;
}

.nv-empty-icon { font-size: 48px; margin-bottom: 16px; }
.nv-empty-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 600 !important;
    margin-bottom: 12px;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #00D4FF 0%, #A78BFA 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    -webkit-text-fill-color: transparent;
}

.nv-empty-text {
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    color: var(--text-2);
    max-width: 500px;
    margin: 0 auto 32px;
    line-height: 1.6;
}

.nv-chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
}

.nv-chip {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-2) !important;
    padding: 8px 14px !important;
    border-radius: 20px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}

/* INPUT */
.nv-input-section {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    width: 100% !important;
    background: var(--bg-deep) !important;
    padding: 20px 2rem !important;
    z-index: 9999 !important;
    border-top: 1px solid var(--border) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}

.nv-input-container {
    max-width: 950px !important;
    margin: 0 auto !important;
    width: 100% !important;
}

section[data-testid="stForm"] {
    background: none !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    position: static !important;
}

.stForm {
    padding: 0 !important;
}

form {
    margin: 0 !important;
    padding: 0 !important;
}

.stTextInput > div > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.stTextInput input {
    background: transparent !important;
    color: var(--text-1) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}

.stTextInput input::placeholder { 
    color: var(--text-3) !important;
    font-family: 'Syne', sans-serif !important;
}

.stFormSubmitButton > button {
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

.stFormSubmitButton > button:hover,
.stFormSubmitButton > button:focus,
.stFormSubmitButton > button:active {
    background: linear-gradient(135deg, #00d4b4 0%, #007c86 100%) !important;
    color: var(--bg-deep) !important;
    border: none !important;
    box-shadow: 0 0 0 1px rgba(0, 212, 180, 0.25) !important;
}

/* PINNED CHAT INPUT */
[data-testid="stBottomBlockContainer"] {
    position: fixed !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 100% !important;
    z-index: 1000 !important;
    background: linear-gradient(180deg, rgba(6, 11, 24, 0) 0%, rgba(6, 11, 24, 0.75) 100%) !important;
    border-top: none !important;
    padding: 8px 16px 12px 16px !important;
}

[data-testid="stBottomBlockContainer"] > div,
[data-testid="stBottomBlockContainer"] [data-testid="stChatFloatingInputContainer"] {
    max-width: 950px !important;
    margin: 0 auto !important;
    width: 100% !important;
    background: transparent !important;
}

[data-testid="stChatInput"] {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    min-height: 44px !important;
    max-height: 180px !important;
    border-radius: 12px !important;
    background: var(--bg-surface) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border) !important;
    padding: 8px 12px !important;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] input:focus {
    outline: none !important;
    border: 1px solid rgba(0, 212, 180, 0.45) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 180, 0.12) !important;
}

[data-testid="stChatInput"] button {
    border-radius: 8px !important;
    background: var(--accent) !important;
    color: var(--bg-deep) !important;
    border: none !important;
    box-shadow: 0 0 0 1px rgba(0, 212, 180, 0.16) !important;
}

[data-testid="stChatInput"] button:hover,
[data-testid="stChatInput"] button:focus,
[data-testid="stChatInput"] button:active {
    background: linear-gradient(135deg, #00d4b4 0%, #007c86 100%) !important;
    color: var(--bg-deep) !important;
    border: none !important;
    box-shadow: 0 0 0 1px rgba(0, 212, 180, 0.22), 0 0 18px rgba(0, 212, 180, 0.12) !important;
}

@media (max-width: 768px) {
    .main .block-container {
        padding: 88px 1rem 190px !important;
    }

    [data-testid="stBottomBlockContainer"] {
        padding: 4px 10px 8px 10px !important;
    }
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--text-3); border-radius: 3px; }

</style>
""", unsafe_allow_html=True)

active_theme = THEME_VARS.get(st.session_state.theme, THEME_VARS["dark"])
st.markdown(
    f"""
    <style>
    :root {{
        --bg-deep: {active_theme['bg_deep']};
        --bg-mid: {active_theme['bg_mid']};
        --bg-surface: {active_theme['bg_surface']};
        --text-1: {active_theme['text_1']};
        --text-2: {active_theme['text_2']};
        --text-3: {active_theme['text_3']};
        --border: {active_theme['border']};
    }}
    .nv-chip {{ background: {active_theme['chip_bg']} !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── SESSION STATE ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── HEADER ────────────────────────────────────────────────────────────────────
toggle_theme = "light" if st.session_state.theme == "dark" else "dark"
toggle_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
st.markdown(
    f"""
    <div class="nv-header">
        <div class="nv-header-left">
            <div class="nv-logo">🧬</div>
            <div class="nv-title">noviq.ai</div>
        </div>
        <div class="nv-header-actions">
            <a class="nv-header-btn" href="?theme={toggle_theme}">{toggle_icon}</a>
            <a class="nv-header-btn" href="#">Login</a>
            <a class="nv-header-btn" href="#">Profile</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Chat History")
    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.rerun()

    if len(st.session_state.messages) > 0:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ── EMPTY STATE ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="nv-empty">
        <div class="nv-empty-icon">🔬</div>
        <div class="nv-empty-title">The search engine for biological science</div>
        <div class="nv-empty-text">
            Ask anything about proteins, genes, pathways, or drug targets. We search live scientific databases and bring back clear, accurate answers instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display messages
    for msg_index, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="nv-msg-user">
                <div class="nv-bubble-user">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Bot message with potential tool outputs
            query_type = msg.get("query_type", "")
            tools = msg.get("tools", {})
            
            # STRUCTURE QUERIES - Show PDB iframe (skip text response)
            if query_type == "structure" and tools and "pdb" in tools:
                pdb_data = tools["pdb"]
                pdb_id = pdb_data.get("pdb_id", "")
                if pdb_id:
                    st.markdown(f"**🔬 PDB Structure: {pdb_id}**")
                    # Embed RCSB PDB viewer
                    iframe_html = f"""
                    <iframe style="width: 100%; height: 600px; border: 1px solid rgba(0,212,180,0.2); border-radius: 8px; margin: 12px 0;"
                        src="https://www.rcsb.org/3d-view/{pdb_id}">
                    </iframe>
                    """
                    st.markdown(iframe_html, unsafe_allow_html=True)
                    if pdb_data.get("title"):
                        st.caption(f"📋 {pdb_data['title']}")
                
                # Also show the bot response text
                st.markdown(f"""
                <div class="nv-msg-bot">
                    <div class="nv-avatar">🧬</div>
                    <div class="nv-bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # GENE QUERIES - Show gene data in cards
            elif query_type == "gene" and tools and ("search_ncbi" in tools or "search_ensembl" in tools):
                if "search_ensembl" in tools:
                    gene_data = tools["search_ensembl"]
                    gene_source = "Ensembl"
                else:
                    gene_data = tools["search_ncbi"]
                    gene_source = "NCBI"
                st.markdown(f"### 🧬 Gene Information ({gene_source})")

                if isinstance(gene_data, dict) and gene_source == "Ensembl":
                    ensembl_url = gene_data.get("ensembl_url", "")
                    if ensembl_url:
                        st.markdown(f"🔗 Ensembl record: [{ensembl_url}]({ensembl_url})")
                
                # Parse string or dict format
                gene_items = []
                if isinstance(gene_data, str):
                    # Parse tab-separated or line-separated gene data
                    for line in gene_data.strip().split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            gene_items.append((key.strip().replace('**', ''), value.strip().replace('**', '')))
                elif isinstance(gene_data, dict):
                    if gene_source == "Ensembl":
                        chromosome = gene_data.get("chromosome", "")
                        start = gene_data.get("start", "")
                        end = gene_data.get("end", "")
                        coordinate = ""
                        if chromosome and start and end:
                            coordinate = f"chr{chromosome}:{start}-{end}"

                        gene_items = [
                            ("Gene Symbol", gene_data.get("gene_symbol", "")),
                            ("Ensembl ID", gene_data.get("ensembl_id", "")),
                            ("Description", gene_data.get("description", "")),
                            ("Biotype", gene_data.get("biotype", "")),
                            ("Genomic Coordinates", coordinate),
                            ("Chromosome", chromosome),
                            ("Start Position", start),
                            ("End Position", end),
                            ("Strand", gene_data.get("strand", "")),
                            ("Assembly", gene_data.get("assembly_name", "")),
                            ("Species", gene_data.get("species", "")),
                            ("Ensembl URL", gene_data.get("ensembl_url", "")),
                        ]
                        gene_items = [(k, v) for k, v in gene_items if v not in (None, "")]
                    else:
                        gene_items = list(gene_data.items())
                
                if gene_items:
                    col1, col2 = st.columns(2)
                    
                    for idx, (key, value) in enumerate(gene_items):
                        value_str = "" if value is None else str(value)
                        display_value = value_str
                        key_norm = key.lower().replace(" ", "_")

                        if isinstance(value, str) and value.startswith(("http://", "https://")):
                            display_value = f'<a href="{value}" target="_blank" style="color: #93c5fd; text-decoration: none; font-weight: 600;">{value}</a>'
                        elif key_norm == "ensembl_id" and value_str:
                            ensembl_link = f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={value_str}"
                            display_value = f'<a href="{ensembl_link}" target="_blank" style="color: #93c5fd; text-decoration: none; font-weight: 600;">{value_str}</a>'

                        with (col1 if idx % 2 == 0 else col2):
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%);
                                border-left: 5px solid #8b5cf6;
                                border-radius: 10px;
                                padding: 16px;
                                margin-bottom: 12px;
                                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2);
                                border: 1px solid rgba(139, 92, 246, 0.15);
                                transition: all 0.3s ease;
                            "
                            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(139, 92, 246, 0.12)'"
                            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(139, 92, 246, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2)'">
                                <div style="color: #c084fc; font-weight: 700; font-size: 11px; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1px;">{key}</div>
                                <div style="color: #f3e8ff; font-size: 13px; font-weight: 500;">{display_value}</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Keep narrative optional to avoid overwhelming data-centric output.
                with st.expander("AI narrative summary", expanded=False):
                    st.markdown(f"""
                    <div class="nv-msg-bot">
                        <div class="nv-avatar">🧬</div>
                        <div class="nv-bubble-bot">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # PATHWAY QUERIES - Show pathway diagram with better visualization
            elif query_type == "pathway" and tools and "search_kegg" in tools:
                pathway_data = tools["search_kegg"]
                st.markdown("### 🛤️ KEGG Pathway Information")
                pathway_records = []

                if isinstance(pathway_data, dict) and isinstance(pathway_data.get("entries"), list):
                    for item in pathway_data.get("entries", []):
                        if not isinstance(item, dict):
                            continue
                        pid = item.get("id", "")
                        if not pid:
                            continue
                        pathway_records.append({
                            "id": pid,
                            "name": item.get("name", ""),
                            "kegg_url": item.get("kegg_url", f"https://www.kegg.jp/pathway/{pid}"),
                            "image_url": item.get("image_url", "")
                        })
                elif isinstance(pathway_data, str):
                    for line in pathway_data.strip().split('\n'):
                        if not line.strip():
                            continue
                        parts = line.split('\t')
                        if len(parts) < 2:
                            continue
                        pid = parts[0].strip().replace("path:", "")
                        if not pid:
                            continue
                        prefix = "map" if pid.startswith("map") else (pid[:3] if len(pid) >= 3 else "map")
                        pathway_records.append({
                            "id": pid,
                            "name": parts[1].strip(),
                            "kegg_url": f"https://www.kegg.jp/pathway/{pid}",
                            "image_url": f"https://www.kegg.jp/kegg/pathway/{prefix}/{pid}.png"
                        })

                if pathway_records:
                    col1, col2 = st.columns(2)

                    for idx, item in enumerate(pathway_records):
                        with (col1 if idx % 2 == 0 else col2):
                            with st.container():
                                pathway_id = item.get("id", "")
                                pathway_name = item.get("name", "")
                                kegg_url = item.get("kegg_url", "")

                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(0, 212, 180, 0.15) 0%, rgba(0, 150, 136, 0.08) 100%);
                                    border-left: 5px solid #00d4b4;
                                    border-radius: 12px;
                                    padding: 20px;
                                    margin-bottom: 16px;
                                    box-shadow: 0 4px 15px rgba(0, 212, 180, 0.1), 0 2px 4px rgba(0, 0, 0, 0.2);
                                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                                    backdrop-filter: blur(10px);
                                    border: 1px solid rgba(0, 212, 180, 0.2);
                                    position: relative;
                                    overflow: hidden;
                                "
                                onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0, 212, 180, 0.2), 0 4px 8px rgba(0, 0, 0, 0.25)'"
                                onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 15px rgba(0, 212, 180, 0.1), 0 2px 4px rgba(0, 0, 0, 0.2)'">
                                    <div style="
                                        position: absolute;
                                        top: 0;
                                        left: 0;
                                        right: 0;
                                        height: 3px;
                                        background: linear-gradient(90deg, #00d4b4, #00a699, transparent);
                                    "></div>
                                    <h3 style="
                                        margin: 0 0 12px 0;
                                        color: #00f0d4;
                                        font-size: 16px;
                                        font-weight: 700;
                                        letter-spacing: 0.5px;
                                        text-transform: uppercase;
                                        display: flex;
                                        align-items: center;
                                        gap: 8px;
                                    ">
                                        🛤️ <a href="{kegg_url}" target="_blank" style="color: #00f0d4; text-decoration: none; transition: color 0.2s;">{pathway_id}</a>
                                    </h3>
                                    <p style="
                                        margin: 0;
                                        color: #d0f0ed;
                                        font-size: 14px;
                                        line-height: 1.6;
                                        font-weight: 400;
                                        letter-spacing: 0.3px;
                                    ">{pathway_name}</p>
                                </div>
                                """, unsafe_allow_html=True)

                    top_image = pathway_records[0].get("image_url", "")
                    if top_image:
                        st.markdown("**Official KEGG Pathway Diagram**")
                        st.image(top_image, use_container_width=True)
                
                # Show bot response below
                st.markdown(f"""
                <div class="nv-msg-bot">
                    <div class="nv-avatar">🧬</div>
                    <div class="nv-bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

            # PATHWAY IMAGE QUERIES - Show KEGG pathway plus image search together
            elif query_type == "pathway_image" and tools and ("search_kegg" in tools or "search_images" in tools):
                st.markdown("### 🛤️ KEGG Pathway + Visual Diagram")
                pathway_records = []

                if "search_kegg" in tools:
                    pathway_data = tools["search_kegg"]

                    if isinstance(pathway_data, dict) and isinstance(pathway_data.get("entries"), list):
                        for item in pathway_data.get("entries", []):
                            if not isinstance(item, dict):
                                continue
                            pid = item.get("id", "")
                            if not pid:
                                continue
                            pathway_records.append({
                                "id": pid,
                                "name": item.get("name", ""),
                                "kegg_url": item.get("kegg_url", f"https://www.kegg.jp/pathway/{pid}"),
                                "image_url": item.get("image_url", "")
                            })
                    elif isinstance(pathway_data, str):
                        for line in pathway_data.strip().split('\n'):
                            if not line.strip():
                                continue
                            parts = line.split('\t')
                            if len(parts) < 2:
                                continue
                            pid = parts[0].strip().replace("path:", "")
                            if not pid:
                                continue
                            prefix = "map" if pid.startswith("map") else (pid[:3] if len(pid) >= 3 else "map")
                            pathway_records.append({
                                "id": pid,
                                "name": parts[1].strip(),
                                "kegg_url": f"https://www.kegg.jp/pathway/{pid}",
                                "image_url": f"https://www.kegg.jp/kegg/pathway/{prefix}/{pid}.png"
                            })

                    if pathway_records:
                        st.markdown("**KEGG Pathway Details**")
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(pathway_records):
                            with (col1 if idx % 2 == 0 else col2):
                                pathway_id = item.get("id", "")
                                pathway_name = item.get("name", "")
                                kegg_url = item.get("kegg_url", "")
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(0, 212, 180, 0.15) 0%, rgba(0, 150, 136, 0.08) 100%);
                                    border-left: 5px solid #00d4b4;
                                    border-radius: 12px;
                                    padding: 16px;
                                    margin-bottom: 12px;
                                    border: 1px solid rgba(0, 212, 180, 0.2);
                                ">
                                    <div style="color:#00f0d4; font-weight:700; margin-bottom:8px;">🛤️ <a href="{kegg_url}" target="_blank" style="color:#00f0d4; text-decoration:none;">{pathway_id}</a></div>
                                    <div style="color:#d0f0ed; font-size:13px; line-height:1.5;">{pathway_name}</div>
                                </div>
                                """, unsafe_allow_html=True)

                        top_record = pathway_records[0]
                        top_image = top_record.get("image_url", "")
                        top_id = top_record.get("id", "")
                        top_url = top_record.get("kegg_url", "")

                        if top_image:
                            st.markdown("**Official KEGG Pathway Diagram**")
                            st.image(top_image, use_container_width=True, caption=f"Official KEGG pathway: {top_id}")
                            if top_url:
                                st.markdown(f"🔗 [View on KEGG]({top_url})")

                if "search_images" in tools:
                    image_data = tools["search_images"]
                    images = []
                    source = "Google Images"
                    source_filter = ""
                    if isinstance(image_data, dict):
                        source = image_data.get("source", source)
                        source_filter = image_data.get("source_filter", "")
                        images = image_data.get("images", []) if isinstance(image_data.get("images", []), list) else []

                    st.markdown("**Visual Diagram Results**")
                    st.caption(f"Source: {source}")
                    if source_filter:
                        st.caption(f"Filter: {source_filter}")

                    if images:
                        col1, col2 = st.columns(2)
                        for idx, item in enumerate(images):
                            with (col1 if idx % 2 == 0 else col2):
                                title = item.get("title", "Image")
                                image_url = item.get("image_url") or item.get("thumbnail_url")
                                link = item.get("link", "")
                                source_name = item.get("source", "")

                                if image_url:
                                    st.image(image_url, use_container_width=True)
                                st.markdown(f"**{title}**")
                                if source_name:
                                    st.caption(source_name)
                                if link:
                                    st.markdown(f"🔗 [Open source]({link})")
                    else:
                        st.info("No additional Google Images diagrams found for this query.")

                with st.expander("AI narrative summary", expanded=False):
                    st.markdown(msg["content"])

            # IMAGE QUERIES - Show Google image search results
            elif query_type == "image" and tools and "search_images" in tools:
                image_data = tools["search_images"]
                st.markdown("### 🖼️ Image Results")

                images = []
                source = "Google Images"
                source_filter = ""
                if isinstance(image_data, dict):
                    source = image_data.get("source", source)
                    source_filter = image_data.get("source_filter", "")
                    images = image_data.get("images", []) if isinstance(image_data.get("images", []), list) else []

                st.caption(f"Source: {source}")
                if source_filter:
                    st.caption(f"Filter: {source_filter}")

                if images:
                    col1, col2 = st.columns(2)
                    for idx, item in enumerate(images):
                        with (col1 if idx % 2 == 0 else col2):
                            title = item.get("title", "Image")
                            image_url = item.get("image_url") or item.get("thumbnail_url")
                            link = item.get("link", "")
                            source_name = item.get("source", "")

                            if image_url:
                                st.image(image_url, use_container_width=True)
                            st.markdown(f"**{title}**")
                            if source_name:
                                st.caption(source_name)
                            if link:
                                st.markdown(f"🔗 [Open source]({link})")

                with st.expander("AI narrative summary", expanded=False):
                    st.markdown(msg["content"])
            
            # INTERACTION QUERIES - Show interaction network with cards
            elif query_type == "interaction" and tools and "search_string" in tools:
                interaction_data = tools["search_string"]
                st.markdown("### 🔗 Protein-Protein Interactions (STRING)")

                if isinstance(interaction_data, dict):
                    protein_name = interaction_data.get("query_protein", "")
                    network_embed_url = interaction_data.get("network_embed_url", "")
                    network_url = interaction_data.get("network_url", "")
                    network_image_url = interaction_data.get("network_image_url", "")
                    resolved_identifier = interaction_data.get("resolved_identifier", protein_name)

                    if not network_embed_url and resolved_identifier:
                        network_embed_url = f"https://string-db.org/cgi/network?identifiers={resolved_identifier}&species=9606"
                    if not network_image_url and resolved_identifier:
                        network_image_url = f"https://string-db.org/api/image/network?identifiers={resolved_identifier}&species=9606&required_score=400"
                    if not network_url:
                        network_url = network_embed_url

                    if network_image_url:
                        st.markdown("### 🌐 STRING Network View")
                        st.image(network_image_url, use_container_width=True)
                        if network_url:
                            st.markdown(f"🔗 Open full network in browser: [{network_url}]({network_url})")

                        with st.expander("Open interactive STRING view", expanded=False):
                            if network_embed_url:
                                iframe_html = f"""
                                <iframe style="width: 100%; height: 620px; border: 1px solid rgba(6,182,212,0.25); border-radius: 10px; margin: 12px 0;"
                                    src="{network_embed_url}">
                                </iframe>
                                """
                                st.markdown(iframe_html, unsafe_allow_html=True)
                
                structured_rows = []
                if isinstance(interaction_data, dict):
                    raw_rows = interaction_data.get("interactions", [])
                    if isinstance(raw_rows, list):
                        for row in raw_rows:
                            if isinstance(row, dict):
                                structured_rows.append({
                                    "Partner": row.get("partner", ""),
                                    "STRING ID": row.get("string_id", ""),
                                    "Score": row.get("score", 0),
                                    "Neighborhood": row.get("neighborhood", 0),
                                    "Fusion": row.get("fusion", 0),
                                    "Cooccurrence": row.get("cooccurrence", 0),
                                    "Coexpression": row.get("coexpression", 0),
                                    "Experiments": row.get("experiments", 0),
                                    "Databases": row.get("databases", 0),
                                    "Textmining": row.get("textmining", 0)
                                })

                if structured_rows:
                    protein_name = interaction_data.get("query_protein", "") if isinstance(interaction_data, dict) else ""
                    if protein_name:
                        st.caption(f"Input protein: {protein_name} | Top functional partners from STRING")
                    st.dataframe(structured_rows, use_container_width=True, hide_index=True)
                else:
                    # Backward compatibility for old/fallback interaction formats.
                    interaction_items = []
                    if isinstance(interaction_data, str):
                        for line in interaction_data.strip().split('\n'):
                            if line.strip():
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    interaction_items.append((parts[0].strip(), parts[1].strip()))
                    elif isinstance(interaction_data, dict):
                        interaction_items = list(interaction_data.items())

                    if interaction_items:
                        with st.expander("🔗 Network Interactions", expanded=True):
                            col1, col2 = st.columns(2)

                            for idx, (key, value) in enumerate(interaction_items):
                                with (col1 if idx % 2 == 0 else col2):
                                    string_link = f'<a href="https://string-db.org/cgi/network?identifiers={key}" target="_blank" style="color: #06b6d4; text-decoration: none; font-weight: bold;">{key}</a>'

                                    st.markdown(f"""
                                    <div style="
                                        background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(34, 211, 238, 0.05) 100%);
                                        border-left: 5px solid #06b6d4;
                                        border-radius: 10px;
                                        padding: 16px;
                                        margin-bottom: 12px;
                                        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2);
                                        border: 1px solid rgba(6, 182, 212, 0.15);
                                        transition: all 0.3s ease;
                                    "
                                    onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(6, 182, 212, 0.12)'"
                                    onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(6, 182, 212, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2)'">
                                        <div style="color: #06b6d4; font-weight: 700; font-size: 11px; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1px;">{string_link}</div>
                                        <div style="color: #cffafe; font-size: 13px; font-weight: 500;">{value}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                # Show bot response below
                st.markdown(f"""
                <div class="nv-msg-bot">
                    <div class="nv-avatar">🧬</div>
                    <div class="nv-bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            # PROTEIN QUERIES - Show protein card with better visualization
            elif query_type == "protein" and tools and "search_uniprot" in tools:
                protein_data = tools["search_uniprot"]
                st.markdown("### ⚗️ Protein Information (UniProt)")
                
                # Parse string or dict format
                protein_items = []
                if isinstance(protein_data, str):
                    # Parse tab-separated protein data
                    for line in protein_data.strip().split('\n'):
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                protein_items.append((parts[0].strip(), parts[1].strip()))
                elif isinstance(protein_data, dict):
                    protein_items = list(protein_data.items())
                
                if protein_items:
                    with st.expander("⚗️ Protein Information", expanded=True):
                        cols = st.columns(2)
                        
                        for idx, (key, value) in enumerate(protein_items):
                            with (cols[0] if idx % 2 == 0 else cols[1]):
                                # Check if value looks like a UniProt ID (e.g., P04637)
                                uniprot_link = ""
                                if len(value) == 6 and value.isupper() and any(c.isdigit() for c in value):
                                    uniprot_link = f'<a href="https://www.uniprot.org/uniprotkb/{value}" target="_blank" style="color: #14b8a6; text-decoration: none; font-weight: bold;">{value}</a>'
                                    display_value = uniprot_link
                                else:
                                    display_value = value
                                
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(20, 184, 166, 0.1) 0%, rgba(13, 148, 136, 0.05) 100%);
                                    border-left: 5px solid #14b8a6;
                                    border-radius: 10px;
                                    padding: 16px;
                                    margin-bottom: 12px;
                                    box-shadow: 0 4px 12px rgba(20, 184, 166, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2);
                                    border: 1px solid rgba(20, 184, 166, 0.15);
                                    transition: all 0.3s ease;
                                    position: relative;
                                "
                                onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(20, 184, 166, 0.12)'"
                                onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(20, 184, 166, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2)'">
                                    <div style="color: #14b8a6; font-weight: 700; font-size: 11px; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1px;">{key}</div>
                                    <div style="color: #c8e6e1; font-size: 13px; font-weight: 500;">{display_value}</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Show bot response below
                st.markdown(f"""
                <div class="nv-msg-bot">
                    <div class="nv-avatar">🧬</div>
                    <div class="nv-bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # COMPOUND QUERIES - Show compound info with better visualization
            elif query_type == "compound" and tools and "search_pubchem" in tools:
                compound_data = tools["search_pubchem"]
                st.markdown("### 💊 Compound Information (PubChem)")

                if isinstance(compound_data, dict):
                    compound_name = compound_data.get("compound", "")
                    image_url = compound_data.get("image_url_2d", "")
                    pubchem_url = compound_data.get("pubchem_url", "")
                    pubchem_3d_url = compound_data.get("pubchem_3d_url", "")
                    sdf_url_2d = compound_data.get("sdf_url_2d", "")
                    sdf_url_3d = compound_data.get("sdf_url_3d", "")
                    requested_view = compound_data.get("requested_view", "2d")

                    if image_url:
                        st.markdown("### 🧪 2D Chemical Structure")
                        st.image(image_url, width=420)
                        if pubchem_url:
                            st.markdown(f"🔗 Open in PubChem: [{pubchem_url}]({pubchem_url})")

                    if requested_view == "3d":
                        st.markdown("### 🧭 3D Conformer")
                        if pubchem_3d_url:
                            st.markdown(f"🔗 Open 3D conformer in PubChem: [{pubchem_3d_url}]({pubchem_3d_url})")
                        if sdf_url_3d:
                            st.markdown(f"⬇️ Download 3D SDF: [{sdf_url_3d}]({sdf_url_3d})")
                        if sdf_url_2d:
                            st.markdown(f"⬇️ Download 2D SDF: [{sdf_url_2d}]({sdf_url_2d})")

                    details_rows = []
                    ordered_keys = [
                        ("Compound", compound_name),
                        ("CID", compound_data.get("cid", "")),
                        ("Molecular Formula", compound_data.get("molecular_formula", "")),
                        ("Molecular Weight", compound_data.get("molecular_weight", "")),
                        ("Canonical SMILES", compound_data.get("canonical_smiles", "")),
                        ("IUPAC Name", compound_data.get("iupac_name", "")),
                    ]
                    for label, value in ordered_keys:
                        if value not in (None, ""):
                            details_rows.append({"Property": label, "Value": value})

                    if details_rows:
                        st.dataframe(details_rows, use_container_width=True, hide_index=True)
                
                # Parse string or dict format
                compound_items = []
                if isinstance(compound_data, str):
                    # Parse tab-separated compound data
                    for line in compound_data.strip().split('\n'):
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                compound_items.append((parts[0].strip(), parts[1].strip()))
                elif isinstance(compound_data, dict) and not compound_data.get("image_url_2d"):
                    compound_items = list(compound_data.items())
                
                if compound_items:
                    with st.expander("💊 Chemical Compounds", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        for idx, (key, value) in enumerate(compound_items):
                            with (col1 if idx % 2 == 0 else col2):
                                # Create PubChem link if key is a CID (compound ID)
                                pubchem_link = key
                                if key.startswith("CID:"):
                                    cid = key.replace("CID:", "").strip()
                                    if cid.isdigit():
                                        pubchem_link = f'<a href="https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" target="_blank" style="color: #d946ef; text-decoration: none; font-weight: bold;">{key}</a>'
                                
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, rgba(217, 70, 239, 0.1) 0%, rgba(189, 24, 239, 0.05) 100%);
                                    border-left: 5px solid #d946ef;
                                    border-radius: 10px;
                                    padding: 16px;
                                    margin-bottom: 12px;
                                    box-shadow: 0 4px 12px rgba(217, 70, 239, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2);
                                    border: 1px solid rgba(217, 70, 239, 0.15);
                                    transition: all 0.3s ease;
                                "
                                onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(217, 70, 239, 0.12)'"
                                onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(217, 70, 239, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2)'">
                                    <div style="color: #d946ef; font-weight: 700; font-size: 11px; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 1px;">{pubchem_link}</div>
                                    <div style="color: #f3e8ff; font-size: 13px; font-weight: 500;">{value}</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Show bot response below
                st.markdown(f"""
                <div class="nv-msg-bot">
                    <div class="nv-avatar">🧬</div>
                    <div class="nv-bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

            # PAPERS QUERIES - Show clickable research papers list
            elif query_type == "papers" and tools and "search_papers" in tools:
                papers_data = tools["search_papers"]
                st.markdown("### 📚 Research Papers")

                papers = []
                topic = ""
                if isinstance(papers_data, dict):
                    topic = papers_data.get("topic", "")
                    papers = papers_data.get("papers", []) if isinstance(papers_data.get("papers", []), list) else []

                if topic:
                    st.caption(f"Topic: {topic} | Source: PubMed")

                if papers:
                    for paper in papers:
                        year_text = str(paper.get("year", ""))
                        year_match = re.search(r"(19|20)\d{2}", year_text)
                        paper["parsed_year"] = int(year_match.group(0)) if year_match else None

                    available_years = [p["parsed_year"] for p in papers if p.get("parsed_year") is not None]
                    sort_option = st.selectbox(
                        "Sort papers",
                        ["Relevance", "Newest first", "Oldest first"],
                        key=f"papers_sort_{msg_index}"
                    )

                    filtered_papers = list(papers)
                    if available_years:
                        min_year = min(available_years)
                        max_year = max(available_years)
                        year_range = st.slider(
                            "Filter by year",
                            min_value=min_year,
                            max_value=max_year,
                            value=(min_year, max_year),
                            key=f"papers_year_{msg_index}"
                        )
                        filtered_papers = [
                            p for p in filtered_papers
                            if p.get("parsed_year") is None or year_range[0] <= p["parsed_year"] <= year_range[1]
                        ]

                    if sort_option == "Newest first":
                        filtered_papers.sort(key=lambda p: p.get("parsed_year") or 0, reverse=True)
                    elif sort_option == "Oldest first":
                        filtered_papers.sort(key=lambda p: p.get("parsed_year") or 9999)

                    for idx, paper in enumerate(filtered_papers):
                        title = paper.get("title", "Untitled")
                        authors = paper.get("authors", "")
                        journal = paper.get("journal", "")
                        year = paper.get("year", "")
                        doi = paper.get("doi", "")
                        brief = paper.get("brief", "")
                        pubmed_url = paper.get("pubmed_url", "")

                        title_block = title
                        if pubmed_url:
                            title_block = f'<a href="{pubmed_url}" target="_blank" style="color: #38bdf8; text-decoration: none; font-weight: 700;">{title}</a>'

                        doi_line = f"<div style='color:#94a3b8; font-size:12px; margin-top:4px;'>DOI: {doi}</div>" if doi else ""

                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(14, 116, 144, 0.06) 100%);
                            border-left: 5px solid #38bdf8;
                            border-radius: 10px;
                            padding: 16px;
                            margin-bottom: 14px;
                            box-shadow: 0 4px 12px rgba(56, 189, 248, 0.08), 0 2px 4px rgba(0, 0, 0, 0.2);
                            border: 1px solid rgba(56, 189, 248, 0.2);
                        ">
                            <div style="font-size: 17px; margin-bottom: 8px;">{title_block}</div>
                            <div style="color:#cbd5e1; font-size:13px; margin-bottom:4px;"><strong>Authors:</strong> {authors}</div>
                            <div style="color:#cbd5e1; font-size:13px; margin-bottom:4px;"><strong>Journal/Year:</strong> {journal} {year}</div>
                            {doi_line}
                            <div style="color:#e2e8f0; font-size:13px; margin-top:8px; line-height:1.55;"><strong>Brief:</strong> {brief}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        citation_text = f"{authors}. {title}. {journal}. {year}."
                        if doi:
                            citation_text += f" doi:{doi}"

                        with st.expander(f"Citation tools: {title[:70]}", expanded=False):
                            st.text_area(
                                "Copy-ready citation",
                                value=citation_text,
                                height=90,
                                key=f"papers_citation_text_{msg_index}_{idx}"
                            )
                            st.download_button(
                                label="Download citation (.txt)",
                                data=citation_text,
                                file_name=f"citation_{paper.get('pmid', idx)}.txt",
                                mime="text/plain",
                                key=f"papers_citation_download_{msg_index}_{idx}"
                            )

                # Show bot response below
                st.markdown(f"""
                <div class="nv-msg-bot">
                    <div class="nv-avatar">🧬</div>
                    <div class="nv-bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

            # VARIANT QUERIES - Show ClinVar mutations/variants
            elif query_type == "variants" and tools and "search_clinvar" in tools:
                clinvar_data = tools["search_clinvar"]
                st.markdown("### 🧬 ClinVar Variants & Mutations")

                gene_name = ""
                variants = []
                if isinstance(clinvar_data, dict):
                    gene_name = clinvar_data.get("gene", "")
                    variants = clinvar_data.get("variants", []) if isinstance(clinvar_data.get("variants", []), list) else []

                if gene_name:
                    st.caption(f"Gene: {gene_name} | Source: ClinVar")

                # Build a clean interpretation directly from ClinVar data (not raw LLM markdown output).
                if variants:
                    significance_counts = {}
                    review_counts = {}
                    for v in variants:
                        sig = str(v.get("clinical_significance", "Not specified")).strip() or "Not specified"
                        significance_counts[sig] = significance_counts.get(sig, 0) + 1
                        rev = str(v.get("review_status", "Not specified")).strip() or "Not specified"
                        review_counts[rev] = review_counts.get(rev, 0) + 1

                    sorted_sigs = sorted(significance_counts.items(), key=lambda kv: kv[1], reverse=True)
                    top_sig = sorted_sigs[0][0] if sorted_sigs else "Not specified"
                    top_sig_count = sorted_sigs[0][1] if sorted_sigs else 0

                    likely_pathogenic_count = sum(
                        1 for v in variants
                        if "pathogenic" in str(v.get("clinical_significance", "")).lower()
                    )
                    benign_count = sum(
                        1 for v in variants
                        if "benign" in str(v.get("clinical_significance", "")).lower()
                    )

                    summary_parts = [
                        f"{len(variants)} ClinVar variants are listed for {gene_name}.",
                        f"Most frequent classification: {top_sig} ({top_sig_count})."
                    ]
                    if likely_pathogenic_count:
                        summary_parts.append(f"Pathogenic/Likely pathogenic calls: {likely_pathogenic_count}.")
                    if benign_count:
                        summary_parts.append(f"Benign/Likely benign calls: {benign_count}.")

                    summary = " ".join(summary_parts)

                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(148, 163, 184, 0.10) 0%, rgba(71, 85, 105, 0.06) 100%);
                        border-left: 4px solid #94a3b8;
                        border-radius: 10px;
                        padding: 14px;
                        margin-bottom: 12px;
                        border: 1px solid rgba(148, 163, 184, 0.20);
                        color: #e2e8f0;
                        font-size: 13px;
                        line-height: 1.55;
                    ">
                        <strong style="color:#cbd5e1;">Clinical interpretation:</strong><br>{summary}
                    </div>
                    """, unsafe_allow_html=True)

                    # Detailed distribution view for clinical classification quality and evidence.
                    col_a, col_b = st.columns(2)
                    with col_a:
                        sig_rows = []
                        total = len(variants)
                        for sig, count in sorted(significance_counts.items(), key=lambda kv: kv[1], reverse=True):
                            pct = round((count / total) * 100, 1) if total else 0
                            sig_rows.append({"Clinical Significance": sig, "Count": count, "Percent": f"{pct}%"})
                        if sig_rows:
                            st.markdown("**Classification Distribution**")
                            st.dataframe(sig_rows, use_container_width=True, hide_index=True)

                    with col_b:
                        review_rows = []
                        total = len(variants)
                        for rev, count in sorted(review_counts.items(), key=lambda kv: kv[1], reverse=True):
                            pct = round((count / total) * 100, 1) if total else 0
                            review_rows.append({"Review Status": rev, "Count": count, "Percent": f"{pct}%"})
                        if review_rows:
                            st.markdown("**Review Evidence Distribution**")
                            st.dataframe(review_rows, use_container_width=True, hide_index=True)

                    # Surface potentially high-impact variants explicitly.
                    high_impact = [
                        v for v in variants
                        if "pathogenic" in str(v.get("clinical_significance", "")).lower()
                    ]
                    if high_impact:
                        st.markdown("**Potentially High-Impact Variants (Pathogenic/Likely Pathogenic)**")
                        for v in high_impact[:6]:
                            v_title = v.get("title", "Variant")
                            v_sig = v.get("clinical_significance", "Not specified")
                            v_url = v.get("clinvar_url", "")
                            if v_url:
                                st.markdown(f"- [{v_title}]({v_url}) | {v_sig}")
                            else:
                                st.markdown(f"- {v_title} | {v_sig}")

                if variants:
                    sig_options = ["All"] + sorted({str(v.get("clinical_significance", "Not specified")) for v in variants})
                    selected_sig = st.selectbox(
                        "Filter table by clinical significance",
                        sig_options,
                        key=f"clinvar_sig_filter_{msg_index}"
                    )

                    filtered_variants = variants
                    if selected_sig != "All":
                        filtered_variants = [
                            v for v in variants
                            if str(v.get("clinical_significance", "Not specified")) == selected_sig
                        ]

                    table_rows = []
                    for variant in filtered_variants:
                        table_rows.append({
                            "Variant": variant.get("title", "Variant"),
                            "Clinical Significance": variant.get("clinical_significance", "Not specified"),
                            "Review Status": variant.get("review_status", "Not specified"),
                            "Variant ID": str(variant.get("variation_id", "")),
                            "Accession": variant.get("accession", ""),
                            "Type": variant.get("variant_type", ""),
                            "Last Evaluated": variant.get("last_evaluated", ""),
                            "ClinVar Link": variant.get("clinvar_url", "")
                        })

                    st.dataframe(
                        table_rows,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "ClinVar Link": st.column_config.LinkColumn("ClinVar Link")
                        }
                    )

                # Keep full narrative available for users who want detailed explanation.
                with st.expander("AI narrative summary", expanded=False):
                    st.markdown(msg["content"])
            
            # GENERAL / MULTIDATA - Show all available tools
            elif tools:
                st.markdown(f"""
                <div class="nv-msg-bot">
                    <div class="nv-avatar">🧬</div>
                    <div class="nv-bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**📊 Data Sources**")
                for tool_name, tool_data in tools.items():
                    with st.expander(f"📂 {tool_name.upper()}", expanded=False):
                        if isinstance(tool_data, dict):
                            st.json(tool_data)
                        else:
                            st.text(tool_data)
            
            # DEFAULT - Just show bot message
            else:
                st.markdown(f"""
                <div class="nv-msg-bot">
                    <div class="nv-avatar">🧬</div>
                    <div class="nv-bubble-bot">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────
# Keep streaming status and live tokens visible above the fixed input dock.
status_slot = st.empty()
live_response_slot = st.empty()

# Reserve extra space so latest assistant message never sits under the fixed input dock.
st.markdown('<div style="height: 130px;"></div>', unsafe_allow_html=True)
user_input = st.chat_input("Ask about proteins, genes, pathways...")

# ── PROCESS MESSAGE ───────────────────────────────────────────────────────────
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Create a placeholder for status updates
    status_placeholder = status_slot
    streaming_response_placeholder = live_response_slot
    
    # Get bot response
    try:
        response = requests.post(
            API_URL,
            json={
                "messages": st.session_state.messages,
                "query": user_input
            },
            stream=True,
            timeout=60
        )
        
        bot_response = ""
        query_type = "general"
        tools_data = {}
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    data = json.loads(decoded[6:])
                    
                    # Display stage and progress updates
                    if "stage" in data:
                        stage = data.get("stage", "")
                        message = data.get("message", "")
                        
                        if stage == "thinking":
                            status_placeholder.info(f"🤔 {message}")
                        elif stage == "tools":
                            progress = data.get("progress", 0)
                            total = data.get("total", 0)
                            if progress and total:
                                status_placeholder.info(f"{message} ({progress}/{total})")
                            else:
                                status_placeholder.info(f"{message}")
                        elif stage == "synthesizing":
                            status_placeholder.info(f"💭 {message}")
                        elif stage == "response":
                            status_placeholder.info(f"📝 {message}")
                    
                    # Parse streaming tokens
                    if "token" in data:
                        bot_response += data["token"]
                        safe_preview = html.escape(bot_response).replace("\n", "<br>")
                        streaming_response_placeholder.markdown(f"""
                        <div class="nv-msg-bot">
                            <div class="nv-avatar">🧬</div>
                            <div class="nv-bubble-bot">{safe_preview}<span style="opacity:0.65">▋</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Extract tools data with query type
                    if "tools" in data:
                        tools_data = data.get("tools", {})
                        query_type = data.get("query_type", "general")
                    
                    # Check for final message
                    if "done" in data:
                        query_type = data.get("query_type", query_type)
                        status_placeholder.empty()  # Clear status
                        streaming_response_placeholder.empty()
                        break
        
        if bot_response:
            # Store message with tools data
            msg_obj = {
                "role": "assistant",
                "content": bot_response,
                "query_type": query_type,
                "tools": tools_data
            }
            st.session_state.messages.append(msg_obj)
        else:
            st.session_state.messages.append({"role": "assistant", "content": "No response received."})
            
    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        status_placeholder.empty()
        streaming_response_placeholder.empty()
    
    st.rerun()