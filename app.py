"""
ATJ 8/86 Reference Web App (Streamlit)
======================================

Purpose
-------
A no-fuss web app that answers questions **only** from the bundled JKR guideline PDF
(ATJ 8/86, Pindaan 2015). It does *not* allow arbitrary uploads by default to keep scope strict.

Highlights
- Loads a bundled PDF: sample_docs/BPIS_ATJ_8-86_19062020.pdf
- Retrieves relevant passages using TF‑IDF (fast, simple, no server)
- Shows page numbers for transparency and cites top matches
- Refuses to answer if evidence is weak: "Not in the guideline (ATJ 8/86)"
- Optional OpenAI composition (strictly grounded in retrieved snippets) via Streamlit **Secrets**
- Bilingual UI toggle: English / Bahasa Melayu (basic labels)

Deployment
- Add this file as app.py in a public GitHub repo
- Add requirements.txt (see below)
- Deploy on Streamlit Community Cloud → choose repo + app.py

requirements.txt
----------------
streamlit
pypdf
scikit-learn
openai

"""

import io
import os
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------
# Optional OpenAI (secrets)
# --------------------
USE_OPENAI = False
client = None
try:
    from openai import OpenAI
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False

# --------------------
# Config
# --------------------
PDF_PATH = "sample_docs/BPIS_ATJ_8-86_19062020.pdf"  # bundled file path in repo
MAX_SNIPPETS = 5
SIM_THRESHOLD = 0.14  # reject answers if top score below this
MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="ATJ 8/86 Reference", page_icon="📘", layout="centered")

# --------------------
# Language (EN/BM) — basic labels
# --------------------
LANG = st.sidebar.selectbox("Language / Bahasa", ["English", "Bahasa Melayu"], index=0)
L = {
    "English": {
        "title": "📘 ATJ 8/86 — Reference Q&A",
        "subtitle": "Answers are grounded strictly in the bundled JKR guideline.",
        "ask": "Ask a question",
        "placeholder": "e.g., Explain stopping sight distance",
        "search": "Search",
        "no_pdf": "Bundled PDF not found. Please include it in sample_docs/",
        "no_text": "Could not extract text from the PDF.",
        "matches": "Top matches (with page numbers)",
        "composed": "Composed answer (from matched snippets)",
        "not_in_guideline": "Not in the guideline (ATJ 8/86).",
        "show_snippets": "Show snippets",
        "k_label": "How many snippets to use?",
        "strict_mode": "Strict mode (refuse if weak evidence)",
    },
    "Bahasa Melayu": {
        "title": "📘 ATJ 8/86 — Rujukan Soal Jawab",
        "subtitle": "Jawapan adalah berdasarkan garis panduan JKR yang dibekalkan sahaja.",
        "ask": "Tanya soalan",
        "placeholder": "cth., Terangkan jarak henti (SSD)",
        "search": "Cari",
        "no_pdf": "PDF tidak dijumpai. Sila letak fail dalam folder sample_docs/",
        "no_text": "Teks tidak dapat diekstrak daripada PDF.",
        "matches": "Padanan teratas (berserta nombor muka surat)",
        "composed": "Jawapan ringkas (berdasarkan petikan yang sepadan)",
        "not_in_guideline": "Tiada dalam garis panduan (ATJ 8/86).",
        "show_snippets": "Tunjuk petikan",
        "k_label": "Bilangan petikan digunakan",
        "strict_mode": "Mod ketat (tolak jika bukti lemah)",
    },
}[LANG]

st.title(L["title"])
st.caption(L["subtitle"])

# --------------------
# Load bundled PDF
# --------------------
if not os.path.exists(PDF_PATH):
    st.error(L["no_pdf"])
    st.stop()

with open(PDF_PATH, "rb") as f:
    raw = f.read()
reader = PdfReader(io.BytesIO(raw))

@dataclass
class Chunk:
    page: int
    text: str

chunks: List[Chunk] = []
for i, page in enumerate(reader.pages):
    txt = page.extract_text() or ""
    # split by lines to keep references compact
    for line in (txt.split("\n") if txt else []):
        line = line.strip()
        if len(line) > 30:  # ignore very short noise
            chunks.append(Chunk(page=i+1, text=line))

if not chunks:
    st.error(L["no_text"])
    st.stop()

# --------------------
# Build TF-IDF index
# --------------------
corpus = [c.text for c in chunks]
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)

# --------------------
# UI controls
# --------------------
strict_mode = st.checkbox(L["strict_mode"], value=True)
k = st.slider(L["k_label"], 1, 10, min(MAX_SNIPPETS, 5))
q = st.text_input(L["ask"], placeholder=L["placeholder"]).strip()

if st.button(L["search"], disabled=not bool(q)):
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, X).ravel()
    order = sims.argsort()[::-1]
    top_idx = order[:k]

    results: List[Tuple[float, Chunk]] = [(float(sims[i]), chunks[i]) for i in top_idx]

    # Guard: weak evidence → refuse
    best_score = results[0][0] if results else 0.0
    if strict_mode and (best_score < SIM_THRESHOLD or not results):
        st.warning(L["not_in_guideline"])
        st.stop()

    st.markdown(f"### {L['matches']}")
    for score, ch in results:
        with st.expander(f"p.{ch.page} · score={score:.3f}", expanded=False):
            st.write(ch.text)

    # Optional composed answer grounded in snippets
    if USE_OPENAI:
        context = "\n\n".join([f"(p.{c.page}) {c.text}" for _, c in results])
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": (
                        "Answer ONLY using the provided ATJ 8/86 context. "
                        "Cite page numbers like (p.X). If information is not in context, reply: 'Not in the guideline (ATJ 8/86)'."
                    )},
                    {"role": "user", "content": f"Question: {q}\n\nContext:\n{context}"}
                ],
                temperature=0.05,
            )
            st.markdown(f"### {L['composed']}")
            st.write(resp.choices[0].message.content)
        except Exception:
            pass

st.caption("This tool is scope-limited to the bundled guideline. For other documents, create a new build or enable uploads.")
