"""
ATJ Web App v1 — Reference + Explanations + Basic 3D (Streamlit)
================================================================

What you get in one file:
- **Reference Q&A**: ask questions; answers grounded in the bundled ATJ PDF (TF‑IDF snippets)
- **Topic Explorer**: curated explanations for major topics/subtopics (editable dictionary)
- **3D Demos**: simple Plotly-based 3D visuals (superelevation crossfall, horizontal curve path)

Keep it vendor-neutral: this is an educational companion; for authoritative wording always refer to the PDF.

Repo layout (minimal):
- app.py (this file)
- BPIS_ATJ_8-86_19062020.pdf  (bundled guideline)
- requirements.txt  (streamlit, pypdf, scikit-learn, plotly, openai optional)

Deploy: push to GitHub → Streamlit Community Cloud → pick app.py.
"""

from __future__ import annotations
import io
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# ============== Optional OpenAI (env vars; safe if not set) ==============
USE_OPENAI = False
client = None
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ============================ Config =====================================
PDF_PATH = "BPIS_ATJ_8-86_19062020.pdf"   # ensure file exists at repo root
SIM_THRESHOLD = 0.14                      # strictness cutoff for refusing answers
MAX_SNIPPETS = 6

st.set_page_config(page_title="ATJ 8/86 — Reference & 3D", page_icon="🛣️", layout="wide")

# ============================ Helpers ====================================
@dataclass
class Chunk:
    page: int
    text: str

@st.cache_data(show_spinner=False)
def load_pdf(path: str) -> Tuple[List[str], List[Chunk]]:
    if not os.path.exists(path):
        return [], []
    with open(path, "rb") as f:
        raw = f.read()
    reader = PdfReader(io.BytesIO(raw))
    pages: List[str] = []
    chunks: List[Chunk] = []
    for i, p in enumerate(reader.pages):
        t = p.extract_text() or ""
        pages.append(t)
        for line in (t.split("\n") if t else []):
            s = line.strip()
            if len(s) > 30:
                chunks.append(Chunk(page=i+1, text=s))
    return pages, chunks

@st.cache_data(show_spinner=False)
def build_index(chunks: List[Chunk]):
    corpus = [c.text for c in chunks]
    if not corpus:
        return None, None
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

def highlight(text: str, query: str) -> str:
    terms = [re.escape(t) for t in (query or "").split() if t.strip()]
    if not terms:
        return text
    pattern = re.compile(r"(" + r"|".join(terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", text)

# ===================== Topic Explorer content (editable) ==================
TOPICS: Dict[str, Dict[str, str]] = {
    "Fundamentals": {
        "Design speed": "Design speed is the selected speed used to determine the geometric features of the road. It guides choices like curve radius, sight distance, and superelevation.",
        "Lane & shoulder": "Typical lane width provides operating space while shoulders support stopped vehicles, lateral clearance, and structural support.",
    },
    "Sight distance": {
        "Stopping sight distance (SSD)": "SSD is the minimum distance a driver needs to perceive, react, and brake to a stop before reaching an object in the path.",
        "Overtaking / passing": "Passing sight distance is the length required for a safe overtaking manoeuvre on two-lane two-way roads under specified assumptions.",
        "Decision sight distance": "Additional distance to allow complex decisions and manoeuvres at locations with high information load (e.g., exits).",
    },
    "Horizontal alignment": {
        "Minimum radius": "For a given speed and superelevation, curves must satisfy a minimum radius to limit lateral acceleration and side friction.",
        "Transition (spiral)": "Transition curves provide gradual change of curvature and superelevation, improving comfort and safety.",
    },
    "Vertical alignment": {
        "Crest curve": "Crest vertical curves connect upgrades to downgrades. Length is often controlled by sight distance over the curve crest.",
        "Sag curve": "Sag curves connect downgrades to upgrades; comfort/visibility at night (headlight control) often governs their length.",
    },
    "Crossfall & superelevation": {
        "Normal crossfall": "Normal crown drains water away from the centreline on tangents.",
        "Superelevation": "On curves, the pavement is rotated to an inward slope (e) to balance part of the lateral acceleration.",
    },
}

# =============================== 3D demos ================================
def demo_superelevation(e_percent: float = 6.0, lane_width: float = 3.5, lanes: int = 2, length: float = 30.0):
    """Simple 3D tilted plane to illustrate superelevation e (percent)."""
    import numpy as np
    total_width = lane_width * lanes
    e = e_percent / 100.0
    x = np.linspace(0, length, 20)
    y = np.linspace(-total_width/2, total_width/2, 10)
    X, Y = np.meshgrid(x, y)
    Z = -e * (Y)  # tilt about centreline
    surf = go.Surface(x=X, y=Y, z=Z, opacity=0.9)
    fig = go.Figure(data=[surf])
    fig.update_layout(scene=dict(
        xaxis_title='Chainage (m)', yaxis_title='Offset (m)', zaxis_title='Elevation (m)'
    ), margin=dict(l=0, r=0, t=0, b=0))
    return fig

def demo_horizontal_curve(radius: float = 250.0, arc_length: float = 120.0):
    """3D polyline showing a plan-view circular arc extruded flat at z=0."""
    import numpy as np
    theta = arc_length / radius  # radians if arc_length in m on circular arc
    t = np.linspace(0, theta, 200)
    x = radius * np.sin(t)
    y = radius * (1 - np.cos(t))
    z = 0 * t
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(width=8)))
    fig.update_layout(scene=dict(
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)'
    ), margin=dict(l=0, r=0, t=0, b=0))
    return fig

# =============================== App UI =================================
PAGES = ["Reference Q&A", "Topic Explorer", "3D Demos", "About"]
page = st.sidebar.radio("Navigate", PAGES, index=0)

# Load PDF + index once
pages_text, chunks = load_pdf(PDF_PATH)
vectorizer, X = build_index(chunks)

if page == "Reference Q&A":
    st.title("📘 ATJ 8/86 — Reference Q&A")
    if not pages_text:
        st.error(f"PDF not found at {PDF_PATH}. Place the file in repo root.")
        st.stop()

    strict = st.sidebar.checkbox("Strict mode (refuse if weak evidence)", value=True)
    k = st.sidebar.slider("Snippets to use", 1, MAX_SNIPPETS, 5)

    q = st.text_input("Ask a question", placeholder="e.g., Explain stopping sight distance").strip()
    if st.button("Search", use_container_width=True) and q:
        if vectorizer is None:
            st.error("Index not built.")
        else:
            q_vec = vectorizer.transform([q])
            sims = cosine_similarity(q_vec, X).ravel()
            order = sims.argsort()[::-1]
            top_idx = order[:k]
            results: List[Tuple[float, Chunk]] = [(float(sims[i]), chunks[i]) for i in top_idx]

            if not results:
                st.warning("No matches found.")
            else:
                best_score, _ = results[0]
                if strict and best_score < SIM_THRESHOLD:
                    st.warning("Not in the guideline (ATJ 8/86).")
                else:
                    st.subheader("Top matches (with page numbers)")
                    for idx, (score, ch) in enumerate(results):
                        with st.expander(f"p.{ch.page} · score={score:.3f} · #{idx}"):
                            st.markdown(highlight(ch.text, q), unsafe_allow_html=True)
                            if st.button(f"View page {ch.page}", key=f"view_{ch.page}_{idx}"):
                                st.session_state["selected_page"] = ch.page

                    # Full page view
                    st.divider()
                    sel = st.session_state.get("selected_page", results[0][1].page)
                    st.markdown(f"### Full page reference — p.{sel}")
                    raw = pages_text[sel-1]
                    st.markdown(
                        "<div style='white-space:pre-wrap; font-family: ui-monospace, monospace;'>" +
                        highlight(raw, q) +
                        "</div>", unsafe_allow_html=True,
                    )

                    if USE_OPENAI:
                        ctx = "\n\n".join([f"(p.{c.page}) {c.text}" for _, c in results])
                        try:
                            resp = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": (
                                        "Answer ONLY using the provided ATJ 8/86 context. "
                                        "Cite pages like (p.X). If not present, reply 'Not in the guideline (ATJ 8/86)'."
                                    )},
                                    {"role": "user", "content": f"Question: {q}\n\nContext:\n{ctx}"}
                                ], temperature=0.05,
                            )
                            st.markdown("### Composed answer")
                            st.write(resp.choices[0].message.content)
                        except Exception:
                            st.info("OpenAI composition skipped.")

elif page == "Topic Explorer":
    st.title("📚 Topic Explorer (Editable)")
    st.caption("High-level explanations. For exact wording/limits, refer to the PDF.")

    # Topic & subtopic selectors
    topic = st.selectbox("Topic", list(TOPICS.keys()))
    sub = st.selectbox("Subtopic", list(TOPICS[topic].keys()))

    st.subheader(f"{topic} — {sub}")
    st.write(TOPICS[topic][sub])

    # Link to reference: simple keyword search on demand
    if pages_text and st.button("Find references in guideline"):
        query = sub
        q_vec = vectorizer.transform([query]) if vectorizer is not None else None
        if q_vec is None:
            st.error("Index not built.")
        else:
            sims = cosine_similarity(q_vec, X).ravel()
            order = sims.argsort()[::-1][:3]
            st.markdown("### Closest references")
            for i, idx in enumerate(order):
                ch = chunks[idx]
                sc = float(sims[idx])
                with st.expander(f"p.{ch.page} · score={sc:.3f} · #{i}"):
                    st.write(ch.text)

elif page == "3D Demos":
    st.title("🧭 Basic 3D Graphical Demos")

    tab1, tab2 = st.tabs(["Superelevation", "Horizontal curve path"])

    with tab1:
        e = st.slider("Superelevation e (%)", 0.0, 12.0, 6.0, 0.5)
        lanes = st.slider("Number of lanes (both directions shown)", 1, 4, 2)
        lane_w = st.number_input("Lane width (m)", 2.75, 4.0, 3.5, 0.05)
        L = st.number_input("Visualized length (m)", 10.0, 200.0, 40.0, 1.0)
        fig = demo_superelevation(e_percent=e, lane_width=lane_w, lanes=lanes, length=L)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Illustration only. Not to scale; for teaching the e% concept.")

    with tab2:
        R = st.number_input("Curve radius R (m)", 50.0, 1000.0, 250.0, 5.0)
        s = st.number_input("Arc length (m)", 20.0, 500.0, 120.0, 5.0)
        fig2 = demo_horizontal_curve(radius=R, arc_length=s)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Plan-view arc shown in 3D axes for clarity.")

else:
    st.title("About this app")
    st.markdown(
        "This educational companion helps explore ATJ 8/86 with quick references, curated explanations, and basic 3D demos. "
        "For contractual work, always verify against the official guideline PDF included with the app."
    )
    st.markdown("**Authoring tips**: Edit the `TOPICS` dictionary in `app.py` to refine text; expand 3D demos as needed.")
