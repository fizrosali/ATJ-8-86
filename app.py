"""
ATJ Web App v2 — Practical 3D + Explanations + PDF Extracts (Streamlit)
=========================================================================

What this does
--------------
- **SSD on grade (3D):** Real‑world inputs (speed, grade, reaction time, friction) → live 3D scene + computed distances.
- **Curve & superelevation (3D):** Inputs (speed, radius, f_max, e_max) → required e(%), pass/fail, live 3D scene.
- **PDF extracts:** Auto‑pull relevant, page‑numbered text snippets from your bundled ATJ PDF for each tool, with highlights.
- **Free Q&A:** Quick keyword search over the PDF with page numbers and full‑page viewer.
- **Grounding:** This is an educational companion; always verify against the official PDF.

Repo layout
-----------
- app.py  (this file)
- BPIS_ATJ_8-86_19062020.pdf  (place at repo root or change PDF_PATH)
- requirements.txt  (streamlit, pypdf, scikit-learn, plotly, numpy, openai [optional])

Run locally
-----------
    pip install -r requirements.txt
    streamlit run app.py

Optional OpenAI composition (still grounded): set env vars `OPENAI_API_KEY`, `OPENAI_MODEL`.
"""

from __future__ import annotations
import io
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# -------------------- Optional OpenAI (env vars; safe if not set) --------------------
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

# ---------------------------------- Config ----------------------------------
PDF_PATH = "BPIS_ATJ_8-86_19062020.pdf"   # your bundled guideline (adjust path if needed)
SIM_THRESHOLD = 0.14                       # weak‑evidence cutoff for refusal in Q&A
MAX_SNIPPETS = 6
G = 9.81                                   # gravity (m/s^2)

st.set_page_config(page_title="ATJ 8/86 — Practical 3D & Reference", page_icon="🛣️", layout="wide")

# --------------------------------- Helpers ----------------------------------
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

def search_snippets(vectorizer, X, chunks: List[Chunk], query: str, k: int = 5):
    if vectorizer is None or X is None or not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).ravel()
    order = sims.argsort()[::-1][:k]
    return [(float(sims[i]), chunks[i]) for i in order]

_def_hl_css = "<div style='white-space:pre-wrap; font-family: ui-monospace, monospace;'>"  # noqa

def highlight(text: str, query: str) -> str:
    terms = [re.escape(t) for t in (query or "").split() if t.strip()]
    if not terms:
        return text
    pattern = re.compile(r"(" + r"|".join(terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", text)

# ----------------------------- Engineering calcs -----------------------------

def calc_ssd(v_kmh: float, t_react: float, f: float, grade_percent: float) -> Dict[str, float]:
    """Compute SSD on grade using a common formulation.
    SSD = v*t + v^2 / (2*g*(f ± G)), with G=grade as decimal (+ upgrade, - downgrade).
    """
    v = max(0.0, v_kmh) / 3.6  # m/s
    Gd = grade_percent / 100.0
    # Effective deceleration term
    denom = f + Gd
    if denom <= 1e-6:  # avoid divide-by-zero/negative (extreme steep downgrade with low f)
        braking = float("inf")
    else:
        braking = (v**2) / (2.0 * G * denom)
    reaction = v * t_react
    ssd = reaction + braking
    return {"v_mps": v, "reaction": reaction, "braking": braking, "ssd": ssd}


def ssd_scene(v_kmh: float, grade_percent: float, ssd: float, lane_w: float = 3.5, lanes: int = 2):
    """3D scene: sloped road (grade), car at x=0, object at x=SSD."""
    total_w = lane_w * lanes
    slope = np.tan(np.arctan(grade_percent / 100.0))  # approx G as slope
    xlen = max(30.0, ssd * 1.2)
    x = np.linspace(0, xlen, 40)
    y = np.linspace(-total_w/2, total_w/2, 12)
    X, Y = np.meshgrid(x, y)
    Z = slope * X

    road = go.Surface(x=X, y=Y, z=Z, opacity=0.9)
    # Centerline and markers
    line = go.Scatter3d(x=[0, xlen], y=[0, 0], z=[0, slope*xlen], mode='lines', line=dict(width=6))
    car = go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=6))
    obj = go.Scatter3d(x=[ssd], y=[0], z=[slope*ssd], mode='markers', marker=dict(size=6))

    fig = go.Figure(data=[road, line, car, obj])
    fig.update_layout(scene=dict(
        xaxis_title='Distance along road (m)', yaxis_title='Offset (m)', zaxis_title='Elevation (m)'
    ), margin=dict(l=0, r=0, t=0, b=0))
    return fig


def calc_required_e(v_kmh: float, R: float, f_max: float) -> float:
    """Required superelevation e (decimal) from v^2/(gR) = e + f; e = v^2/(gR) - f.
    Clamped to [0, 0.12] for display purposes.
    """
    v = max(0.0, v_kmh) / 3.6
    e_req = (v*v) / (G * R) - f_max
    return float(np.clip(e_req, 0.0, 0.12))


def superelevated_road(e: float, radius: float = 250.0, arc_len: float = 120.0, lane_w: float = 3.5, lanes: int = 2):
    """3D surface of an arced, superelevated roadway (constant e)."""
    total_w = lane_w * lanes
    theta = arc_len / radius  # radians
    t = np.linspace(0, theta, 160)
    x = radius * np.sin(t)
    y = radius * (1 - np.cos(t))
    # Local road coordinates: along-arc s vs offset y_off; rotate offset by e
    s = np.linspace(0, arc_len, 40)
    y_off = np.linspace(-total_w/2, total_w/2, 16)
    S, Y = np.meshgrid(s, y_off)
    Z = -e * (Y)  # tilt about centerline

    # Map S back to (x,y) along the precomputed plan arc by nearest indices
    idx = (S / arc_len * (len(t)-1)).astype(int)
    XX = x[idx]
    YY = y[idx]

    road = go.Surface(x=XX, y=YY + Y, z=Z, opacity=0.92)
    center = go.Scatter3d(x=x, y=y, z=0*y, mode='lines', line=dict(width=6))

    fig = go.Figure(data=[road, center])
    fig.update_layout(scene=dict(
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Elevation (m)'
    ), margin=dict(l=0, r=0, t=0, b=0))
    return fig

# ----------------------------- Load + Index -----------------------------
pages_text, chunks = load_pdf(PDF_PATH)
vectorizer, X = build_index(chunks)

# --------------------------------- UI -----------------------------------
st.sidebar.title("ATJ 8/86 — Tools")
view = st.sidebar.radio("Choose view", ["SSD on Grade", "Curve & Superelevation", "Free Q&A", "About"], index=0)

# =============================== SSD on Grade ===============================
if view == "SSD on Grade":
    st.title("🛑 Stopping Sight Distance on Grade — 3D & Reference")

    cols = st.columns([1,1,1,1,1])
    with cols[0]: v_kmh = st.number_input("Speed (km/h)", 10.0, 140.0, 80.0, 1.0)
    with cols[1]: grade = st.number_input("Grade (%)  (+up / −down)", -12.0, 12.0, 0.0, 0.5)
    with cols[2]: t_react = st.number_input("Reaction time t (s)", 0.5, 3.5, 2.5, 0.1)
    with cols[3]: f = st.number_input("Friction f", 0.10, 0.60, 0.35, 0.01)
    with cols[4]: lanes = st.slider("Lanes shown", 1, 4, 2)

    out = calc_ssd(v_kmh, t_react, f, grade)
    ssd = out["ssd"]

    m1, m2 = st.columns([2,1])
    with m1:
        fig = ssd_scene(v_kmh, grade, ssd, lanes=lanes)
        st.plotly_chart(fig, use_container_width=True)
    with m2:
        st.subheader("Results")
        st.metric("Reaction distance (m)", f"{out['reaction']:.1f}")
        st.metric("Braking distance (m)", f"{(np.inf if np.isinf(out['braking']) else round(out['braking'],1))}")
        st.metric("SSD (m)", f"{(np.inf if np.isinf(ssd) else round(ssd,1))}")
        if np.isinf(ssd):
            st.warning("Parameters produce unrealistic SSD (denominator ≤ 0). Increase f or reduce downgrade.")

    # PDF extracts for SSD keywords
    st.divider()
    st.subheader("PDF extracts — related to SSD & grades")
    queries = ["stopping sight distance", "SSD", "grade", "braking", "perception reaction time"]
    qtext = "; ".join(queries)
    if pages_text:
        hits = search_snippets(vectorizer, X, chunks, " ".join(queries), k=5)
        for i, (score, ch) in enumerate(hits):
            with st.expander(f"p.{ch.page} · score={score:.3f} · #{i}"):
                st.markdown(highlight(ch.text, qtext), unsafe_allow_html=True)
                if st.button(f"View page {ch.page}", key=f"ssd_view_{ch.page}_{i}"):
                    st.session_state["selected_page"] = ch.page
        if "selected_page" in st.session_state:
            p = st.session_state["selected_page"]
            st.markdown(f"**Full page {p}**")
            st.markdown(_def_hl_css + highlight(pages_text[p-1], qtext) + "</div>", unsafe_allow_html=True)
    else:
        st.info(f"PDF not found at {PDF_PATH}.")

# ============================ Curve & Superelevation ============================
elif view == "Curve & Superelevation":
    st.title("↪️ Horizontal Curve & Superelevation — 3D & Reference")

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1: v_kmh = st.number_input("Speed (km/h)", 20.0, 140.0, 80.0, 1.0, key="curve_v")
    with c2: R = st.number_input("Radius R (m)", 30.0, 2000.0, 250.0, 5.0)
    with c3: f_max = st.number_input("Max side friction f", 0.05, 0.25, 0.15, 0.01)
    with c4: e_max = st.number_input("Max superelevation e_max (%)", 2.0, 12.0, 8.0, 0.5)

    e_req = calc_required_e(v_kmh, R, f_max)  # decimal
    e_max_dec = e_max / 100.0
    ok = e_req <= e_max_dec + 1e-6

    L_vis = st.slider("Visualized arc length (m)", 40.0, 300.0, 120.0, 5.0)
    lanes = st.slider("Lanes shown", 1, 4, 2, key="curve_lanes")
    lane_w = st.number_input("Lane width (m)", 2.75, 4.00, 3.50, 0.05, key="curve_lane_w")

    g1, g2 = st.columns([2,1])
    with g1:
        fig2 = superelevated_road(e=min(e_req, e_max_dec), radius=R, arc_len=L_vis, lane_w=lane_w, lanes=lanes)
        st.plotly_chart(fig2, use_container_width=True)
    with g2:
        st.subheader("Results")
        st.metric("Required e (%)", f"{100*e_req:.2f}")
        st.metric("Allowed e_max (%)", f"{e_max:.2f}")
        st.success("Within e_max") if ok else st.error("Exceeds e_max — increase R or reduce speed.")

    # PDF extracts for curve & e
    st.divider()
    st.subheader("PDF extracts — curves & superelevation")
    queries = ["superelevation", "side friction", "horizontal curve", "minimum radius", "e ="]
    qtext = "; ".join(queries)
    if pages_text:
        hits = search_snippets(vectorizer, X, chunks, " ".join(queries), k=5)
        for i, (score, ch) in enumerate(hits):
            with st.expander(f"p.{ch.page} · score={score:.3f} · #{i}"):
                st.markdown(highlight(ch.text, qtext), unsafe_allow_html=True)
                if st.button(f"View page {ch.page}", key=f"curve_view_{ch.page}_{i}"):
                    st.session_state["selected_page"] = ch.page
        if "selected_page" in st.session_state:
            p = st.session_state["selected_page"]
            st.markdown(f"**Full page {p}**")
            st.markdown(_def_hl_css + highlight(pages_text[p-1], qtext) + "</div>", unsafe_allow_html=True)
    else:
        st.info(f"PDF not found at {PDF_PATH}.")

# ================================== Free Q&A ==================================
elif view == "Free Q&A":
    st.title("🔎 Free Q&A over the PDF")

    strict = st.sidebar.checkbox("Strict mode (refuse if weak evidence)", value=True, key="qa_strict")
    k = st.sidebar.slider("Snippets to use", 1, MAX_SNIPPETS, 5, key="qa_k")

    q = st.text_input("Ask a question", placeholder="e.g., Define transition curve or SSD factors").strip()
    if st.button("Search", use_container_width=True, key="qa_btn") and q:
        if vectorizer is None:
            st.error("Index not built (PDF missing)")
        else:
            hits = search_snippets(vectorizer, X, chunks, q, k=k)
            if not hits:
                st.warning("No matches found.")
            else:
                best = hits[0][0]
                if strict and best < SIM_THRESHOLD:
                    st.warning("Not in the guideline (ATJ 8/86).")
                else:
                    st.subheader("Top matches")
                    for i, (score, ch) in enumerate(hits):
                        with st.expander(f"p.{ch.page} · score={score:.3f} · #{i}"):
                            st.markdown(highlight(ch.text, q), unsafe_allow_html=True)
                            if st.button(f"View page {ch.page}", key=f"qa_view_{ch.page}_{i}"):
                                st.session_state["selected_page"] = ch.page

                    if "selected_page" in st.session_state:
                        p = st.session_state["selected_page"]
                        st.markdown(f"**Full page {p}**")
                        st.markdown(_def_hl_css + highlight(pages_text[p-1], q) + "</div>", unsafe_allow_html=True)

# ==================================== About ====================================
else:
    st.title("About this app")
    st.markdown(
        "This interactive tool demonstrates practical highway design concepts with live 3D visuals and pulls related text from the bundled ATJ 8/86 PDF.\n"
        "Use the extracts for *reference* and consult the official guideline for contractual decisions."
    )
    st.markdown("**Tips**:\n- Tune friction and grade to see SSD sensitivities.\n- Explore different R and speed to see superelevation needs.\n- Add your own queries in Free Q&A for quick lookups.")
