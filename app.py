# ATJ Web App v2 — Practical 3D + Cars + PDF Extracts (Streamlit)
# ---------------------------------------------------------------
# Tabs:
# 1) Stopping Sight Distance (SSD) on grade — car visual, brake line, headlight line
# 2) Horizontal Curve & Superelevation — banked road, car on curve, required e
# 3) Free Q&A — TF-IDF search over ATJ PDF
#
# PDF extracts: shows page + snippet matching current topic and inputs
# OpenAI optional via env vars OPENAI_API_KEY/OPENAI_MODEL (not required)

from __future__ import annotations
import io, os, re, math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= Basic config =================
st.set_page_config(page_title="ATJ 8/86 — Practical Road Lab", page_icon="🚗", layout="wide")
PDF_PATH = "BPIS_ATJ_8-86_19062020.pdf"       # place your PDF at repo root
SIM_THRESHOLD = 0.14                           # refusal cutoff for weak retrieval
RESULT_SNIPPETS = 4

# =============== Optional OpenAI (disabled by default) ===============
USE_OPENAI = False
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
try:
    from openai import OpenAI
    _key = os.getenv("OPENAI_API_KEY")
    if _key:
        oai = OpenAI(api_key=_key)
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False

# ================= PDF loading & retrieval =================
@dataclass
class Chunk:
    page: int
    text: str

@st.cache_data(show_spinner=False)
def load_pdf_text(path: str) -> Tuple[List[str], List[Chunk]]:
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
        for line in t.split("\n"):
            s = (line or "").strip()
            if len(s) > 40:
                chunks.append(Chunk(page=i+1, text=s))
    return pages, chunks

@st.cache_data(show_spinner=False)
def build_tfidf(chunks: List[Chunk]):
    corpus = [c.text for c in chunks]
    if not corpus:
        return None, None
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(corpus)
    return vec, X

def search_snippets(query: str, vec, X, chunks: List[Chunk], topk=RESULT_SNIPPETS) -> List[Tuple[float, Chunk]]:
    if (not query) or (vec is None):
        return []
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    # best per page
    best: Dict[int, Tuple[float, Chunk]] = {}
    for score, ch in zip(sims, chunks):
        cur = best.get(ch.page)
        if (cur is None) or (score > cur[0]):
            best[ch.page] = (float(score), ch)
    out = sorted(best.values(), key=lambda x: x[0], reverse=True)[:topk]
    return out

def highlight(text: str, query: str) -> str:
    terms = [re.escape(t) for t in (query or "").split() if t.strip()]
    if not terms:
        return text
    pattern = re.compile(r"(" + r"|".join(terms) + r")", re.IGNORECASE)
    return pattern.sub(r"<mark>\\1</mark>", text)

PAGES, CHUNKS = load_pdf_text(PDF_PATH)
VEC, X = build_tfidf(CHUNKS)

# ===================== Road math (practical) =====================
G = 9.81  # m/s^2

def kmh_to_ms(v_kmh: float) -> float:
    return v_kmh / 3.6

def ssd_on_grade(v_kmh: float, t_react: float, f: float, grade_percent: float) -> float:
    """
    SSD = v * t + v^2 / (2*g*(f ± G))
    G positive for upgrade reduces stopping distance (resists motion)
    Here: grade_percent positive = upgrade; negative = downgrade
    """
    v = kmh_to_ms(v_kmh)
    Ggrade = (grade_percent / 100.0)
    denom = 2 * G * (f + Ggrade)  # upgrade -> +Ggrade, downgrade -> -|G| (by sign in input)
    # more standard: SSD = v*t + v^2 / (2g (f ± G)), if downgrade use (f - |G|)
    # We'll implement using sign of grade_percent:
    if grade_percent < 0:
        denom = 2 * G * (f - abs(Ggrade))
    braking = v**2 / max(denom, 1e-6)
    return v * t_react + max(braking, 0.0)

def e_required(v_kmh: float, R: float, f: float) -> float:
    """
    From lateral equilibrium: e + f = v^2/(gR)
    Return e_required (decimal). Clip to [0, 0.12] (typical design envelope).
    """
    v = kmh_to_ms(v_kmh)
    e = (v**2) / (G * R) - f
    return float(np.clip(e, 0.0, 0.12))

# ===================== 3D builders (cars & roads) =====================
def car_mesh(center=(0,0,0), L=4.4, W=1.8, H=1.5, yaw_deg=0.0, color="red", name="car"):
    """
    Simple box car body as Mesh3d; center at (x,y,z), yaw about z.
    """
    cx, cy, cz = center
    yaw = math.radians(yaw_deg)
    dx = L/2; dy = W/2; dz = H
    corners = np.array([
        [-dx,-dy,0],[ dx,-dy,0],[ dx, dy,0],[-dx, dy,0],  # bottom
        [-dx,-dy,dz],[ dx,-dy,dz],[ dx, dy,dz],[-dx, dy,dz] # top
    ], dtype=float)
    # rotate yaw
    rot = np.array([[ math.cos(yaw), -math.sin(yaw), 0],
                    [ math.sin(yaw),  math.cos(yaw), 0],
                    [ 0, 0, 1]])
    corners = corners @ rot.T
    corners += np.array([cx, cy, cz])
    x, y, z = corners.T
    # faces (triangles)
    I = [0,0,0, 1,2,3, 4,4,4, 5,6,7, 0,1,5, 4,5,6, 6,7,3, 7,4,0]  # will be corrected with J,K
    # Define rectangular faces explicitly (12 triangles):
    I = [0,1,2, 0,2,3,  # bottom
         4,5,6, 4,6,7,  # top
         0,1,5, 0,5,4,  # side
         1,2,6, 1,6,5,
         2,3,7, 2,7,6,
         3,0,4, 3,4,7]
    J = [1,2,3, 2,3,0,
         5,6,7, 6,7,4,
         1,5,4, 5,4,0,
         2,6,5, 6,5,1,
         3,7,6, 7,6,2,
         0,4,7, 4,7,3]
    K = [2,3,0, 3,0,1,
         6,7,4, 7,4,5,
         5,4,0, 4,0,1,
         6,5,1, 5,1,2,
         7,6,2, 6,2,3,
         4,7,3, 7,3,0]
    return go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=color, opacity=0.95, name=name)

def road_plane(length=200, width=8, grade_percent=0.0, bank_e=0.0, center=(0,0,0), name="road", color="#808080"):
    """
    Road as a slightly-tesselated plane so we can tilt along x (grade) and bank along y (superelevation).
    length along +x; width along y; grade tilts in x-z; bank tilts across y-z.
    """
    L = int(max(10, length//5))
    W = 6
    xs = np.linspace(0, length, L)
    ys = np.linspace(-width/2, width/2, W)
    X, Y = np.meshgrid(xs, ys)
    gx = math.tan(math.radians(math.atan(grade_percent/100.0)*180/math.pi))  # but easier: z = x * tan(theta)
    # Simplify: z grade via slope = grade_percent/100
    z_grade = (grade_percent/100.0) * X
    # bank (superelevation) rotates crossfall: z varies linearly with Y
    z_bank = bank_e * Y
    Z = z_grade + z_bank + center[2]
    return go.Surface(x=X+center[0], y=Y+center[1], z=Z, colorscale=[[0,color],[1,color]], showscale=False, opacity=0.85, name=name)

def headlight_ray(x0, y0, z0, pitch_deg=-1.0, length=120.0, name="headlight"):
    """Simple ray from car front representing headlight/sight line."""
    pitch = math.radians(pitch_deg)
    xs = np.linspace(0, length, 50)
    ys = np.zeros_like(xs)
    zs = xs*math.tan(pitch)
    return go.Scatter3d(x=x0+xs, y=y0+ys, z=z0+zs, mode="lines", line=dict(width=6), name=name)

# ================ UI =================
st.title("🚗 ATJ 8/86 — Practical Road Lab")

tabs = st.tabs(["Stopping Sight Distance (SSD) on Grade", "Horizontal Curve & Superelevation", "Free Q&A (ATJ PDF)"])

# ---------- Tab 1: SSD ----------
with tabs[0]:
    colL, colR = st.columns([0.52, 0.48])
    with colL:
        st.subheader("Inputs")
        v = st.slider("Speed (km/h)", 30, 140, 90, 5)
        t = st.slider("Perception–Reaction time t (s)", 1.0, 3.0, 2.5, 0.1)
        f = st.slider("Longitudinal friction f", 0.20, 0.50, 0.35, 0.01)
        grade = st.slider("Grade (%): + upgrade, − downgrade", -8, 8, 0, 1)
        lane_w = st.slider("Lane width (m)", 3.0, 4.0, 3.5, 0.1)
        car_h = st.slider("Driver eye height (m)", 0.9, 1.5, 1.1, 0.05)
        # compute
        ssd = ssd_on_grade(v, t, f, grade)
        st.metric("Calculated SSD (m)", f"{ssd:,.1f}")
        st.caption("Formula: SSD = v·t + v² / (2·g·(f ± G)). Downgrade uses (f − |G|).")
        # pull PDF extracts
        q_topic = "stopping sight distance braking grade"
        results = search_snippets(q_topic, VEC, X, CHUNKS, topk=RESULT_SNIPPETS)
        if results and results[0][0] >= SIM_THRESHOLD:
            st.write("**ATJ extracts (related):**")
            for i, (score, ch) in enumerate(results):
                st.markdown(f"- *(p.{ch.page}, score={score:.3f})* {highlight(ch.text, 'stopping sight distance')}", unsafe_allow_html=True)
        elif PAGES:
            st.info("No strong snippet found. Try searching in the Q&A tab for exact wording.")

    with colR:
        st.subheader("3D road & car (SSD demo)")
        road_len = max(120.0, ssd + 40.0)   # ensure the stopping line is visible
        # Build road with grade
        plane = road_plane(length=road_len, width=2*lane_w, grade_percent=grade, bank_e=0.0, center=(0,0,0))
        # Place car near x=10
        car_front_bumper_to_cg = 1.0
        car = car_mesh(center=(10.0, 0.0, 0.0), L=4.4, W=1.8, H=1.5, yaw_deg=0, color="crimson")
        # Stopping line at SSD from front bumper
        stop_x = 10.0 + ssd
        stop = go.Scatter3d(x=[stop_x, stop_x], y=[-lane_w/2, lane_w/2], z=[(grade/100.0)*stop_x]*2,
                            mode="lines", line=dict(width=12), name="Stopping line")
        # Headlight/sight line from approx headlight height (driver eye line demo)
        headlight_height = car_h
        ray = headlight_ray(x0=10.0, y0=0.0, z0=headlight_height, pitch_deg=-1.0, length=road_len-10.0, name="Sight/Headlight")

        fig = go.Figure(data=[plane, car, stop, ray])
        fig.update_layout(
            scene=dict(
                xaxis_title="x (m)",
                yaxis_title="y (m)",
                zaxis_title="z (m)",
                aspectmode="manual",
                aspectratio=dict(x=2, y=0.6, z=0.25),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Move the sliders to see how SSD shifts the stopping line and how grade changes road elevation.")

# ---------- Tab 2: Curve & Superelevation ----------
with tabs[1]:
    colL, colR = st.columns([0.52, 0.48])
    with colL:
        st.subheader("Inputs")
        v2 = st.slider("Speed (km/h)", 30, 140, 80, 5, key="v2")
        R = st.slider("Curve radius R (m)", 60, 1500, 300, 10)
        f_lat = st.slider("Lateral friction f", 0.05, 0.20, 0.12, 0.01)
        e_set = st.slider("Provided superelevation e (decimal)", 0.00, 0.12, 0.06, 0.01)
        lane_w2 = st.slider("Lane + shoulder width modeled (m)", 3.0, 8.0, 6.0, 0.1)
        # computes
        e_need = e_required(v2, R, f_lat)
        demand = (kmh_to_ms(v2)**2)/(G*R)
        st.metric("Required e (decimal)", f"{e_need:.3f}")
        st.metric("Demand (e + f)", f"{demand:.3f}")
        ok = (e_set + f_lat) >= demand
        st.success("OK: Provided e + f meets demand ✅" if ok else "Not OK: Increase e or reduce speed/radius ❌")
        # extracts
        q_topic2 = "superelevation horizontal curve friction"
        results2 = search_snippets(q_topic2, VEC, X, CHUNKS, topk=RESULT_SNIPPETS)
        if results2 and results2[0][0] >= SIM_THRESHOLD:
            st.write("**ATJ extracts (related):**")
            for i, (score, ch) in enumerate(results2):
                st.markdown(f"- *(p.{ch.page}, score={score:.3f})* {highlight(ch.text, 'superelevation curve')}", unsafe_allow_html=True)
        elif PAGES:
            st.info("No strong snippet found. Try the Q&A tab for exact phrasing.")

    with colR:
        st.subheader("3D banked curve with car")
        # Make a short arc of the curve as a banked ribbon
        theta_span = max(20.0, min(120.0, 60.0 * (300.0/R)))  # shorter arc for big R
        theta = np.linspace(0, math.radians(theta_span), 80)
        x_arc = R*np.sin(theta)
        y_arc = R*(1 - np.cos(theta))
        # Apply banking: crossfall e_set across width
        width = lane_w2
        y_left = y_arc - width/2
        y_right = y_arc + width/2
        z_left = -e_set*(width/2)
        z_right = e_set*(width/2)
        # Create ribbon surface by sweeping along arc
        X = np.vstack([x_arc, x_arc])
        Y = np.vstack([y_left, y_right])
        Z = np.vstack([np.full_like(x_arc, z_left), np.full_like(x_arc, z_right)])

        road = go.Surface(x=X, y=Y, z=Z, colorscale=[[0,"#808080"],[1,"#808080"]], showscale=False, opacity=0.9, name="Banked curve")

        # Place a car near start of arc, yaw aligned to tangent
        yaw_deg = math.degrees(theta[10]) if len(theta) > 10 else 0.0
        car2 = car_mesh(center=(x_arc[10], y_arc[10], 0.75*e_set*width/2), L=4.4, W=1.8, H=1.5, yaw_deg=yaw_deg, color="royalblue", name="car")
        fig2 = go.Figure(data=[road, car2])
        fig2.update_layout(
            scene=dict(
                xaxis_title="x (m)",
                yaxis_title="y (m)",
                zaxis_title="z (m)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Demand = v²/(gR). You supply e and f. Increase e or reduce speed if demand > e+f.")

# ---------- Tab 3: Free Q&A ----------
with tabs[2]:
    st.subheader("Ask ATJ (PDF grounded)")
    query = st.text_input("Your question", placeholder="e.g., Define stopping sight distance and list key factors.")
    strict = st.checkbox("Strict mode (refuse weak evidence)", value=True)
    if st.button("Search"):
        if not CHUNKS:
            st.error("PDF not found or text not extracted. Place the PDF at the repo root as 'BPIS_ATJ_8-86_19062020.pdf'.")
        else:
            hits = search_snippets(query, VEC, X, CHUNKS, topk=RESULT_SNIPPETS)
            if not hits:
                st.warning("No matches found.")
            else:
                best_score, best_chunk = hits[0]
                if strict and best_score < SIM_THRESHOLD:
                    st.warning("Not in the guideline (ATJ 8/86).")
                else:
                    for i, (score, ch) in enumerate(hits):
                        with st.expander(f"p.{ch.page}  ·  score={score:.3f}  ·  #{i}"):
                            st.markdown(highlight(ch.text, query), unsafe_allow_html=True)
                    if USE_OPENAI:
                        context = "\n\n".join([f"(p.{c.page}) {c.text}" for _, c in hits])
                        try:
                            resp = oai.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": "Answer ONLY from the ATJ context. Cite pages like (p.X). If absent, say 'Not in the guideline (ATJ 8/86)'."},
                                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
                                ],
                                temperature=0.05,
                            )
                            st.markdown("### Composed answer")
                            st.write(resp.choices[0].message.content)
                        except Exception:
                            st.info("OpenAI composition skipped (no key or API error).")

st.caption("Note: Visuals are illustrative. Always confirm with ATJ design tables and local practice.")
