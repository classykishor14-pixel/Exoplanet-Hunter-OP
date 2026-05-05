"""
==============================================================================
  EXOPLANET DETECTION SYSTEM — Streamlit Web Application
  app.py  [UPGRADED: HD CSS Nebula + Full Glassmorphism HUD]
==============================================================================
HOW TO RUN
    pip install streamlit lightkurve astropy matplotlib numpy
    streamlit run app.py
==============================================================================

BACKGROUND STRATEGY
-------------------
We use a pure-CSS multi-layer nebula instead of fetching an external image.
This guarantees:
  • Zero network dependency  → loads instantly on any connection / cloud host
  • No CORS or CDN failures  → works on Streamlit Cloud, local, everywhere
  • Instant first-paint      → no blank screen while image loads
  • Pixel-perfect parallax   → background-attachment:fixed on all layers

Layer stack (painter's order, bottom → top):
  1. Deep navy void          linear-gradient base
  2. Purple nebula cloud     radial ellipse, centre-left
  3. Teal star-cluster glow  radial ellipse, top-right
  4. Amber emission wisps    radial ellipse, bottom-right
  5. Faint blue core haze    large radial overlay, keeps centre readable
  6. Radial vignette scrim   darkens edges → forces text contrast at margins
  7. SVG film-grain noise    inline data-URI, adds photographic texture

GLASSMORPHISM STRATEGY
----------------------
Every container gets the same three-property glass treatment:
  background : rgba(…, alpha)      — semi-transparent fill
  backdrop-filter : blur(Npx)      — blurs whatever is behind the element
  border : 1px solid rgba(…, low) — hairline light edge ("frosted" rim)
  box-shadow : inset highlight +
               outer depth shadow  — simulates glass thickness
"""

import streamlit as st

# set_page_config MUST be the absolute first Streamlit call in the file
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

import warnings
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from astropy.time import Time as AstropyTime
import astropy.units as u

warnings.filterwarnings("ignore")

# ── Astrophysics constants ────────────────────────────────────────────────────
BLS_MIN_PERIOD   = 0.5
BLS_MAX_PERIOD   = 20.0
BLS_MIN_DURATION = 0.01
BLS_MAX_DURATION = 0.25
BLS_FREQ_FACTOR  = 10
N_BINS           = 60
SG_WINDOW        = 101
SG_POLY          = 3

# ── Matplotlib palette ────────────────────────────────────────────────────────
BG_DARK  = "#03050f"
BG_PANEL = "#0d1525"
C_GRID   = "#1a2a44"
C_TICK   = "#6a85b0"
C_RAW    = "#7ecfff"
C_TREND  = "#ff6b6b"
C_FLAT   = "#ffe66d"
C_CLEAN  = "#a8ff78"
C_FOLD   = "#4a7cff"
C_BIN    = "#ffffff"
C_PEAK   = "#ff4f6e"
C_PERI   = "#00d4ff"
C_ANNO   = "#ffe66d"

# =============================================================================
# ░░░  MASTER CSS — HD NEBULA BACKGROUND + GLASSMORPHISM HUD  ░░░
# =============================================================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Exo+2:wght@300;400;600;800&display=swap');

/* ════════════════════════════════════════════════════════════════════════════
   NEBULA BACKGROUND — applied directly to every Streamlit root container.

   WHY THIS APPROACH:
   Streamlit renders: <body> → stApp → stAppViewContainer → stMain → ...
   Each layer can have its own background that paints over the one below.
   Using ::before pseudo-elements with z-index:-2 fails because Streamlit's
   intermediate containers intercept and cover them.

   SOLUTION: Apply the exact same nebula background-image to ALL root
   containers simultaneously with background-attachment:fixed. Because
   fixed backgrounds are always positioned relative to the VIEWPORT (not
   the element), every container shows the identical scene — the layers
   visually merge into one seamless image regardless of which DOM node
   is actually "on top".
════════════════════════════════════════════════════════════════════════════ */

/* Shared nebula definition — reused on every container */
html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stDecoration"],
section[data-testid="stSidebar"] ~ div,
.main {
    background-color: #02030a !important;
    background-image:
        /* Grain — tiled SVG noise, zero network cost */
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.68' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23g)' opacity='0.04'/%3E%3C/svg%3E"),
        /* Edge vignette — darkens corners, forces text contrast */
        radial-gradient(ellipse 120% 120% at 50% 50%, transparent 28%, rgba(1,2,10,0.80) 72%, rgba(1,2,8,0.97) 100%),
        /* Blue core haze */
        radial-gradient(ellipse 80% 65% at 52% 48%, rgba(8,18,55,0.55) 0%, transparent 70%),
        /* Amber emission wisps — bottom-right */
        radial-gradient(ellipse 52% 38% at 82% 82%, rgba(210,95,12,0.30) 0%, rgba(160,55,8,0.14) 45%, transparent 72%),
        /* Teal star-cluster glow — top-right */
        radial-gradient(ellipse 48% 42% at 80% 14%, rgba(0,185,230,0.34) 0%, rgba(0,100,175,0.16) 50%, transparent 78%),
        /* Purple nebula cloud — centre-left */
        radial-gradient(ellipse 68% 58% at 16% 52%, rgba(105,22,165,0.44) 0%, rgba(55,10,95,0.22) 48%, transparent 78%),
        /* Deep navy void — absolute base */
        linear-gradient(158deg, #04060f 0%, #020309 55%, #030509 100%) !important;

    background-size:
        400px 400px,
        100% 100%, 100% 100%, 100% 100%,
        100% 100%, 100% 100%, 100% 100% !important;

    background-repeat:
        repeat,
        no-repeat, no-repeat, no-repeat,
        no-repeat, no-repeat, no-repeat !important;

    /* CRITICAL: fixed means the background is anchored to the viewport,
       not the element — so every container shows the same nebula scene */
    background-attachment: fixed !important;

    min-height: 100vh;
}

/* Star-field — 10 fixed 1px radial dots that slowly pulse */
body::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background-image:
        radial-gradient(1.2px 1.2px at  8%  12%, rgba(255,255,255,0.60) 0%, transparent 100%),
        radial-gradient(1.0px 1.0px at 22%  78%, rgba(255,255,255,0.45) 0%, transparent 100%),
        radial-gradient(1.2px 1.2px at 37%  31%, rgba(180,220,255,0.55) 0%, transparent 100%),
        radial-gradient(1.0px 1.0px at 55%   9%, rgba(255,255,255,0.50) 0%, transparent 100%),
        radial-gradient(1.2px 1.2px at 68%  62%, rgba(200,230,255,0.42) 0%, transparent 100%),
        radial-gradient(1.0px 1.0px at 81%  24%, rgba(255,255,255,0.56) 0%, transparent 100%),
        radial-gradient(1.2px 1.2px at 92%  87%, rgba(180,210,255,0.46) 0%, transparent 100%),
        radial-gradient(1.0px 1.0px at 14%  55%, rgba(255,255,255,0.38) 0%, transparent 100%),
        radial-gradient(1.2px 1.2px at 46%  90%, rgba(255,255,255,0.52) 0%, transparent 100%),
        radial-gradient(1.0px 1.0px at 73%  43%, rgba(200,240,255,0.42) 0%, transparent 100%);
    background-size: 100% 100%;
    background-repeat: no-repeat;
    background-attachment: fixed;
    animation: starTwinkle 7s ease-in-out infinite alternate;
}
@keyframes starTwinkle {
    0%   { opacity: 0.50; }
    50%  { opacity: 1.00; }
    100% { opacity: 0.60; }
}


/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — MAIN CONTENT PANEL
   alpha=0.46 lets the nebula show through while still anchoring text
════════════════════════════════════════════════════════════════════════════ */
.main .block-container {
    background: rgba(3, 7, 20, 0.46) !important;
    backdrop-filter:          blur(22px) saturate(170%) brightness(0.95) !important;
    -webkit-backdrop-filter:  blur(22px) saturate(170%) brightness(0.95) !important;
    border-radius: 20px !important;
    padding: 2rem 2.5rem !important;
    margin-top: 0.6rem !important;
    border: 1px solid rgba(120, 180, 255, 0.11) !important;
    box-shadow:
        /* Outer depth */
        0 12px 48px rgba(0, 0, 0, 0.60),
        /* Top highlight — simulates light hitting glass rim */
        inset 0  1px 0 rgba(255, 255, 255, 0.07),
        /* Bottom shadow inside — simulates glass thickness */
        inset 0 -1px 0 rgba(0,   0,   0,   0.35) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — SIDEBAR
   Stronger blur (28px) because sidebar is narrower and needs more opacity
════════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: rgba(3, 7, 22, 0.72) !important;
    backdrop-filter:         blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border-right: 1px solid rgba(90, 150, 255, 0.14) !important;
    box-shadow:
        4px 0 36px rgba(0, 0, 0, 0.60),
        inset -1px 0 0 rgba(255, 255, 255, 0.04) !important;
    min-width: 268px !important;
    max-width: 325px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
}

/* Hide collapse toggle — sidebar stays open permanently */
button[data-testid="collapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
.css-1lcbmhc,
.css-1d391kg { display: none !important; }

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — MATPLOTLIB FIGURE WRAPPERS
════════════════════════════════════════════════════════════════════════════ */
div[data-testid="stPyplotRootElement"] {
    background: rgba(2, 6, 18, 0.50) !important;
    backdrop-filter:         blur(12px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(12px) saturate(140%) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(50, 90, 180, 0.20) !important;
    padding: 8px !important;
    box-shadow:
        0 6px 30px rgba(0, 0, 0, 0.55),
        inset 0 1px 0 rgba(255, 255, 255, 0.045) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — ALERT / INFO / ERROR / SPINNER BOXES
════════════════════════════════════════════════════════════════════════════ */
div[data-testid="stExpander"],
div[data-testid="stInfo"],
div[data-testid="stSuccess"],
div[data-testid="stWarning"],
div[data-testid="stError"] {
    background: rgba(5, 12, 30, 0.62) !important;
    backdrop-filter:         blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(80, 130, 220, 0.22) !important;
    border-radius: 12px !important;
}

div[data-testid="stSpinner"] > div {
    background: rgba(3, 9, 24, 0.70) !important;
    backdrop-filter:         blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0, 212, 255, 0.18) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — STAT CARDS
   Each card is its own mini glass pane with a coloured top-accent bar
   and a frosted shimmer pseudo-element
════════════════════════════════════════════════════════════════════════════ */
.stat-card {
    background:              rgba(5, 12, 32, 0.58);
    backdrop-filter:         blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid rgba(65, 105, 200, 0.24);
    border-radius: 14px;
    padding: 15px 22px;
    flex: 1; min-width: 140px;
    position: relative; overflow: hidden;
    box-shadow:
        0 6px 30px rgba(0, 0, 0, 0.55),
        inset 0  1px 0 rgba(255, 255, 255, 0.09),
        inset 0 -1px 0 rgba(0,   0,   0,   0.24);
    transition: transform 0.22s ease, box-shadow 0.28s ease, border-color 0.28s ease;
}
.stat-card:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow:
        0 12px 42px rgba(0, 0, 0, 0.65),
        0  0   26px rgba(0, 160, 255, 0.14),
        inset 0 1px 0 rgba(255, 255, 255, 0.11);
    border-color: rgba(90, 160, 255, 0.40);
}

/* Coloured accent bar along the top edge of each card */
.stat-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: 14px 14px 0 0;
}

/* Diagonal frosted shimmer sweep across the card face */
.stat-card::after {
    content: "";
    position: absolute; top: 0; left: -90%;
    width: 60%; height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.05),
        transparent
    );
    transform: skewX(-14deg);
    pointer-events: none;
}

.stat-card.blue::before  { background: linear-gradient(90deg, #00d4ff, #0070ff); }
.stat-card.green::before { background: linear-gradient(90deg, #a8ff78, #00d460); }
.stat-card.gold::before  { background: linear-gradient(90deg, #ffe66d, #ff9800); }
.stat-card.pink::before  { background: linear-gradient(90deg, #ff4f6e, #ff0055); }
.stat-card.cyan::before  { background: linear-gradient(90deg, #4a7cff, #00d4ff); }

.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.60rem; letter-spacing: 0.20em; text-transform: uppercase;
    color: #45628a; margin-bottom: 6px;
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.24rem; font-weight: 700; color: #eaf4ff;
    text-shadow: 0 0 16px rgba(0, 180, 255, 0.24);
}
.stat-unit { font-size: 0.67rem; color: #45628a; margin-left: 3px; }

/* ════════════════════════════════════════════════════════════════════════════
   TYPOGRAPHY
════════════════════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    color: #ccdff5;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.9rem; font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #a8ff78 52%, #ffe66d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px; line-height: 1.05; margin-bottom: 0.28rem;
    filter: drop-shadow(0 0 32px rgba(0, 212, 255, 0.42));
}
.hero-sub {
    font-size: 0.88rem; color: #5e7eaa;
    letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 2rem;
    text-shadow: 0 1px 14px rgba(0, 0, 0, 0.90);
}
.sidebar-label {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: #3d5880; margin-bottom: 0.3rem;
}
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.70rem;
    letter-spacing: 0.24em; text-transform: uppercase; color: #3d5580;
    border-bottom: 1px solid rgba(45, 75, 135, 0.42);
    padding-bottom: 8px; margin: 2.4rem 0 1.1rem 0;
}
.desc-text {
    font-size: 0.82rem; color: #90b4d5; line-height: 1.68;
    margin-bottom: 0.95rem;
    text-shadow: 0 1px 10px rgba(0, 0, 0, 0.80);
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — WIDGET OVERRIDES
════════════════════════════════════════════════════════════════════════════ */
.stTextInput > div > div > input {
    background:              rgba(4, 11, 30, 0.75) !important;
    backdrop-filter:         blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(45, 80, 140, 0.55) !important;
    color: #c8d8f0 !important;
    border-radius: 9px !important;
    box-shadow:
        inset 0 2px 12px rgba(0, 0, 0, 0.36),
        inset 0 1px 0   rgba(255, 255, 255, 0.05) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00d4ff !important;
    box-shadow:
        0 0 0 2px rgba(0, 212, 255, 0.22),
        inset 0 2px 10px rgba(0, 0, 0, 0.30) !important;
}
.stTextInput > div > div > input::placeholder { color: #2e4a70 !important; }

/* Primary action button */
.stButton > button[kind="primary"] {
    background:              rgba(0, 212, 255, 0.07) !important;
    backdrop-filter:         blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(0, 212, 255, 0.44) !important;
    color: #00d4ff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important; letter-spacing: 0.12em !important;
    border-radius: 11px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07) !important;
    transition: all 0.24s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background:  rgba(0, 212, 255, 0.17) !important;
    border-color: rgba(0, 212, 255, 0.70) !important;
    box-shadow:
        0 0 32px rgba(0, 212, 255, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
    transform: translateY(-1px) !important;
}

/* Secondary (default) buttons */
.stButton > button:not([kind="primary"]) {
    background:              rgba(8, 20, 50, 0.58) !important;
    backdrop-filter:         blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(50, 90, 160, 0.40) !important;
    color: #7aabcc !important;
    border-radius: 9px !important;
    transition: all 0.22s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    background:  rgba(10, 28, 65, 0.72) !important;
    border-color: rgba(0, 180, 255, 0.38) !important;
    color: #a0d0ee !important;
}

/* Slider track glass pill */
.stSlider > div > div > div {
    background: rgba(5, 14, 38, 0.68) !important;
    border-radius: 8px !important;
}

/* Radio buttons */
div.row-widget.stRadio > div {
    flex-direction: row; gap: 15px;
    justify-content: center; margin-bottom: 10px;
}
div.row-widget.stRadio > div > label {
    background: rgba(5, 12, 32, 0.60) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(50, 90, 160, 0.32) !important;
    border-radius: 8px !important; padding: 4px 14px !important;
    color: #7aabcc !important; font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}
div.row-widget.stRadio > div > label:hover {
    border-color: rgba(0, 212, 255, 0.45) !important;
    color: #c0e0f8 !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   ANIMATIONS
════════════════════════════════════════════════════════════════════════════ */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes glassIn {
    from { opacity: 0; transform: scale(0.96) translateY(14px); }
    to   { opacity: 1; transform: scale(1.00) translateY(0);    }
}

.animate-in { animation: fadeUp  0.58s cubic-bezier(0.22, 0.68, 0, 1.20) both; }
.glass-in   { animation: glassIn 0.62s cubic-bezier(0.22, 0.68, 0, 1.10) both; }
.delay-1    { animation-delay: 0.08s; }
.delay-2    { animation-delay: 0.20s; }
.delay-3    { animation-delay: 0.34s; }

/* ── Misc ── */
#MainMenu, footer, header { visibility: hidden; }

/* Styled thin scrollbar */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: rgba(2, 5, 16, 0.55); }
::-webkit-scrollbar-thumb {
    background: rgba(0, 120, 210, 0.36); border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(0, 168, 255, 0.55); }

/* Sidebar internal divider lines */
section[data-testid="stSidebar"] hr {
    border-color: rgba(40, 70, 130, 0.30) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar state lock (session state + JS belt-and-suspenders) ───────────────
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

st.markdown("""
<script>
window.addEventListener('load', function () {
    setTimeout(function () {
        var btn = document.querySelector('[data-testid="collapsedControl"]');
        if (btn) btn.style.display = 'none';
        var sb = document.querySelector('[data-testid="stSidebar"]');
        if (sb) { sb.style.minWidth = '268px'; sb.style.transform = 'none'; }
    }, 600);
});
</script>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS — MATPLOTLIB DARK THEME
# =============================================================================

def apply_dark_theme(ax):
    ax.set_facecolor("#060e1e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d50"); sp.set_linewidth(0.8)
    ax.tick_params(colors=C_TICK, labelsize=8.5)
    ax.grid(True, linestyle=":", linewidth=0.35, color=C_GRID, alpha=0.9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))

def make_fig(w=13, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#03060f")
    apply_dark_theme(ax)
    return fig, ax


# =============================================================================
# PIPELINE — Kepler + TESS integrated, returns only plain numpy arrays
# (plain numpy arrays are always pickle-serialisable by @st.cache_data)
# =============================================================================

def clear_lk_cache():
    for path in [Path.home() / ".lightkurve" / "cache",
                 Path.home() / ".lightkurve-cache"]:
        if path.exists():
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def fetch_and_clean(target: str, mission: str, time_segment: int):
    """
    Download + clean a Kepler quarter or TESS sector.
    Returns eleven plain numpy arrays — fully pickle-safe for @st.cache_data.
    """
    if mission == "Kepler":
        result = lk.search_lightcurve(
            target, mission="Kepler", quarter=time_segment,
            cadence="long", author="Kepler",
        )
        err_msg = f"No Kepler data for '{target}' in Q{time_segment}."
    else:
        result = lk.search_lightcurve(
            target, mission="TESS", sector=time_segment, author="SPOC",
        )
        err_msg = f"No TESS data for '{target}' in Sector {time_segment}."

    if len(result) == 0:
        raise ValueError(f"{err_msg} Check the star name or try a different segment.")

    lc = result.download_all().stitch()
    cols_lower = [c.lower() for c in lc.columns]
    if "pdcsap_flux" in cols_lower:
        lc = lc.select_flux("pdcsap_flux")
    elif "sap_flux" in cols_lower:
        lc = lc.select_flux("sap_flux")

    lc_raw = lc.remove_nans().normalize()
    lc_flat, lc_trend = lc_raw.flatten(
        window_length=SG_WINDOW, polyorder=SG_POLY,
        return_trend=True, break_tolerance=5, niters=3, sigma=3,
    )
    lc_clean = lc_flat.remove_outliers(sigma=4.0, sigma_lower=1e5)

    # Extract everything to plain numpy — no LightCurve objects leave this fn
    return (
        np.array(lc_raw.time.value),   np.array(lc_raw.flux.value),   np.array(lc_raw.flux_err.value),
        np.array(lc_trend.time.value), np.array(lc_trend.flux.value),
        np.array(lc_flat.time.value),  np.array(lc_flat.flux.value),  np.array(lc_flat.flux_err.value),
        np.array(lc_clean.time.value), np.array(lc_clean.flux.value), np.array(lc_clean.flux_err.value),
    )


@st.cache_data(show_spinner=False)
def run_bls_cached(time_arr, flux_arr, err_arr):
    """BLS periodogram — accepts and returns plain numpy arrays only."""
    bls = BoxLeastSquares(
        time_arr * u.day,
        flux_arr * u.dimensionless_unscaled,
        dy=err_arr * u.dimensionless_unscaled,
    )
    dur_grid = np.logspace(
        np.log10(BLS_MIN_DURATION), np.log10(BLS_MAX_DURATION), 40
    ) * u.day
    pg = bls.autopower(
        duration=dur_grid,
        minimum_period=BLS_MIN_PERIOD * u.day,
        maximum_period=BLS_MAX_PERIOD * u.day,
        frequency_factor=BLS_FREQ_FACTOR,
        minimum_n_transit=2,
    )
    best_idx = int(np.argmax(pg.power))
    return (
        np.array(pg.period.value), np.array(pg.power),
        float(pg.period[best_idx].value),
        float(pg.transit_time[best_idx].value),
        float(pg.duration[best_idx].value),
        float(pg.depth[best_idx]),
    )


def phase_fold_arrays(clean_time, clean_flux, clean_ferr, period, t0):
    """Phase-fold and bin. Uses astropy.time.Time directly (lk.time.Time doesn't exist)."""
    lc_tmp = lk.LightCurve(
        time=AstropyTime(clean_time, format="bkjd", scale="tdb"),
        flux=clean_flux,
        flux_err=clean_ferr,
    )
    lc_folded   = lc_tmp.fold(period=period, epoch_time=t0)
    phase_hours = lc_folded.phase.value * period * 24.0
    fv          = lc_folded.flux.value

    edges   = np.linspace(phase_hours.min(), phase_hours.max(), N_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    meds    = np.full(N_BINS, np.nan)
    errs    = np.full(N_BINS, np.nan)
    for i in range(N_BINS):
        m = (phase_hours >= edges[i]) & (phase_hours < edges[i + 1])
        if m.sum() > 0:
            meds[i] = np.median(fv[m])
            errs[i] = 1.48 * np.median(np.abs(fv[m] - meds[i])) / np.sqrt(m.sum())
    return fv, phase_hours, centres, meds, errs


# =============================================================================
# PLOT BUILDERS — accept plain numpy arrays
# =============================================================================

def plot_raw(raw_time, raw_flux, raw_ferr, trend_time, trend_flux):
    fig, ax = make_fig(h=4)
    ax.plot(raw_time, raw_flux, color=C_RAW, lw=0.5, alpha=0.7, zorder=2, label="Raw flux")
    ax.fill_between(raw_time, raw_flux - raw_ferr, raw_flux + raw_ferr,
                    color=C_RAW, alpha=0.10, zorder=1)
    ax.plot(trend_time, trend_flux, color=C_TREND, lw=1.8, alpha=0.9, zorder=3,
            label=f"SG trend  (window={SG_WINDOW}, poly={SG_POLY})")
    ax.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Time  [Days]", color=C_TICK, fontsize=9)
    ax.set_ylabel("Normalised Flux", color=C_RAW, fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.25,
              facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    ax.set_title("Raw Light Curve  +  Stellar Trend (red = what gets removed)",
                 color=C_TICK, fontsize=9, loc="left", pad=6)
    fig.tight_layout(pad=1.5)
    return fig


def plot_flat(flat_time, flat_flux, clean_time, clean_flux):
    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor("#03060f")
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    apply_dark_theme(ax1); apply_dark_theme(ax2)

    ax1.plot(flat_time, flat_flux, color=C_FLAT, lw=0.5, alpha=0.65, zorder=2,
             label="Flattened  (stellar trend removed)")
    ax1.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Flux", color=C_FLAT, fontsize=9)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.25,
               facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    ax1.set_title("Flattened (top)  →  After outlier removal (bottom)",
                  color=C_TICK, fontsize=9, loc="left", pad=6)

    ax2.plot(clean_time, clean_flux, color=C_CLEAN, lw=0.5, alpha=0.65, zorder=2,
             label="Planet-search ready  (outliers removed)")
    ax2.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    ax2.set_xlabel("Time  [Days]", color=C_TICK, fontsize=9)
    ax2.set_ylabel("Flux", color=C_CLEAN, fontsize=9)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.25,
               facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    noise = float(np.std(clean_flux)) * 1e6
    ax2.text(0.01, 0.08, f"Noise floor ≈ {noise:.0f} ppm",
             transform=ax2.transAxes, fontsize=7.5, color=C_CLEAN, style="italic")
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    fig.tight_layout(pad=1.5)
    return fig


def plot_bls(periods, power, clean_time, clean_flux, clean_ferr,
             best_period, best_t0, best_duration, best_depth):
    dur_h      = best_duration * 24.0
    half_dur_h = dur_h / 2.0
    margin     = max(best_depth * 6, 0.0015)

    fold_flux, phase_hours, bin_c, bin_m, bin_e = phase_fold_arrays(
        clean_time, clean_flux, clean_ferr, best_period, best_t0,
    )

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#03060f")
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42,
                              top=0.90, bottom=0.07, left=0.07, right=0.97)
    top_ax  = fig.add_subplot(outer[0])
    bot     = gridspec.GridSpecFromSubplotSpec(
                  1, 2, subplot_spec=outer[1], wspace=0.32, width_ratios=[1, 1.1])
    fold_ax = fig.add_subplot(bot[0])
    zoom_ax = fig.add_subplot(bot[1])
    for ax in (top_ax, fold_ax, zoom_ax):
        apply_dark_theme(ax)

    peak_pwr = float(power.max())
    top_ax.plot(periods, power, color=C_PERI, lw=0.7, alpha=0.85, zorder=2)
    top_ax.fill_between(periods, 0, power, color=C_PERI, alpha=0.08, zorder=1)
    top_ax.axvline(best_period, color=C_PEAK, lw=1.6, ls="--", alpha=0.9, zorder=3)
    top_ax.scatter([best_period], [peak_pwr], color=C_PEAK, s=80, zorder=5,
                   label=f"Best period = {best_period:.5f} d")
    top_ax.annotate(
        f"P = {best_period:.5f} d",
        xy=(best_period, peak_pwr),
        xytext=(best_period + (BLS_MAX_PERIOD - best_period) * 0.10, peak_pwr * 0.86),
        color=C_ANNO, fontsize=8.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ANNO, lw=1.0,
                        connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK,
                  edgecolor=C_ANNO, alpha=0.85), zorder=6,
    )
    harmonic = best_period / 2.0
    if harmonic > BLS_MIN_PERIOD:
        top_ax.axvline(harmonic, color="#555", lw=0.8, ls=":", alpha=0.5)
        top_ax.text(harmonic, peak_pwr * 0.04, " P/2", color="#444", fontsize=7, va="bottom")
    top_ax.set_xlabel("Trial Period  [days]", color=C_TICK, fontsize=9.5)
    top_ax.set_ylabel("BLS Power", color=C_PERI, fontsize=9.5)
    top_ax.set_xlim(BLS_MIN_PERIOD, BLS_MAX_PERIOD)
    top_ax.set_ylim(0, peak_pwr * 1.18)
    top_ax.legend(loc="upper right", fontsize=8.5, framealpha=0.25,
                  facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    top_ax.set_title("① BLS Periodogram — tallest spike = planet orbital period",
                     color=C_TICK, fontsize=9, pad=6, loc="left")

    valid = ~np.isnan(bin_m)
    fold_ax.scatter(phase_hours, fold_flux, color=C_FOLD, s=1.8, alpha=0.22, zorder=2)
    fold_ax.errorbar(bin_c[valid], bin_m[valid], yerr=bin_e[valid],
                     fmt="o", color=C_BIN, ms=3.5, lw=0.8,
                     elinewidth=0.7, capsize=1.5, zorder=4, label=f"{N_BINS} bins")
    fold_ax.axvspan(-half_dur_h, half_dur_h, color=C_PEAK, alpha=0.10, zorder=1,
                    label=f"Transit ({dur_h:.2f} h)")
    fold_ax.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    fold_ax.set_ylim(1.0 - margin, 1.0 + margin * 0.5)
    fold_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    fold_ax.set_xlabel("Phase  [hours]", color=C_TICK, fontsize=9.5)
    fold_ax.set_ylabel("Normalised Flux", color=C_FOLD, fontsize=9.5)
    fold_ax.legend(loc="lower center", fontsize=7.5, framealpha=0.25,
                   facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    fold_ax.set_title("② Full Phase-Folded Curve", color=C_TICK, fontsize=9, pad=5, loc="left")

    zoom_h   = dur_h * 4
    zs       = np.abs(phase_hours) <= zoom_h
    zb       = valid & (np.abs(bin_c) <= zoom_h)
    bottom_y = 1.0 - best_depth
    arr_x    = dur_h * 1.9

    zoom_ax.scatter(phase_hours[zs], fold_flux[zs], color=C_FOLD, s=5, alpha=0.45, zorder=2)
    zoom_ax.errorbar(bin_c[zb], bin_m[zb], yerr=bin_e[zb],
                     fmt="o", color=C_BIN, ms=5, lw=1.0, elinewidth=0.9, capsize=2.5, zorder=4)
    zoom_ax.axvspan(-half_dur_h, half_dur_h, color=C_PEAK, alpha=0.12, zorder=1)
    zoom_ax.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    zoom_ax.annotate("", xy=(arr_x, bottom_y), xytext=(arr_x, 1.0),
                     arrowprops=dict(arrowstyle="<->", color=C_ANNO, lw=1.3, mutation_scale=10))
    zoom_ax.text(arr_x * 1.05, (1.0 + bottom_y) / 2,
                 f"Δ = {best_depth*1e6:.0f} ppm", color=C_ANNO, fontsize=8, va="center")
    zoom_ax.annotate("", xy=(-half_dur_h, bottom_y * 0.9999),
                     xytext=(half_dur_h, bottom_y * 0.9999),
                     arrowprops=dict(arrowstyle="<->", color="#88aaff", lw=1.0, mutation_scale=8))
    zoom_ax.text(0, bottom_y - margin * 0.35,
                 f"Duration = {dur_h:.2f} h", color="#88aaff", fontsize=7.5, ha="center")
    zoom_ax.set_xlim(-zoom_h, zoom_h)
    zoom_ax.set_ylim(1.0 - margin, 1.0 + margin * 0.4)
    zoom_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    zoom_ax.set_xlabel("Phase  [hours]", color=C_TICK, fontsize=9.5)
    zoom_ax.set_ylabel("Normalised Flux", color=C_FOLD, fontsize=9.5)
    zoom_ax.set_title("③ Zoomed Transit  — U-shaped dip",
                      color=C_TICK, fontsize=9, pad=5, loc="left")
    fig.suptitle("BLS Planet Detection", color="white", fontsize=13,
                 fontweight="bold", y=0.96)
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.4rem'>
      <div style='font-family:Space Mono,monospace;font-size:1.05rem;
                  color:#00d4ff;font-weight:700;letter-spacing:-0.5px;
                  text-shadow:0 0 20px rgba(0,212,255,0.50);'>
        🔭 EXOPLANET<br>HUNTER
      </div>
      <div style='font-size:0.68rem;color:#2a4a6a;text-transform:uppercase;
                  letter-spacing:0.13em;margin-top:5px;'>
        Multi-Mission BLS Engine
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">TELESCOPE MISSION</div>',
                unsafe_allow_html=True)
    selected_mission = st.radio("mission_radio", ["Kepler", "TESS"],
                                label_visibility="collapsed")

    st.markdown('<div class="sidebar-label" style="margin-top:14px;">OBSERVATION WINDOW</div>',
                unsafe_allow_html=True)
    if selected_mission == "Kepler":
        time_segment = st.slider("quarter_slider", min_value=0, max_value=17,
                                 value=6, label_visibility="collapsed")
        st.markdown(
            f"<div style='font-size:0.67rem;color:#2a4870;margin-top:-4px;margin-bottom:6px;'>"
            f"Quarter: <span style='color:#00d4ff'>Q{time_segment}</span> · Range 0–17</div>",
            unsafe_allow_html=True)
    else:
        time_segment = st.slider("sector_slider", min_value=1, max_value=85,
                                 value=1, label_visibility="collapsed")
        st.markdown(
            f"<div style='font-size:0.67rem;color:#2a4870;margin-top:-4px;margin-bottom:6px;'>"
            f"Sector: <span style='color:#00d4ff'>S{time_segment}</span> · Range 1–85</div>",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.71rem;color:#2a4870;line-height:2.0;'>
      <b style='color:#3a5a8a'>Detection Pipeline</b><br>
      ① Download from MAST<br>
      ② NaN removal + normalise<br>
      ③ Savitzky-Golay flatten<br>
      ④ Outlier sigma-clip (4σ)<br>
      ⑤ BLS periodogram<br>
      ⑥ Phase-fold &amp; bin
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👨‍🚀 About the Developer")
    st.markdown("""
    **Name:** Kishore (Kai)

    **Academic Status:** Recently completed 12th Grade Examinations.

    **Goal:** 🎯 **MIT Class of 2031** — *Astrophysics & Aeronautical Engineering*

    **Portfolio:**
    - **ATS-1:** Asteroid Tracker System
    - **Exoplanet Hunter:** NASA MAST BLS engine
    """)
    st.info("Searching the cosmos for the next Earth-like world. 🌍")


# =============================================================================
# SESSION STATE
# =============================================================================
if "search_btn"  not in st.session_state: st.session_state.search_btn  = False
if "star_name"   not in st.session_state: st.session_state.star_name   = "Kepler-10"


# =============================================================================
# MAIN PAGE — LANDING
# =============================================================================
st.markdown('<div class="hero-title animate-in">EXOPLANET HUNTER</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub animate-in delay-1">'
            'Multi-Mission · Box Least Squares Detection Engine</div>',
            unsafe_allow_html=True)

if not st.session_state.search_btn:
    c1, c2, c3 = st.columns(3)
    for col, color, label, value in [
        (c1, "blue",  "MISSION",     f"{selected_mission} Space Telescope"),
        (c2, "green", "METHOD",      "Box Least Squares (BLS)"),
        (c3, "gold",  "DATA SOURCE", "NASA MAST Archive"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card {color} glass-in delay-2">
              <div class="stat-label">{label}</div>
              <div style='font-family:Space Mono,monospace;font-size:0.88rem;
                          color:#e8f4ff;margin-top:5px;'>{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.session_state.star_name = st.text_input(
        "star_input",
        value=st.session_state.star_name,
        placeholder="Enter star name  (e.g. Kepler-10, TRAPPIST-1) …",
        label_visibility="collapsed",
    )

    # Dynamic suggestions per mission
    if selected_mission == "Kepler":
        chips = ["Kepler-10", "Kepler-22", "Kepler-90", "Kepler-186"]
    else:
        chips = ["TRAPPIST-1", "TOI-700", "WASP-126", "HD 209458"]

    chip_html = " ".join(
        f'<code style="color:#00d4ff;background:rgba(0,212,255,0.09);'
        f'padding:2px 9px;border-radius:5px;border:1px solid rgba(0,212,255,0.22);">{c}</code>'
        for c in chips
    )
    st.markdown(f"""
    <div style="display:flex;gap:12px;justify-content:center;
                margin-top:-8px;margin-bottom:22px;flex-wrap:wrap;">
      <span style="font-size:0.78rem;color:#2a4870;align-self:center;">Suggestions:</span>
      {chip_html}
    </div>""", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button("🚀  INITIATE DISCOVERY SCAN",
                     use_container_width=True, type="primary"):
            st.session_state.search_btn = True
            st.rerun()
    st.stop()


# =============================================================================
# MAIN PAGE — RESULTS
# =============================================================================
time_label = "QUARTER" if selected_mission == "Kepler" else "SECTOR"
st.markdown(f"""
<div class="animate-in" style='font-family:Space Mono,monospace;
     font-size:0.78rem;color:#3a5a80;margin-bottom:1.2rem;'>
  ANALYSING &nbsp;<span style='color:#00d4ff'>{st.session_state.star_name.upper()}</span>
  &nbsp;·&nbsp; {selected_mission.upper()} {time_label} {time_segment}
</div>""", unsafe_allow_html=True)

# ── Fetch & clean ─────────────────────────────────────────────────────────────
with st.spinner("📡 Contacting NASA MAST archive …"):
    try:
        (raw_t, raw_f, raw_fe,
         trend_t, trend_f,
         flat_t, flat_f, flat_fe,
         clean_t, clean_f, clean_fe) = fetch_and_clean(
            st.session_state.star_name, selected_mission, time_segment)
    except Exception as e:
        st.error(
            f"**Download failed:** {e}\n\n"
            "**Common fixes:**\n"
            f"- Make sure you're using the right mission "
            f"(TRAPPIST-1 → TESS, Kepler-10 → Kepler)\n"
            f"- Try a different {time_label.lower()}\n"
            "- **SSL error?** Your ISP is blocking NASA — enable a VPN and retry"
        )
        st.session_state.search_btn = False
        st.stop()

# ── BLS ───────────────────────────────────────────────────────────────────────
with st.spinner(f"🔍 Running BLS — scanning {BLS_MIN_PERIOD}–{BLS_MAX_PERIOD} d …"):
    try:
        periods, power, best_period, best_t0, best_duration, best_depth = \
            run_bls_cached(clean_t, clean_f, clean_fe)
    except Exception as e:
        st.error(f"BLS failed: {e}")
        st.stop()

# ── Derived stats ─────────────────────────────────────────────────────────────
t_span     = clean_t.max() - clean_t.min()
n_transits = int(np.floor(t_span / best_period))
noise_ppm  = float(np.std(clean_f)) * 1e6
snr        = float(power.max()) / float(np.median(power)) if np.median(power) > 0 else 0.0

# ── Stat cards ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-1">★  Detected Planet Parameters</div>',
            unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)
for col, color, label, value, unit in [
    (c1, "blue",  "PERIOD",   f"{best_period:.5f}", "days"),
    (c2, "green", "DURATION", f"{best_duration*24:.3f}", "hours"),
    (c3, "gold",  "DEPTH",    f"{best_depth*1e6:.1f}", "ppm"),
    (c4, "pink",  "TRANSITS", f"~{n_transits}", "observed"),
    (c5, "cyan",  "BLS SNR",  f"{snr:.1f}", "×"),
]:
    with col:
        st.markdown(f"""
        <div class="stat-card {color} glass-in delay-2">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}<span class="stat-unit">{unit}</span></div>
        </div>""", unsafe_allow_html=True)

# ── Graph 1: Raw ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-2">01 · Raw Light Curve</div>',
            unsafe_allow_html=True)
st.markdown("""<div class='desc-text'>Raw stellar brightness over time.
The <span style='color:#ff6b6b'>red curve</span> is the Savitzky-Golay stellar trend —
slow variability that completely hides the tiny planet transits.</div>""",
            unsafe_allow_html=True)
fig_raw = plot_raw(raw_t, raw_f, raw_fe, trend_t, trend_f)
st.pyplot(fig_raw, use_container_width=True); plt.close(fig_raw)

# ── Graph 2: Flat ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-2">02 · Cleaned &amp; Flattened Light Curve</div>',
            unsafe_allow_html=True)
st.markdown(f"""<div class='desc-text'>Top: stellar trend removed.
Bottom: outlier spikes clipped. Noise floor ≈
<b style='color:#a8ff78'>{noise_ppm:.0f} ppm</b> — BLS search input.</div>""",
            unsafe_allow_html=True)
fig_flat = plot_flat(flat_t, flat_f, clean_t, clean_f)
st.pyplot(fig_flat, use_container_width=True); plt.close(fig_flat)

# ── Graph 3: BLS ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-3">03 · BLS Periodogram &amp; Phase-Folded Transit</div>',
            unsafe_allow_html=True)
st.markdown(f"""<div class='desc-text'>BLS tested every period {BLS_MIN_PERIOD}–{BLS_MAX_PERIOD} d.
Tallest spike = orbital period
(<b style='color:#ffe66d'>P = {best_period:.5f} d</b>).
All ~{n_transits} transits stacked — the U-shaped dip is the exoplanet's shadow.</div>""",
            unsafe_allow_html=True)
with st.spinner("Rendering BLS detection plot …"):
    fig_bls = plot_bls(periods, power, clean_t, clean_f, clean_fe,
                       best_period, best_t0, best_duration, best_depth)
st.pyplot(fig_bls, use_container_width=True); plt.close(fig_bls)

# ── Back button ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, back_col, _ = st.columns([1, 2, 1])
with back_col:
    if st.button("🔄  Search Another Star", use_container_width=True):
        st.session_state.search_btn = False
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;font-family:Space Mono,monospace;
     font-size:0.68rem;color:#2e4870;padding:1rem 0 2rem;'>
  {st.session_state.star_name.upper()} ·
  {selected_mission} {time_label[0]}{time_segment} ·
  {len(clean_t):,} cadences · {t_span:.1f} d ·
  Noise {noise_ppm:.0f} ppm ·
  P = {best_period:.5f} d · Depth {best_depth*1e6:.0f} ppm ·
  NASA MAST Archive
</div>""", unsafe_allow_html=True)