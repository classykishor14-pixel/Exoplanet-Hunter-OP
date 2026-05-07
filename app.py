"""
==============================================================================
  EXOPLANET DETECTION SYSTEM — Streamlit Web Application
  app.py  [UPGRADED: Neon Space Mono + Ken Burns Zoom + Glassmorphism]
==============================================================================
HOW TO RUN
    pip install streamlit lightkurve astropy matplotlib numpy
    streamlit run app.py
==============================================================================

NEW FEATURES:
  • Ken Burns cosmic drift animation
  • Neon Space Mono terminal aesthetic
  • Glowing cyan/teal text effects
  • CRT scanline overlay (optional)
"""

import streamlit as st
import base64

# set_page_config MUST be the absolute first Streamlit call in the file
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="Exohunt.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- NEW BACKGROUND INJECTION CODE ---
import os
import base64

def set_bg_image():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bg_path = os.path.join(current_dir, "BG.png")
    
    with open(bg_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        .cosmic-drift {{
            background-color: #02030a !important;
            background-image: url("data:image/png;base64,{encoded_string}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()
# -------------------------------------

import warnings
import shutil
import requests
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

# =============================================================================
# SESSION STATE — must be initialised before any widget reads it
# =============================================================================
if "search_btn"           not in st.session_state: st.session_state.search_btn           = False
if "star_name"            not in st.session_state: st.session_state.star_name            = "Kepler-10"
if "star_radius_solar"    not in st.session_state: st.session_state.star_radius_solar    = 1.0
if "planet_mass_earth"    not in st.session_state: st.session_state.planet_mass_earth    = 1.0
if "star_luminosity_solar" not in st.session_state: st.session_state.star_luminosity_solar = 1.0
if "semi_major_axis_au"   not in st.session_state: st.session_state.semi_major_axis_au   = 1.0
if "nasa_synced_planet"   not in st.session_state: st.session_state.nasa_synced_planet   = ""
if "nasa_sync_status"     not in st.session_state: st.session_state.nasa_sync_status     = None
if "nasa_sync_fields"     not in st.session_state: st.session_state.nasa_sync_fields     = []

# =============================================================================
# NASA EXOPLANET ARCHIVE — TAP Service Integration
# =============================================================================

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_nasa_exoplanet_data(planet_name: str) -> dict | None:
    """
    Bulletproof query against the NASA Exoplanet Archive TAP service.

    Why the old approach failed
    ---------------------------
    The TAP 'ps' table stores one row *per publication*, so a single planet
    like "Kepler-442 b" may have dozens of rows, many with NULL radii.
    Sorting by a nullable column (pl_rade DESC) caused NULL rows to bubble up
    in some TAP implementations, and an exact-string match could silently fail
    if the archive value has a trailing space or a variant hyphen encoding.

    Resolution cascade (stops at first non-empty result set)
    ---------------------------------------------------------
    Stage A — pscomppars table (one best-row per planet, rarely NULL)
      A1. exact pl_name match        "Kepler-442 b"
      A2. exact hostname match       "Kepler-442"
      A3. pl_name LIKE prefix        "Kepler-442%"
      A4. hostname LIKE prefix       "Kepler-442%"

    Stage B — ps table (all publications, more rows, more NULLs)
      B1–B4: same four patterns as Stage A

    Stage C — normalised-name variants fed through A1 + B1
      Handles: stripped suffix ("Kepler-442"), hyphen-vs-space ambiguity,
      letter-suffix with/without space ("Kepler-442b" vs "Kepler-442 b")

    Within each result-set the row with the highest count of non-NULL
    science fields is returned (no reliance on ORDER BY a nullable column).

    Returns
    -------
    dict  with keys pl_name, pl_rade, pl_masse, st_rad, st_lum, pl_orbsmax
    None  if all 12+ strategies are exhausted or a network error occurs.
    """

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _esc(s: str) -> str:
        """Escape single-quotes for ADQL string literals."""
        return s.replace("'", "''")

    def _score(r: dict) -> int:
        """Count non-NULL science fields — used to pick the best row."""
        return sum(1 for k in ("pl_rade", "pl_masse", "st_rad", "st_lum", "pl_orbsmax")
                   if r.get(k) is not None)

    def _parse(row: dict) -> dict:
        lum_log = row.get("st_lum")
        return {
            "pl_name"    : row.get("pl_name"),
            "pl_rade"    : float(row["pl_rade"])    if row.get("pl_rade")    is not None else None,
            "pl_masse"   : float(row["pl_masse"])   if row.get("pl_masse")   is not None else None,
            "st_rad"     : float(row["st_rad"])     if row.get("st_rad")     is not None else None,
            "st_lum"     : (10 ** float(lum_log))   if lum_log              is not None else None,
            "pl_orbsmax" : float(row["pl_orbsmax"]) if row.get("pl_orbsmax") is not None else None,
        }

    COLS = "pl_name,pl_rade,pl_masse,st_rad,st_lum,pl_orbsmax"

    def _run(adql: str):
        """Fire one ADQL query; return list of rows or []."""
        try:
            r = requests.get(NASA_TAP_URL,
                             params={"query": adql, "format": "json"},
                             timeout=15)
            r.raise_for_status()
            return r.json() or []
        except Exception:
            return []

    def _queries_for(token: str, table: str) -> list:
        """
        Build the four ADQL patterns for a given search token and table name.
        No ORDER BY on nullable columns — sorting is done in Python.
        """
        t = _esc(token)
        return [
            f"SELECT {COLS} FROM {table} WHERE LOWER(pl_name)=LOWER('{t}')",
            f"SELECT {COLS} FROM {table} WHERE LOWER(hostname)=LOWER('{t}')",
            f"SELECT {COLS} FROM {table} WHERE LOWER(pl_name) LIKE LOWER('{t}%')",
            f"SELECT {COLS} FROM {table} WHERE LOWER(hostname) LIKE LOWER('{t}%')",
        ]

    # ── Build normalised search tokens ────────────────────────────────────────
    raw   = planet_name.strip()
    tokens = [raw]                          # always try the input as-is first

    # If it ends with a space + single letter (e.g. "Kepler-442 b"),
    # also try: the host part alone, and the suffix-less compact form.
    import re as _re
    m = _re.match(r"^(.+?)\s+([a-zA-Z])$", raw)
    if m:
        host, letter = m.group(1), m.group(2)
        tokens.append(host)                             # "Kepler-442"
        tokens.append(host + letter)                   # "Kepler-442b"  (no space)
        tokens.append(host + " " + letter.lower())     # normalised lower suffix
        tokens.append(host + " " + letter.upper())
    else:
        # Input has no letter suffix — try appending a wildcard host strip
        # in case the user typed something like "Kepler-442b" (no space)
        m2 = _re.match(r"^(.+?)([a-zA-Z])$", raw)
        if m2 and not raw[-2].isdigit() is False:
            tokens.append(m2.group(1))                 # strip trailing letter
            tokens.append(m2.group(1) + " " + m2.group(2))  # add space before letter

    # De-duplicate while preserving order
    seen, unique_tokens = set(), []
    for tk in tokens:
        if tk.lower() not in seen:
            seen.add(tk.lower())
            unique_tokens.append(tk)

    # ── Run cascade: pscomppars first (best data density), then ps ───────────
    # pscomppars = one aggregated row per planet (NASA's recommended table)
    # ps         = one row per reference publication (more rows, more NULLs)
    for token in unique_tokens:
        for table in ("pscomppars", "ps"):
            for adql in _queries_for(token, table):
                rows = _run(adql)
                if rows:
                    return _parse(max(rows, key=_score))

    return None   # every strategy exhausted


# =============================================================================
# NASA PLANET NAME AUTOCOMPLETE
# =============================================================================

@st.cache_data(show_spinner=False, ttl=86400)   # refresh once per day
def fetch_all_planet_names() -> list[str]:
    """
    Download every confirmed pl_name AND hostname from the NASA Exoplanet
    Archive ps table and merge them into a single de-duplicated sorted list.

    Having hostnames in the list means the autocomplete dropdown can surface
    "Kepler-307" even though the archive only stores "Kepler-307 b", "Kepler-307 c" etc.
    Falls back to an empty list on any network error.
    """
    adql = "SELECT pl_name, hostname FROM ps WHERE pl_name IS NOT NULL"
    try:
        resp = requests.get(
            NASA_TAP_URL,
            params={"query": adql, "format": "json"},
            timeout=20,
        )
        resp.raise_for_status()
        rows   = resp.json()
        names  = set()
        for r in rows:
            if r.get("pl_name"):
                names.add(r["pl_name"].strip())
            if r.get("hostname"):
                names.add(r["hostname"].strip())
        return sorted(names)
    except Exception:
        return []


def search_planets(query: str) -> list[str]:
    """
    Case-insensitive substring filter over all NASA planet names + host-star names.
    Returns up to 15 suggestions, planet designations first, then host names.
    """
    q = query.strip().lower()
    if not q:
        return []
    all_names = fetch_all_planet_names()
    # Separate exact planet designations (end with a space + letter) from hostnames
    planets   = [n for n in all_names if q in n.lower() and len(n) > 2 and n[-2] == " "]
    hosts     = [n for n in all_names if q in n.lower() and not (len(n) > 2 and n[-2] == " ")]
    combined  = planets[:10] + hosts[:5]
    return combined[:15]

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
# ░░░  MASTER CSS — NEON SPACE MONO + KEN BURNS + GLASSMORPHISM  ░░░
# =============================================================================
st.markdown("""
<style>
/* ── Google Fonts (Space Mono for terminal feel) ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Share+Tech+Mono&display=swap');

/* ════════════════════════════════════════════════════════════════════════════
   KEN BURNS DRIFT EFFECT - Cosmic Animation
════════════════════════════════════════════════════════════════════════════ */

@keyframes cosmicZoom {
    0% {
        transform: scale(1.00);
    }
    50% {
        transform: scale(1.12);
    }
    100% {
        transform: scale(1.00);
    }
}

@keyframes starDrift {
    0% {
        transform: translate(0%, 0%) scale(1);
        opacity: 0.5;
    }
    50% {
        transform: translate(-2%, -1.5%) scale(1.08);
        opacity: 0.9;
    }
    100% {
        transform: translate(0%, 0%) scale(1);
        opacity: 0.5;
    }
}

@keyframes scanline {
    0% {
        transform: translateY(-100%);
    }
    100% {
        transform: translateY(100%);
    }
}

.cosmic-drift {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -2;
    pointer-events: none;
    animation: cosmicZoom 28s ease-in-out infinite alternate;
    will-change: transform;
    background-color: #02030a;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}

.star-drift {
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    animation: starDrift 35s ease-in-out infinite alternate;
}

.star-drift::after {
    content: "";
    position: absolute;
    inset: 0;
    /* Dark overlay so satellite images show through but remain readable behind the UI */
    background: linear-gradient(
        180deg,
        rgba(2, 4, 14, 0.60) 0%,
        rgba(2, 4, 14, 0.55) 33.33%,
        rgba(2, 4, 14, 0.58) 66.66%,
        rgba(2, 4, 14, 0.60) 100%
    );
    background-size: 100% 100%;
    background-repeat: no-repeat;
}

/* Optional CRT scanline effect (uncomment to enable) */
/* 
.scanline {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        to bottom,
        transparent 50%,
        rgba(0, 0, 0, 0.3) 50%
    );
    background-size: 100% 4px;
    pointer-events: none;
    z-index: 999;
    animation: scanline 8s linear infinite;
}
*/

/* ════════════════════════════════════════════════════════════════════════════
   NEON SPACE MONO - HERO TITLE & SUB-HEADERS
   Futuristic NASA terminal aesthetic with glow effects
════════════════════════════════════════════════════════════════════════════ */

/* Main Hero Title - Neon Space Mono */
.hero-title {
    font-family: 'Space Mono', 'Share Tech Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -1px;
    line-height: 1.05;
    margin-bottom: 0.28rem;
    text-align: center;
    
    /* Neon cyan core */
    color: #00ffff;
    text-shadow: 
        0 0 5px rgba(0, 255, 255, 0.3),
        0 0 10px rgba(0, 255, 255, 0.5),
        0 0 20px rgba(0, 255, 255, 0.7),
        0 0 40px rgba(0, 100, 255, 0.5),
        0 0 80px rgba(0, 150, 255, 0.3);
    
    animation: neonPulse 3s ease-in-out infinite;
}

@keyframes neonPulse {
    0%, 100% {
        text-shadow: 
            0 0 5px rgba(0, 255, 255, 0.3),
            0 0 10px rgba(0, 255, 255, 0.5),
            0 0 20px rgba(0, 255, 255, 0.7),
            0 0 40px rgba(0, 100, 255, 0.5);
        opacity: 1;
    }
    50% {
        text-shadow: 
            0 0 10px rgba(0, 255, 255, 0.5),
            0 0 20px rgba(0, 255, 255, 0.7),
            0 0 30px rgba(0, 255, 255, 0.9),
            0 0 60px rgba(0, 100, 255, 0.7),
            0 0 100px rgba(0, 150, 255, 0.5);
        opacity: 0.95;
    }
}

/* Hero Subtitle - Glowing Teal */
.hero-sub {
    font-family: 'Space Mono', 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 2rem;
    
    color: #00ccaa;
    text-shadow: 
        0 0 3px rgba(0, 204, 170, 0.5),
        0 0 8px rgba(0, 204, 170, 0.3);
    
    animation: subPulse 4s ease-in-out infinite;
}

@keyframes subPulse {
    0%, 100% {
        text-shadow: 0 0 3px rgba(0, 204, 170, 0.5), 0 0 8px rgba(0, 204, 170, 0.3);
        letter-spacing: 0.2em;
    }
    50% {
        text-shadow: 0 0 6px rgba(0, 204, 170, 0.8), 0 0 12px rgba(0, 204, 170, 0.5);
        letter-spacing: 0.22em;
    }
}

/* Section Headers - Glowing Cyan Terminal Style */
.section-header {
    font-family: 'Space Mono', 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #00ddff;
    border-bottom: 1px solid rgba(0, 212, 255, 0.42);
    padding-bottom: 8px;
    margin: 2.4rem 0 1.1rem 0;
    
    text-shadow: 
        0 0 4px rgba(0, 221, 255, 0.4),
        0 0 8px rgba(0, 221, 255, 0.2);
    
    position: relative;
    overflow: hidden;
}

/* Terminal cursor effect for section headers */
.section-header::before {
    content: ">";
    position: absolute;
    left: -20px;
    color: #00ffff;
    font-weight: bold;
    animation: blink 1s step-end infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

/* Sidebar Header - Neon Pulse */
.sidebar-header {
    font-family: 'Space Mono', 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00ccff;
    text-shadow: 0 0 5px rgba(0, 204, 255, 0.5);
}

/* Description text - Terminal style */
.desc-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #90b4d5;
    line-height: 1.68;
    margin-bottom: 0.95rem;
    text-shadow: 0 1px 10px rgba(0, 0, 0, 0.80);
    border-left: 2px solid rgba(0, 212, 255, 0.3);
    padding-left: 14px;
}

/* Code blocks - Terminal green */
code {
    font-family: 'Space Mono', monospace;
    background: rgba(0, 212, 255, 0.1);
    color: #00ff88;
    text-shadow: 0 0 3px rgba(0, 255, 136, 0.5);
}

/* Sidebar labels - Terminal style */
.sidebar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00ccff;
    margin-bottom: 0.3rem;
    text-shadow: 0 0 3px rgba(0, 204, 255, 0.3);
}

/* Stat card labels */
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.60rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00ccaa;
    margin-bottom: 6px;
    text-shadow: 0 0 3px rgba(0, 204, 170, 0.3);
}

/* Status text */
.status-text {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #00ffcc;
    text-shadow: 0 0 4px rgba(0, 255, 204, 0.5);
}

/* ════════════════════════════════════════════════════════════════════════════
   BASE CONTAINERS
   Strategy: give the app a solid deep-space background on every known
   Streamlit testid so the cosmic-drift overlay sits on top of it cleanly.
   We do NOT use backdrop-filter on the root containers — blurring a
   transparent element against a white/grey Streamlit default produces the
   "frosted blur" blank-page artefact seen in some deployments.
════════════════════════════════════════════════════════════════════════════ */

/* Deep-space solid base — covers ALL Streamlit shell layers */
html, body {
    background: #02030a !important;
    background-color: #02030a !important;
}

/* Cover every testid variant across Streamlit 1.28–1.45 */
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stDecoration"],
[data-testid="stToolbar"],
section[data-testid="stSidebar"] ~ div,
.main {
    background: transparent !important;
    background-color: transparent !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   MAIN CONTENT PANEL
   Use a solid semi-transparent dark background — NO backdrop-filter blur.
   Blur on a container that sits over an animated fixed background causes
   the "frosted blank" rendering bug on cloud deployments.
════════════════════════════════════════════════════════════════════════════ */
.main .block-container,
[data-testid="stMainBlockContainer"] {
    background: rgba(3, 7, 22, 0.72) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border-radius: 20px !important;
    padding: 2rem 2.5rem !important;
    margin-top: 0.6rem !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    box-shadow:
        0 12px 48px rgba(0, 0, 0, 0.65),
        inset 0  1px 0 rgba(255, 255, 255, 0.06),
        inset 0 -1px 0 rgba(0, 212, 255, 0.08) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — SIDEBAR
════════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: rgba(3, 7, 22, 0.92) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border-right: 1px solid rgba(0, 212, 255, 0.18) !important;
    box-shadow:
        4px 0 36px rgba(0, 0, 0, 0.70),
        inset -1px 0 0 rgba(0, 212, 255, 0.10) !important;
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
    background: rgba(2, 6, 18, 0.82) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border-radius: 16px !important;
    border: 1px solid rgba(0, 212, 255, 0.20) !important;
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
    background: rgba(5, 12, 30, 0.90) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: 1px solid rgba(0, 212, 255, 0.22) !important;
    border-radius: 12px !important;
}

div[data-testid="stSpinner"] > div {
    background: rgba(3, 9, 24, 0.90) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0, 212, 255, 0.18) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — STAT CARDS
════════════════════════════════════════════════════════════════════════════ */
.stat-card {
    background:              rgba(5, 12, 32, 0.88);
    backdrop-filter:         none;
    -webkit-backdrop-filter: none;
    border: 1px solid rgba(0, 212, 255, 0.24);
    border-radius: 14px;
    padding: 15px 22px;
    flex: 1; min-width: 140px;
    position: relative; overflow: hidden;
    box-shadow:
        0 6px 30px rgba(0, 0, 0, 0.55),
        inset 0  1px 0 rgba(255, 255, 255, 0.09),
        inset 0 -1px 0 rgba(0, 212, 255, 0.15);
    transition: transform 0.22s ease, box-shadow 0.28s ease, border-color 0.28s ease;
}
.stat-card:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow:
        0 12px 42px rgba(0, 0, 0, 0.65),
        0  0   26px rgba(0, 160, 255, 0.14),
        inset 0 1px 0 rgba(255, 255, 255, 0.11);
    border-color: rgba(0, 212, 255, 0.55);
}

/* Coloured accent bar along the top edge of each card */
.stat-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: 14px 14px 0 0;
}

.stat-card.blue::before  { background: linear-gradient(90deg, #00d4ff, #0070ff); }
.stat-card.green::before { background: linear-gradient(90deg, #a8ff78, #00d460); }
.stat-card.gold::before  { background: linear-gradient(90deg, #ffe66d, #ff9800); }
.stat-card.pink::before  { background: linear-gradient(90deg, #ff4f6e, #ff0055); }
.stat-card.cyan::before  { background: linear-gradient(90deg, #4a7cff, #00d4ff); }

.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.24rem;
    font-weight: 700;
    color: #eaf4ff;
    text-shadow: 0 0 16px rgba(0, 180, 255, 0.24);
}

.stat-unit { font-size: 0.67rem; color: #00ccaa; margin-left: 3px; }

/* ════════════════════════════════════════════════════════════════════════════
   TYPOGRAPHY
════════════════════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Space Mono', 'Share Tech Mono', monospace;
    color: #ccdff5;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — WIDGET OVERRIDES
════════════════════════════════════════════════════════════════════════════ */
.stTextInput > div > div > input {
    background:              rgba(4, 11, 30, 0.95) !important;
    backdrop-filter:         none !important;
    -webkit-backdrop-filter: none !important;
    border: 1px solid rgba(0, 212, 255, 0.55) !important;
    color: #c8d8f0 !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 9px !important;
    box-shadow:
        inset 0 2px 12px rgba(0, 0, 0, 0.36),
        inset 0 1px 0   rgba(255, 255, 255, 0.05) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00ffff !important;
    box-shadow:
        0 0 0 2px rgba(0, 212, 255, 0.22),
        inset 0 2px 10px rgba(0, 0, 0, 0.30) !important;
}
.stTextInput > div > div > input::placeholder { 
    font-family: 'Space Mono', monospace;
    color: #2e4a70 !important; 
}

/* Primary action button - Terminal style */
.stButton > button[kind="primary"] {
    background:              rgba(0, 212, 255, 0.10) !important;
    backdrop-filter:         none !important;
    -webkit-backdrop-filter: none !important;
    border: 1px solid rgba(0, 212, 255, 0.44) !important;
    color: #00ffff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    border-radius: 11px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.07) !important;
    transition: all 0.24s ease !important;
    text-transform: uppercase;
}
.stButton > button[kind="primary"]:hover {
    background:  rgba(0, 212, 255, 0.17) !important;
    border-color: rgba(0, 212, 255, 0.70) !important;
    box-shadow:
        0 0 32px rgba(0, 212, 255, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
    transform: translateY(-1px) !important;
    text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
}

/* Secondary (default) buttons */
.stButton > button:not([kind="primary"]) {
    background:              rgba(8, 20, 50, 0.90) !important;
    backdrop-filter:         none !important;
    -webkit-backdrop-filter: none !important;
    border: 1px solid rgba(0, 212, 255, 0.40) !important;
    color: #00ccaa !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 9px !important;
    transition: all 0.22s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    background:  rgba(10, 28, 65, 0.72) !important;
    border-color: rgba(0, 212, 255, 0.58) !important;
    color: #00ffff !important;
    text-shadow: 0 0 3px rgba(0, 255, 255, 0.3);
}

/* Slider track glass pill */
.stSlider > div > div > div {
    background: rgba(5, 14, 38, 0.68) !important;
    border-radius: 8px !important;
}

/* Radio buttons - Terminal style */
div.row-widget.stRadio > div {
    flex-direction: row; gap: 15px;
    justify-content: center; margin-bottom: 10px;
}
div.row-widget.stRadio > div > label {
    background: rgba(5, 12, 32, 0.92) !important;
    backdrop-filter: none !important;
    border: 1px solid rgba(0, 212, 255, 0.32) !important;
    border-radius: 8px !important;
    padding: 4px 14px !important;
    color: #00ccaa !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    transition: all 0.2s ease !important;
}
div.row-widget.stRadio > div > label:hover {
    border-color: rgba(0, 212, 255, 0.65) !important;
    color: #00ffff !important;
    text-shadow: 0 0 3px rgba(0, 255, 255, 0.3);
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

/* Styled thin scrollbar - Terminal cyan */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: rgba(2, 5, 16, 0.55); }
::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.36);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(0, 212, 255, 0.65); }

/* Sidebar internal divider lines */
section[data-testid="stSidebar"] hr {
    border-color: rgba(0, 212, 255, 0.30) !important;
}

/* Success/Warning/Error text overrides */
div.element-container div.stAlert {
    font-family: 'Space Mono', monospace;
}

/* ════════════════════════════════════════════════════════════════════════════
   PLANET PROFILE CARD — full-width hero panel
════════════════════════════════════════════════════════════════════════════ */

@keyframes profileGlow {
    0%, 100% { box-shadow: 0 0 40px rgba(0,212,255,0.10), 0 8px 48px rgba(0,0,0,0.65),
               inset 0 1px 0 rgba(255,255,255,0.06); }
    50%       { box-shadow: 0 0 70px rgba(0,212,255,0.18), 0 8px 48px rgba(0,0,0,0.65),
               inset 0 1px 0 rgba(255,255,255,0.08); }
}

@keyframes orbitSpin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

@keyframes planetPulse {
    0%, 100% { transform: scale(1.00); filter: brightness(1.0); }
    50%       { transform: scale(1.04); filter: brightness(1.15); }
}

@keyframes ringPulse {
    0%, 100% { opacity: 0.35; transform: scale(1.00); }
    50%       { opacity: 0.55; transform: scale(1.06); }
}

@keyframes tickerScroll {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

.planet-profile-card {
    background: linear-gradient(135deg,
        rgba(3, 8, 24, 0.82) 0%,
        rgba(5, 14, 38, 0.78) 50%,
        rgba(3, 8, 24, 0.84) 100%);
    backdrop-filter:         blur(28px) saturate(180%) brightness(0.92);
    -webkit-backdrop-filter: blur(28px) saturate(180%) brightness(0.92);
    border: 1px solid rgba(0, 212, 255, 0.28);
    border-radius: 22px;
    padding: 32px 36px 28px;
    margin: 1.4rem 0 1.6rem;
    position: relative;
    overflow: hidden;
    animation: profileGlow 5s ease-in-out infinite;
}

/* Scanline texture overlay */
.planet-profile-card::before {
    content: "";
    position: absolute; inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 3px,
        rgba(0, 212, 255, 0.012) 3px,
        rgba(0, 212, 255, 0.012) 4px
    );
    border-radius: 22px;
    pointer-events: none;
    z-index: 0;
}

/* Corner accent brackets */
.planet-profile-card::after {
    content: "";
    position: absolute; top: 12px; left: 12px;
    width: 28px; height: 28px;
    border-top: 2px solid rgba(0, 212, 255, 0.55);
    border-left: 2px solid rgba(0, 212, 255, 0.55);
    border-radius: 4px 0 0 0;
    pointer-events: none;
}

.ppc-content { position: relative; z-index: 1; }

/* ── Planet orb (SVG-based, rendered inline) ── */
.ppc-orb-wrap {
    display: flex; align-items: center; justify-content: center;
    position: relative; width: 120px; height: 120px; margin: 0 auto;
}

/* ── Column icon panels ── */
.ppc-icon-panel {
    background: rgba(5, 14, 38, 0.60);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 16px;
    padding: 22px 20px 18px;
    text-align: center;
    transition: transform 0.22s ease, border-color 0.28s ease, box-shadow 0.28s ease;
    height: 100%;
}
.ppc-icon-panel:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 212, 255, 0.45);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.50), 0 0 20px rgba(0,212,255,0.10);
}

.ppc-icon {
    font-size: 2.6rem;
    line-height: 1;
    margin-bottom: 10px;
    display: block;
    filter: drop-shadow(0 0 8px currentColor);
}

.ppc-panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #00ccaa;
    margin-bottom: 8px;
}

.ppc-panel-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.25;
    margin-bottom: 6px;
}

.ppc-panel-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.70rem;
    color: #7090b0;
    line-height: 1.5;
}

/* ── Radial score ring ── */
.ppc-score-ring {
    position: relative;
    width: 108px; height: 108px;
    margin: 0 auto 10px;
}
.ppc-score-ring svg { width: 108px; height: 108px; overflow: visible; }

/* ── Ticker tape at bottom ── */
.ppc-ticker {
    margin-top: 22px;
    border-top: 1px solid rgba(0,212,255,0.16);
    padding-top: 10px;
    overflow: hidden;
    white-space: nowrap;
}
.ppc-ticker-inner {
    display: inline-block;
    animation: tickerScroll 22s linear infinite;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #004466;
    letter-spacing: 0.15em;
}
.ppc-ticker-inner span { color: #00aabb; margin: 0 3px; }
</style>

<div class="cosmic-drift"></div>
<div class="star-drift"></div>
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
        sp.set_edgecolor("#00aacc"); sp.set_linewidth(0.8)
    ax.tick_params(colors=C_TICK, labelsize=8.5)
    ax.grid(True, linestyle=":", linewidth=0.35, color="#1a3a55", alpha=0.9)
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


# =============================================================================
# PLANET COMPOSITION CLASSIFIER  (NASA Exoplanet Archive logic)
# =============================================================================

def get_planet_composition(planet_mass: float, planet_radius: float) -> dict:
    """
    Classify an exoplanet's composition using NASA Exoplanet Archive-style
    mass–radius–density thresholds.

    Parameters
    ----------
    planet_mass   : float  — planet mass in Earth masses  (M⊕)
    planet_radius : float  — planet radius in Earth radii (R⊕)

    Returns
    -------
    dict with keys:
        category    : str   — one of 'Gas Giant', 'Super-Earth', 'Rocky', 'Lava World'
        density_gcc : float — mean bulk density in g/cm³
        density_rel : float — density relative to Earth (ρ⊕ = 5.51 g/cm³)
        emoji       : str   — visual indicator
        description : str   — brief science note
        color       : str   — hex colour for UI theming
    """
    # Physical constants
    EARTH_MASS_G   = 5.972e27          # grams
    EARTH_RADIUS_CM = 6.371e8          # centimetres
    PI              = 3.141592653589793
    EARTH_DENSITY   = 5.51             # g/cm³  (reference)

    # Convert to CGS
    mass_g   = planet_mass   * EARTH_MASS_G
    radius_cm = planet_radius * EARTH_RADIUS_CM

    # Volume of a sphere (cm³)
    volume_cm3 = (4.0 / 3.0) * PI * radius_cm ** 3

    # Bulk density in g/cm³
    density_gcc = mass_g / volume_cm3
    density_rel = density_gcc / EARTH_DENSITY

    # ── Classification thresholds ──────────────────────────────────────────
    # Based on Rogers (2015), Fortney et al. (2007), and the NASA Exoplanet
    # Archive classification scheme:
    #
    #  Gas Giant   : radius > 4 R⊕  OR  density < 0.5 g/cm³
    #                (H/He-dominated; Jupiter ≈ 1.33 g/cm³, Saturn ≈ 0.69)
    #  Super-Earth : 1.6 ≤ radius ≤ 4 R⊕  AND  0.5 ≤ density < 3.5 g/cm³
    #                (volatile/water envelope + rocky core)
    #  Lava World  : radius < 1.6 R⊕  AND  density ≥ 7.5 g/cm³
    #                (ultra-dense; likely desiccated iron-rich remnant or
    #                 magma ocean — e.g. 55 Cnc e, Kepler-10b)
    #  Rocky       : radius < 1.6 R⊕  AND  3.5 ≤ density < 7.5 g/cm³
    #                (silicate-dominated; Earth/Venus analogue)

    if planet_radius > 4.0 or density_gcc < 0.5:
        category    = "Gas Giant"
        emoji       = "🪐"
        description = (
            "H/He gas-dominated envelope. No solid surface. "
            "Similar to Jupiter or Saturn in our Solar System."
        )
        color = "#4a7cff"

    elif density_gcc >= 7.5 and planet_radius < 1.6:
        category    = "Lava World"
        emoji       = "🌋"
        description = (
            "Extreme density suggests an iron-rich or desiccated interior, "
            "possibly hosting a global magma ocean (e.g. Kepler-10b, 55 Cnc e)."
        )
        color = "#ff4f2e"

    elif planet_radius < 1.6 and density_gcc >= 3.5:
        category    = "Rocky"
        emoji       = "🪨"
        description = (
            "Silicate/iron-dominated composition, similar to Earth or Mars. "
            "May host a thin atmosphere or bare rock surface."
        )
        color = "#a8ff78"

    else:
        # Catch-all: sub-Neptune / Super-Earth regime
        category    = "Super-Earth"
        emoji       = "🌍"
        description = (
            "Intermediate between rocky and gas-rich. Likely a rocky core "
            "wrapped in a thick volatile or water-vapour envelope."
        )
        color = "#00d4ff"

    return {
        "category"   : category,
        "density_gcc": round(density_gcc, 3),
        "density_rel": round(density_rel, 3),
        "emoji"      : emoji,
        "description": description,
        "color"      : color,
    }


def estimate_planet_radius_earth(transit_depth: float,
                                  star_radius_solar: float = 1.0) -> float:
    """
    Estimate planet radius in Earth radii from transit depth and stellar radius.

    transit_depth      : dimensionless flux drop  (e.g. 0.01 = 1 %)
    star_radius_solar  : host-star radius in solar radii (default 1.0)
    Returns: planet radius in Earth radii
    """
    SOLAR_TO_EARTH_RADII = 109.076   # 1 R☉ = 109.076 R⊕
    rp_over_rs = float(np.sqrt(max(transit_depth, 0.0)))
    return rp_over_rs * star_radius_solar * SOLAR_TO_EARTH_RADII


# =============================================================================
# ATMOSPHERE POTENTIAL CLASSIFIER
# =============================================================================

def get_atmosphere_potential(
    planet_radius_earth: float,
    planet_mass_earth:   float,
    density_gcc:         float,
    hz_zone_label:       str,
    flux_ratio:          float,
) -> dict:
    """
    Estimate the likelihood and type of a retained atmosphere.

    Logic is based on:
      - Escape velocity proxy (mass/radius²)
      - Stellar irradiation (photoevaporation risk via flux_ratio)
      - Planet size relative to the "radius gap" (Fulton gap ~1.5–2.0 R⊕)
      - HZ zone position

    Returns dict with: label, emoji, color, likelihood_pct, note
    """
    # Escape velocity proxy:  v_esc² ∝ M / R  (in Earth units)
    v_esc_sq = planet_mass_earth / max(planet_radius_earth, 0.01)

    # Photoevaporation risk:  high flux → atmosphere stripped
    # Flux > 40 S⊕  (hot rocky zone) → very high stripping risk
    photo_risk = min(1.0, float(flux_ratio) / 40.0)

    # Base retention score from escape velocity (0–1)
    if v_esc_sq >= 25.0:         # Jupiter-class
        base = 0.97
    elif v_esc_sq >= 6.0:        # Sub-Neptune / big Super-Earth
        base = 0.82
    elif v_esc_sq >= 1.5:        # Earth-like
        base = 0.62
    elif v_esc_sq >= 0.4:        # Mars-like
        base = 0.30
    else:
        base = 0.08              # Mercury-like

    # Irradiation penalty
    retained = base * (1.0 - 0.72 * photo_risk)

    # Radius gap adjustment: planets 1.5–2.0 R⊕ are transitional
    if 1.5 <= planet_radius_earth <= 2.0:
        retained *= 0.80   # uncertain regime

    likelihood_pct = int(np.clip(retained * 100, 2, 98))

    # ── Classify ────────────────────────────────────────────────────────────
    hz_in = hz_zone_label in ("Green Zone", "Cool Edge", "Hot Edge")

    if planet_radius_earth > 4.0:
        label   = "Thick H/He Envelope"
        emoji   = "🌫️"
        color   = "#4a7cff"
        note    = "Gas-dominated — no solid surface to walk on."
    elif planet_radius_earth > 1.8 and density_gcc < 3.0:
        label   = "Volatile-Rich Envelope"
        emoji   = "💨"
        color   = "#00d4ff"
        note    = "Water vapour / H₂ envelope likely; sub-Neptune class."
    elif likelihood_pct >= 65 and hz_in:
        label   = "Earth-like Atmosphere"
        emoji   = "🌬️"
        color   = "#00ff88"
        note    = "Mass & flux suggest nitrogen/oxygen or CO₂ atmosphere could persist."
    elif likelihood_pct >= 40:
        label   = "Thin Atmosphere"
        emoji   = "🌀"
        color   = "#ffe66d"
        note    = "Moderate retention; Mars or Venus analogue possible."
    elif flux_ratio > 10.0:
        label   = "Atmosphere Stripped"
        emoji   = "☢️"
        color   = "#ff4f2e"
        note    = "High irradiation — photoevaporation likely denuded any primordial atmosphere."
    else:
        label   = "Bare Rock / Tenuous"
        emoji   = "🪨"
        color   = "#aaaaaa"
        note    = "Insufficient gravity to retain a significant atmosphere."

    return {
        "label"          : label,
        "emoji"          : emoji,
        "color"          : color,
        "likelihood_pct" : likelihood_pct,
        "note"           : note,
    }


# =============================================================================
# PLANET PROFILE CARD BUILDER
# =============================================================================

def _svg_radial_gauge(score: float, color: str, size: int = 108) -> str:
    """Return an inline SVG radial progress ring for the Habitability Score."""
    r       = 44
    cx = cy = size // 2
    circ    = 2 * 3.14159 * r
    filled  = circ * (score / 100.0)
    gap     = circ - filled
    # Neon glow via multiple filter layers
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="glowRing" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="3.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="{color}" stop-opacity="1"/>
      <stop offset="100%" stop-color="{color}" stop-opacity="0.55"/>
    </linearGradient>
  </defs>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="rgba(0,212,255,0.10)" stroke-width="8"/>
  <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
          stroke="url(#ringGrad)" stroke-width="8"
          stroke-linecap="round"
          stroke-dasharray="{filled:.1f} {gap:.1f}"
          transform="rotate(-90 {cx} {cy})"
          filter="url(#glowRing)"/>
  <text x="{cx}" y="{cy - 6}" text-anchor="middle"
        font-family="Space Mono, monospace" font-size="18" font-weight="700"
        fill="{color}">{score:.0f}</text>
  <text x="{cx}" y="{cy + 11}" text-anchor="middle"
        font-family="Space Mono, monospace" font-size="9"
        fill="rgba(0,204,170,0.85)">/100</text>
</svg>"""


def _svg_planet_orb(comp_color: str, hz_color: str, radius_earth: float) -> str:
    """Return an inline SVG animated planet orb with orbital ring."""
    # Scale visual size: clamp between 28–52 px radius
    vis_r = int(np.clip(28 + (radius_earth - 0.5) * 6, 28, 52))
    cx = cy = 60
    orb_r   = vis_r
    ring_rx = orb_r + 22
    ring_ry = int(ring_rx * 0.28)

    # Pick gradient stops from composition colour
    c1 = comp_color
    c2 = hz_color

    return f"""
<svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="orbGrad" cx="38%" cy="32%" r="65%">
      <stop offset="0%"   stop-color="{c1}" stop-opacity="0.95"/>
      <stop offset="55%"  stop-color="{c2}" stop-opacity="0.70"/>
      <stop offset="100%" stop-color="#020816" stop-opacity="0.98"/>
    </radialGradient>
    <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%"   stop-color="{c1}" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="{c1}" stop-opacity="0"/>
    </radialGradient>
    <filter id="orbGlow">
      <feGaussianBlur stdDeviation="5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <clipPath id="orbClip">
      <circle cx="{cx}" cy="{cy}" r="{orb_r}"/>
    </clipPath>
  </defs>
  <circle cx="{cx}" cy="{cy}" r="{orb_r + 16}"
          fill="url(#glowGrad)" opacity="0.7">
    <animate attributeName="r"
      values="{orb_r+14};{orb_r+20};{orb_r+14}"
      dur="4s" repeatCount="indefinite"/>
    <animate attributeName="opacity"
      values="0.5;0.85;0.5" dur="4s" repeatCount="indefinite"/>
  </circle>
  <ellipse cx="{cx}" cy="{cy}" rx="{ring_rx}" ry="{ring_ry}"
           fill="none" stroke="{c2}" stroke-width="1.2" opacity="0.30"/>
  <circle cx="{cx}" cy="{cy}" r="{orb_r}"
          fill="url(#orbGrad)" filter="url(#orbGlow)"/>
  <ellipse cx="{cx}" cy="{int(cy - orb_r*0.18)}"
           rx="{int(orb_r*0.72)}" ry="{int(orb_r*0.14)}"
           fill="{c1}" opacity="0.12" clip-path="url(#orbClip)"/>
  <ellipse cx="{int(cx - orb_r*0.28)}" cy="{int(cy - orb_r*0.30)}"
           rx="{int(orb_r*0.24)}" ry="{int(orb_r*0.14)}"
           fill="white" opacity="0.18" clip-path="url(#orbClip)"/>
  <path d="M {cx-ring_rx} {cy}
           A {ring_rx} {ring_ry} 0 0 0 {cx+ring_rx} {cy}"
        fill="none" stroke="{c2}" stroke-width="1.4" opacity="0.55"/>
</svg>"""


def build_planet_profile_card(
    star_name:            str,
    best_period:          float,
    best_depth:           float,
    best_duration:        float,
    planet_radius_earth:  float,
    planet_mass_earth:    float,
    comp:                 dict,
    hz:                   dict,
    atm:                  dict,
    effective_sma:        float,
    star_luminosity_solar: float,
    star_radius_solar:    float,
    n_transits:           int,
    snr:                  float,
) -> str:
    """
    Build and return the full Planet Profile Card as an HTML string
    ready for st.markdown(..., unsafe_allow_html=True).

    Layout (Streamlit columns are used around this call):
      ┌──────────────────────────────────────────────────────────────────────┐
      │  [ORB]  PLANET DESIGNATION · orbital strip                          │
      │─────────────────────────────────────────────────────────────────────│
      │  [COMPOSITION]   [ATMOSPHERE]   [HABITABILITY SCORE]   [QUICK STATS]│
      │─────────────────────────────────────────────────────────────────────│
      │  TICKER: raw telemetry data stream                                  │
      └──────────────────────────────────────────────────────────────────────┘
    """
    hi          = hz["hi"]
    hz_color    = hz["zone_color"]
    comp_color  = comp["color"]
    atm_color   = atm["color"]

    # ── Tier badge ───────────────────────────────────────────────────────────
    if hi >= 70:
        tier_lbl, tier_bg = "POTENTIALLY HABITABLE",  "rgba(0,255,136,0.12)"
        tier_border = "rgba(0,255,136,0.45)"
    elif hi >= 40:
        tier_lbl, tier_bg = "MARGINAL CONDITIONS",    "rgba(255,208,68,0.10)"
        tier_border = "rgba(255,208,68,0.42)"
    elif hi >= 15:
        tier_lbl, tier_bg = "UNLIKELY HABITABLE",     "rgba(255,136,0,0.10)"
        tier_border = "rgba(255,136,0,0.40)"
    else:
        tier_lbl, tier_bg = "NOT HABITABLE",          "rgba(255,51,0,0.10)"
        tier_border = "rgba(255,51,0,0.38)"

    # ── SVG elements ────────────────────────────────────────────────────────
    orb_svg   = _svg_planet_orb(comp_color, hz_color, planet_radius_earth)
    gauge_svg = _svg_radial_gauge(hi, hz_color)

    # ── Atmosphere likelihood bar ────────────────────────────────────────────
    atm_pct   = atm["likelihood_pct"]
    atm_bar   = f"""
<div style="margin-top:8px;background:rgba(255,255,255,0.07);
            border-radius:3px;height:5px;overflow:hidden;">
  <div style="width:{atm_pct}%;height:100%;
              background:linear-gradient(90deg,{atm_color},{atm_color}88);
              border-radius:3px;box-shadow:0 0 6px {atm_color}66;"></div>
</div>
<div style="font-size:0.60rem;color:{atm_color};margin-top:3px;
            font-family:'Space Mono',monospace;">{atm_pct}% retention est.</div>"""

    # ── Quick-stats column ───────────────────────────────────────────────────
    dur_h  = best_duration * 24.0
    qs_rows = [
        ("Period",   f"{best_period:.4f} d"),
        ("Duration", f"{dur_h:.2f} h"),
        ("Depth",    f"{best_depth*1e6:.0f} ppm"),
        ("Radius",   f"{planet_radius_earth:.2f} R⊕"),
        ("SMA",      f"{effective_sma:.4f} AU"),
        ("Flux",     f"{hz['flux_ratio']:.3f} S⊕"),
        ("Transits", f"~{n_transits}"),
        ("BLS SNR",  f"{snr:.1f}×"),
    ]
    qs_html = "".join(
        f"<div style='display:flex;justify-content:space-between;"
        f"padding:3px 0;border-bottom:1px solid rgba(0,212,255,0.07);'>"
        f"<span style='color:#4a6a8a;'>{k}</span>"
        f"<span style='color:#c8e0f4;'>{v}</span></div>"
        for k, v in qs_rows
    )

    # ── Ticker tape data ─────────────────────────────────────────────────────
    ticker_items = [
        ("STAR",       star_name.upper()),
        ("PERIOD",     f"{best_period:.5f} D"),
        ("DEPTH",      f"{best_depth*1e6:.1f} PPM"),
        ("R_P",        f"{planet_radius_earth:.2f} R⊕"),
        ("M_P",        f"{planet_mass_earth:.2f} M⊕"),
        ("DENSITY",    f"{comp['density_gcc']:.3f} G/CM³"),
        ("COMPOSITION",comp['category'].upper()),
        ("SMA",        f"{effective_sma:.4f} AU"),
        ("FLUX",       f"{hz['flux_ratio']:.3f} S⊕"),
        ("HZ ZONE",    hz['zone_label'].upper()),
        ("HI SCORE",   f"{hi:.1f}/100"),
        ("ATM",        atm['label'].upper()),
        ("L_STAR",     f"{star_luminosity_solar:.4f} L☉"),
        ("R_STAR",     f"{star_radius_solar:.2f} R☉"),
        ("BLS SNR",    f"{snr:.1f}×"),
    ]
    ticker_str = "  ·  ".join(
        f"<span>{k}</span> {v}" for k, v in ticker_items
    )
    # duplicate for seamless loop
    ticker_str = ticker_str + "  ·  " + ticker_str

    # ── Assemble full card HTML ──────────────────────────────────────────────
    card = f"""
<div class="planet-profile-card glass-in">
<div class="ppc-content">

  <div style="display:flex;align-items:center;gap:28px;margin-bottom:26px;flex-wrap:wrap;">

<div style="flex-shrink:0;">{orb_svg}</div>

<div style="flex:1;min-width:200px;">
  <div style="font-family:'Space Mono',monospace;font-size:0.60rem;
              letter-spacing:0.30em;color:#00ccaa;margin-bottom:4px;">
    ◈ PLANET PROFILE — CANDIDATE EXOWORLD
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:1.65rem;
              font-weight:700;color:#eaf4ff;line-height:1.1;
              text-shadow:0 0 30px rgba(0,212,255,0.30);">
    {star_name.upper()} b
  </div>
  <div style="font-size:0.75rem;color:#4a7aaa;font-family:'Space Mono',monospace;
              margin-top:3px;">
    {comp['emoji']} {comp['category']}
    &nbsp;·&nbsp; {hz['zone_emoji']} {hz['zone_label']}
    &nbsp;·&nbsp; {atm['emoji']} {atm['label']}
  </div>
  <div style="margin-top:12px;display:flex;align-items:center;gap:10px;">
    <div style="flex:1;height:3px;border-radius:2px;
                background:linear-gradient(90deg,
                  rgba(0,212,255,0.08),
                  {hz_color}55 {int(min(100, hz['flux_ratio']*40))}%,
                  rgba(0,212,255,0.05));
                position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:{int(min(95,hz['flux_ratio']*38))}%;
                  width:6px;height:3px;border-radius:3px;
                  background:{hz_color};box-shadow:0 0 8px {hz_color};"></div>
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                color:#336688;white-space:nowrap;">
      {effective_sma:.4f} AU  ·  P = {best_period:.4f} d
    </div>
  </div>
</div>

<div style="flex-shrink:0;padding:8px 18px;border-radius:10px;
            background:{tier_bg};border:1px solid {tier_border};
            font-family:'Space Mono',monospace;font-size:0.62rem;
            letter-spacing:0.10em;color:#eaf4ff;text-align:center;
            text-shadow:0 0 8px {hz_color}66;">
  {hz['zone_emoji']}<br>
  <span style="font-size:0.78rem;font-weight:700;color:{hz_color};">
    {tier_lbl}
  </span>
</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;
          margin-bottom:4px;">

<div class="ppc-icon-panel" style="border-color:rgba({int(comp_color[1:3],16)},{int(comp_color[3:5],16)},{int(comp_color[5:7],16)},0.35);">
  <div class="ppc-panel-title">◈ Composition</div>
  <div class="ppc-icon" style="color:{comp_color};">{comp['emoji']}</div>
  <div class="ppc-panel-value" style="color:{comp_color};
       text-shadow:0 0 14px {comp_color}66;">
    {comp['category']}
  </div>
  <div class="ppc-panel-sub" style="margin-top:6px;">
    <span style="color:#00ccaa;">{comp['density_gcc']:.3f}</span> g/cm³<br>
    <span style="color:#00ccaa;">{comp['density_rel']:.2f}×</span> Earth<br>
    R<sub>p</sub> = <span style="color:#00ccaa;">{planet_radius_earth:.2f}</span> R⊕
  </div>
  <div style="margin-top:9px;background:rgba(255,255,255,0.06);
              border-radius:3px;height:4px;overflow:hidden;">
    <div style="width:{int(min(100, comp['density_gcc'] / 14.0 * 100))}%;
                height:100%;background:{comp_color};opacity:0.75;
                border-radius:3px;"></div>
  </div>
  <div style="font-size:0.58rem;color:#336655;margin-top:3px;
              font-family:'Space Mono',monospace;">
    density  (0 → 14 g/cm³)
  </div>
</div>

<div class="ppc-icon-panel" style="border-color:rgba({int(atm_color[1:3],16)},{int(atm_color[3:5],16)},{int(atm_color[5:7],16)},0.32);">
  <div class="ppc-panel-title">💨 Atmosphere</div>
  <div class="ppc-icon" style="color:{atm_color};">{atm['emoji']}</div>
  <div class="ppc-panel-value" style="color:{atm_color};
       text-shadow:0 0 14px {atm_color}66;font-size:0.88rem;">
    {atm['label']}
  </div>
  <div class="ppc-panel-sub" style="margin-top:6px;">
    {atm['note']}
  </div>
  {atm_bar}
</div>

<div class="ppc-icon-panel" style="border-color:{hz_color}44;">
  <div class="ppc-panel-title">🌿 Habitability Score</div>
  <div class="ppc-score-ring">{gauge_svg}</div>
  <div class="ppc-panel-value" style="color:{hz_color};
       text-shadow:0 0 14px {hz_color}66;font-size:0.92rem;">
    {hz['zone_label']}
  </div>
  <div class="ppc-panel-sub" style="margin-top:4px;">
    <span style="color:#00ccaa;">HZ pos</span>
      {hz['score_breakdown']['hz_position']:.1f}
    &nbsp;·&nbsp;
    <span style="color:#00ccaa;">size</span>
      {hz['score_breakdown']['planet_size']:.1f}
    &nbsp;·&nbsp;
    <span style="color:#00ccaa;">orbit</span>
      {hz['score_breakdown']['orbital']:.1f}
  </div>
</div>

<div class="ppc-icon-panel">
  <div class="ppc-panel-title">📡 Telemetry</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
              margin-top:4px;">
    {qs_html}
  </div>
</div>

  </div>
  <div class="ppc-ticker">
<div class="ppc-ticker-inner">{ticker_str}</div>
  </div>

</div></div>"""
    return card


# =============================================================================
# HABITABILITY INDEX CALCULATOR  —  Goldilocks / Circumstellar HZ
# =============================================================================

def calculate_habitability_index(
    semi_major_axis_au: float,
    star_luminosity_solar: float,
    planet_radius_earth: float  = 1.0,
    planet_mass_earth:   float  = 1.0,
) -> dict:
    """
    Compute a multi-factor Habitability Index (HI, 0–100) based on:

      1. Goldilocks Zone position  — Kopparapu et al. (2013/2014) flux limits
      2. Planet size bonus/penalty — rocky sub-Neptune sweet spot
      3. Orbital stability factor — circular-orbit penalty for extreme radii

    Parameters
    ----------
    semi_major_axis_au    : orbital semi-major axis in AU
    star_luminosity_solar : stellar luminosity in units of L☉
    planet_radius_earth   : planet radius in R⊕  (default 1.0)
    planet_mass_earth     : planet mass   in M⊕  (default 1.0)

    Returns
    -------
    dict with keys
        hi              : float  0–100 Habitability Index
        hz_inner_au     : float  inner HZ boundary (AU)
        hz_outer_au     : float  outer HZ boundary (AU)
        hz_opt_inner_au : float  optimistic inner edge (Venus-runaway)
        hz_opt_outer_au : float  optimistic outer edge (maximum greenhouse)
        flux_ratio      : float  stellar flux at planet / Earth's flux
        zone_label      : str    "Scorched", "Hot Edge", "Green Zone",
                                 "Cool Edge", "Frozen", "Deep Freeze"
        zone_color      : str    hex colour for UI
        zone_emoji      : str    emoji symbol
        score_breakdown : dict   sub-scores for each factor
        description     : str    science narrative
    """

    # ── Kopparapu et al. (2013) flux limits ─────────────────────────────────
    # Empirical coefficients for Sun-like star effective stellar flux (S_eff)
    # Conservative HZ:   runaway greenhouse → maximum greenhouse
    # Optimistic HZ:     recent Venus       → early Mars

    # Effective stellar flux at planet (Earth = 1.0)
    flux_ratio = star_luminosity_solar / (semi_major_axis_au ** 2)

    # Conservative boundaries in terms of S_eff (referenced to Earth = 1.0)
    S_inner_conservative = 1.107   # Runaway Greenhouse
    S_outer_conservative = 0.356   # Maximum Greenhouse

    # Optimistic boundaries
    S_inner_optimistic   = 1.776   # Recent Venus
    S_outer_optimistic   = 0.320   # Early Mars

    # Convert S_eff boundaries → AU for this star
    hz_inner_au     = float(np.sqrt(star_luminosity_solar / S_inner_conservative))
    hz_outer_au     = float(np.sqrt(star_luminosity_solar / S_outer_conservative))
    hz_opt_inner_au = float(np.sqrt(star_luminosity_solar / S_inner_optimistic))
    hz_opt_outer_au = float(np.sqrt(star_luminosity_solar / S_outer_optimistic))

    # ── Zone classification ──────────────────────────────────────────────────
    if flux_ratio > S_inner_optimistic:
        zone_label = "Scorched"
        zone_color = "#ff2200"
        zone_emoji = "🔥"
        hz_score   = 0.0

    elif flux_ratio > S_inner_conservative:
        # Between optimistic and conservative inner edge → hot but possible
        t = (flux_ratio - S_inner_conservative) / (S_inner_optimistic - S_inner_conservative)
        hz_score   = 25.0 * (1.0 - t)
        zone_label = "Hot Edge"
        zone_color = "#ff8800"
        zone_emoji = "☀️"

    elif flux_ratio >= S_outer_conservative:
        # Inside the conservative HZ — full score, peak at Earth-like ~1.0
        # Gaussian centred on flux_ratio = 0.75 (Earth's optimum)
        centre    = 0.75
        sigma_hz  = 0.25
        hz_score  = 55.0 * np.exp(-0.5 * ((flux_ratio - centre) / sigma_hz) ** 2) + 45.0
        zone_label = "Green Zone  🌿"
        zone_color = "#00ff88"
        zone_emoji = "🌿"

    elif flux_ratio >= S_outer_optimistic:
        # Between conservative and optimistic outer edge → cold but possible
        t = (S_outer_conservative - flux_ratio) / (S_outer_conservative - S_outer_optimistic)
        hz_score   = 25.0 * (1.0 - t)
        zone_label = "Cool Edge"
        zone_color = "#4488ff"
        zone_emoji = "❄️"

    else:
        zone_label = "Frozen"
        zone_color = "#aaccff"
        zone_emoji = "🧊"
        hz_score   = 0.0

    # ── Planet size factor (0–20 pts) ───────────────────────────────────────
    # Peak at 1.0–1.5 R⊕; drops off outside 0.5–2.5 R⊕
    r = planet_radius_earth
    if 0.5 <= r <= 2.5:
        size_score = 20.0 * np.exp(-0.5 * ((r - 1.2) / 0.8) ** 2)
    elif r < 0.5:
        size_score = 5.0 * (r / 0.5)
    else:
        # Larger → less habitable (sub-Neptune → gas mini-Neptune)
        size_score = max(0.0, 20.0 - (r - 2.5) * 8.0)
    size_score = float(np.clip(size_score, 0.0, 20.0))

    # ── Orbital stability factor (0–10 pts) ─────────────────────────────────
    # Penalty for very tight (tidal lock risk) or very wide orbits
    if   semi_major_axis_au < 0.05:
        orbital_score = 2.0
    elif semi_major_axis_au < 0.15:
        orbital_score = 7.0
    elif semi_major_axis_au < 5.0:
        orbital_score = 10.0
    else:
        orbital_score = max(0.0, 10.0 - (semi_major_axis_au - 5.0) * 1.5)
    orbital_score = float(np.clip(orbital_score, 0.0, 10.0))

    # ── Total HI ────────────────────────────────────────────────────────────
    # hz_score (0–100) is weighted 70 %, size 20 %, orbital 10 %
    hi = float(np.clip(
        0.70 * hz_score + size_score + orbital_score,
        0.0, 100.0
    ))

    # ── Narrative ───────────────────────────────────────────────────────────
    narratives = {
        "Scorched"  : "Far too close to the host star. Runaway greenhouse is inevitable; "
                      "surface liquid water is impossible.",
        "Hot Edge"  : "Inside the optimistic inner HZ. Surface conditions may permit water "
                      "transiently under high atmospheric pressure (Venus analogue).",
        "Green Zone  🌿" : "Inside the conservative Habitable Zone. Stellar flux allows "
                      "liquid water on the surface. Prime target for atmospheric follow-up.",
        "Cool Edge" : "Between conservative and optimistic outer edges. "
                      "CO₂-rich greenhouse may keep surface above freezing.",
        "Frozen"    : "Beyond the outer HZ boundary. Maximum greenhouse effect is "
                      "insufficient; surface water permanently frozen.",
    }
    description = narratives.get(zone_label,
        "Deep-freeze regime — far beyond the star's influence on habitability.")

    return {
        "hi"              : round(hi, 1),
        "hz_inner_au"     : round(hz_inner_au,     4),
        "hz_outer_au"     : round(hz_outer_au,     4),
        "hz_opt_inner_au" : round(hz_opt_inner_au, 4),
        "hz_opt_outer_au" : round(hz_opt_outer_au, 4),
        "flux_ratio"      : round(flux_ratio,       4),
        "zone_label"      : zone_label.replace("  🌿", ""),
        "zone_color"      : zone_color,
        "zone_emoji"      : zone_emoji,
        "score_breakdown" : {
            "hz_position" : round(0.70 * hz_score, 1),
            "planet_size" : round(size_score,       1),
            "orbital"     : round(orbital_score,    1),
        },
        "description"     : description,
    }


def plot_goldilocks_zone(
    hz_result:          dict,
    semi_major_axis_au: float,
    star_luminosity_solar: float,
    planet_name:        str = "Detected Planet",
) -> "plt.Figure":
    """
    Render a publication-style Goldilocks Zone diagram showing:
      • Temperature gradient background (hot → cold)
      • Optimistic HZ band (warm amber)
      • Conservative HZ band (bright green)
      • Planet marker + Earth reference
      • Solar system reference planets (Mercury→Mars)
      • Annotated HZ boundaries
    """
    fig, ax = make_fig(h=4.2)

    r     = hz_result
    a_max = max(r["hz_opt_outer_au"] * 1.55, semi_major_axis_au * 1.25, 0.25)
    a_min = max(0.0, min(r["hz_opt_inner_au"] * 0.38, semi_major_axis_au * 0.55, 0.04))

    x = np.linspace(a_min, a_max, 800)

    # ── Temperature-gradient background ─────────────────────────────────────
    flux_x = star_luminosity_solar / (x ** 2)
    # Normalise flux to 0-1 for colour mapping (log scale)
    log_flux = np.log10(np.clip(flux_x, 0.01, 1000))
    log_min, log_max = np.log10(0.01), np.log10(1000)
    norm_flux = (log_flux - log_min) / (log_max - log_min)

    for i in range(len(x) - 1):
        v = float(norm_flux[i])
        if v > 0.75:
            c = (0.9, 0.2 * (1 - v), 0.0, 0.28)
        elif v > 0.45:
            c = (0.9 * v, 0.5 * v, 0.1, 0.22)
        else:
            c = (0.05, 0.15 + 0.4 * v, 0.55 + 0.35 * (1 - v), 0.28)
        ax.axvspan(x[i], x[i + 1], color=c, linewidth=0)

    # ── Zone bands ──────────────────────────────────────────────────────────
    ax.axvspan(r["hz_opt_inner_au"], r["hz_opt_outer_au"],
               color="#ffd04488", alpha=0.35, label="Optimistic HZ", zorder=2)
    ax.axvspan(r["hz_inner_au"], r["hz_outer_au"],
               color="#00ff8866", alpha=0.50, label="Conservative HZ", zorder=3)

    # Boundary lines
    for val, ls, col in [
        (r["hz_opt_inner_au"],  ":",  "#ffaa22"),
        (r["hz_inner_au"],      "--", "#88ffaa"),
        (r["hz_outer_au"],      "--", "#88ffaa"),
        (r["hz_opt_outer_au"],  ":",  "#4488ff"),
    ]:
        ax.axvline(val, color=col, lw=1.0, ls=ls, alpha=0.70, zorder=4)

    # Boundary labels (top of axes)
    labels_top = [
        (r["hz_opt_inner_au"],  "Opt.\nInner",  "#ffaa22", "right"),
        (r["hz_inner_au"],      "HZ\nInner",    "#88ffaa", "right"),
        (r["hz_outer_au"],      "HZ\nOuter",    "#88ffaa", "left"),
        (r["hz_opt_outer_au"],  "Opt.\nOuter",  "#4488ff", "left"),
    ]
    for xv, lbl, col, ha in labels_top:
        if a_min < xv < a_max:
            ax.text(xv, 0.97, lbl, transform=ax.get_xaxis_transform(),
                    color=col, fontsize=6.5, ha=ha, va="top",
                    fontfamily="monospace", alpha=0.85)

    # ── Reference planets ───────────────────────────────────────────────────
    sol_planets = [
        (0.387, "☿", "#aaaaaa"),
        (0.723, "♀", "#ffcc66"),
        (1.000, "⊕", "#4a9eff"),
        (1.524, "♂", "#ff6644"),
    ]
    for d_au, sym, col in sol_planets:
        if a_min < d_au < a_max:
            ax.scatter(d_au, 0.5, s=70, color=col, zorder=6, alpha=0.65)
            ax.text(d_au, 0.56, sym, color=col, ha="center", fontsize=9,
                    transform=ax.get_xaxis_transform(), zorder=7, alpha=0.80)

    # ── Planet marker ────────────────────────────────────────────────────────
    pcolor = hz_result["zone_color"]
    ax.scatter(semi_major_axis_au, 0.5, s=200, color=pcolor,
               zorder=10, edgecolors="white", linewidths=1.2,
               transform=ax.get_xaxis_transform())
    ax.text(semi_major_axis_au, 0.38,
            f"{hz_result['zone_emoji']}  {semi_major_axis_au:.3f} AU",
            color=pcolor, ha="center", fontsize=8, fontweight="bold",
            transform=ax.get_xaxis_transform(), zorder=11,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#03060f",
                      edgecolor=pcolor, alpha=0.88))

    # ── Cosmetics ────────────────────────────────────────────────────────────
    ax.set_xlim(a_min, a_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Orbital Distance  [AU]", color=C_TICK, fontsize=9.5)
    ax.yaxis.set_visible(False)
    ax.set_title(
        f"Goldilocks Zone  ·  L★ = {star_luminosity_solar:.3f} L☉  "
        f"·  Flux = {hz_result['flux_ratio']:.3f} S⊕",
        color=C_TICK, fontsize=9, loc="left", pad=6,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.30,
              facecolor=BG_PANEL, edgecolor="#00aacc", labelcolor="white")
    fig.tight_layout(pad=1.5)
    return fig


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
              facecolor=BG_PANEL, edgecolor="#00aacc", labelcolor="white")
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
               facecolor=BG_PANEL, edgecolor="#00aacc", labelcolor="white")
    ax1.set_title("Flattened (top)  →  After outlier removal (bottom)",
                  color=C_TICK, fontsize=9, loc="left", pad=6)

    ax2.plot(clean_time, clean_flux, color=C_CLEAN, lw=0.5, alpha=0.65, zorder=2,
             label="Planet-search ready  (outliers removed)")
    ax2.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    ax2.set_xlabel("Time  [Days]", color=C_TICK, fontsize=9)
    ax2.set_ylabel("Flux", color=C_CLEAN, fontsize=9)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.25,
               facecolor=BG_PANEL, edgecolor="#00aacc", labelcolor="white")
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
                  facecolor=BG_PANEL, edgecolor="#00aacc", labelcolor="white")
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
                   facecolor=BG_PANEL, edgecolor="#00aacc", labelcolor="white")
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
              color:#00ffff;font-weight:700;letter-spacing:-0.5px;
              text-shadow:0 0 20px rgba(0,255,255,0.50);'>
    🔭 EXOPLANET<br>HUNTER
  </div>
  <div style='font-size:0.68rem;color:#00ccaa;text-transform:uppercase;
              letter-spacing:0.13em;margin-top:5px;
              text-shadow:0 0 5px rgba(0,204,170,0.3);'>
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
            f"<div style='font-size:0.67rem;color:#00ccaa;margin-top:-4px;margin-bottom:6px;'>"
            f"Quarter: <span style='color:#00ffff'>Q{time_segment}</span> · Range 0–17</div>",
            unsafe_allow_html=True)
    else:
        time_segment = st.slider("sector_slider", min_value=1, max_value=85,
                                 value=1, label_visibility="collapsed")
        st.markdown(
            f"<div style='font-size:0.67rem;color:#00ccaa;margin-top:-4px;margin-bottom:6px;'>"
            f"Sector: <span style='color:#00ffff'>S{time_segment}</span> · Range 1–85</div>",
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">🌟 HOST STAR RADIUS</div>', unsafe_allow_html=True)
    star_radius_solar = st.number_input(
        "star_radius_input",
        min_value=0.1, max_value=100.0,
        value=float(st.session_state.star_radius_solar),
        step=0.05,
        format="%.2f",
        label_visibility="collapsed",
        help="Host star radius in solar radii — used to convert transit depth → planet radius.",
    )
    st.session_state.star_radius_solar = star_radius_solar
    st.markdown(
        "<div style='font-size:0.67rem;color:#00ccaa;margin-top:-4px;margin-bottom:6px;'>"
        f"R★ = <span style='color:#00ffff'>{star_radius_solar:.2f} R☉</span>"
        " &nbsp;·&nbsp; Sun = 1.00</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label" style="margin-top:10px;">⚖️ PLANET MASS ESTIMATE</div>',
                unsafe_allow_html=True)
    planet_mass_earth = st.number_input(
        "planet_mass_input",
        min_value=0.01, max_value=5000.0,
        value=float(st.session_state.planet_mass_earth),
        step=0.5,
        format="%.2f",
        label_visibility="collapsed",
        help="Planet mass in Earth masses (M⊕). Use radial-velocity or TTV data, or keep default for a rough estimate.",
    )
    st.session_state.planet_mass_earth = planet_mass_earth
    st.markdown(
        "<div style='font-size:0.67rem;color:#00ccaa;margin-top:-4px;margin-bottom:6px;'>"
        f"M = <span style='color:#00ffff'>{planet_mass_earth:.2f} M⊕</span>"
        " &nbsp;·&nbsp; Earth = 1.00</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label" style="margin-top:10px;">💡 STELLAR LUMINOSITY</div>',
                unsafe_allow_html=True)
    star_luminosity_solar = st.number_input(
        "star_lum_input",
        min_value=0.0001, max_value=1_000_000.0,
        value=float(st.session_state.star_luminosity_solar),
        step=0.1,
        format="%.4f",
        label_visibility="collapsed",
        help="Host star luminosity in solar units (L☉). Sun = 1.0. "
             "Red dwarfs ≈ 0.001–0.08; F-type ≈ 2–5; giants > 100.",
    )
    st.session_state.star_luminosity_solar = star_luminosity_solar
    st.markdown(
        "<div style='font-size:0.67rem;color:#00ccaa;margin-top:-4px;margin-bottom:6px;'>"
        f"L★ = <span style='color:#00ffff'>{star_luminosity_solar:.4f} L☉</span>"
        " &nbsp;·&nbsp; Sun = 1.0000</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label" style="margin-top:10px;">🪐 SEMI-MAJOR AXIS</div>',
                unsafe_allow_html=True)
    semi_major_axis_au = st.number_input(
        "sma_input",
        min_value=0.001, max_value=500.0,
        value=float(st.session_state.semi_major_axis_au),
        step=0.01,
        format="%.4f",
        label_visibility="collapsed",
        help="Orbital semi-major axis in Astronomical Units (AU). "
             "Earth = 1.0 AU. Derived from BLS period via Kepler's 3rd law if left at default.",
    )
    st.session_state.semi_major_axis_au = semi_major_axis_au
    _sma_note = (
        "Set manually or use Kepler's 3rd law estimate below"
        if semi_major_axis_au == 1.0
        else f"a = <span style='color:#00ffff'>{semi_major_axis_au:.4f} AU</span>"
    )
    st.markdown(
        f"<div style='font-size:0.67rem;color:#00ccaa;margin-top:-4px;margin-bottom:6px;'>"
        f"{_sma_note} &nbsp;·&nbsp; Earth = 1.0000 AU</div>",
        unsafe_allow_html=True,
    )

    # ── NASA Sync Status Badge ────────────────────────────────────────────────
    _sync = st.session_state.nasa_sync_status
    if _sync == "ok":
        _fields = " · ".join(getattr(st.session_state, "nasa_sync_fields", []))
        st.markdown(f"""
<div style='margin-top:10px;padding:8px 12px;border-radius:8px;
            background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.35);
            font-family:Space Mono,monospace;font-size:0.63rem;color:#00ff88;'>
  ✅ NASA LIVE DATA<br>
  <span style='color:#88ffcc;'>{_fields}</span>
</div>""", unsafe_allow_html=True)
    elif _sync == "partial":
        _fields = " · ".join(getattr(st.session_state, "nasa_sync_fields", []))
        st.markdown(f"""
<div style='margin-top:10px;padding:8px 12px;border-radius:8px;
            background:rgba(255,208,0,0.08);border:1px solid rgba(255,208,0,0.35);
            font-family:Space Mono,monospace;font-size:0.63rem;color:#ffd044;'>
  ⚡ PARTIAL NASA DATA<br>
  <span style='color:#ffe88a;'>{_fields} synced</span>
</div>""", unsafe_allow_html=True)
    elif _sync == "not_found":
        st.markdown("""
<div style='margin-top:10px;padding:8px 12px;border-radius:8px;
            background:rgba(255,60,60,0.08);border:1px solid rgba(255,60,60,0.32);
            font-family:Space Mono,monospace;font-size:0.63rem;color:#ff6060;'>
  ⚠️ NOT IN NASA ARCHIVE<br>
  <span style='color:#ffaaaa;'>Sliders unchanged</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style='font-size:0.71rem;color:#00ccaa;line-height:2.0;'>
  <b style='color:#00ffff'>Detection Pipeline</b><br>
  ① Download from MAST<br>
  ② NaN removal + normalise<br>
  ③ Savitzky-Golay flatten<br>
  ④ Outlier sigma-clip (4σ)<br>
  ⑤ BLS periodogram<br>
  ⑥ Phase-fold &amp; bin
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">👨‍🚀 SYSTEMS OFFICER</div>', unsafe_allow_html=True)
    st.markdown("""
<div style='font-family:Space Mono,monospace;font-size:0.75rem;'>
  <b style='color:#00ffff'>Kishore (Kai)</b><br>
  <span style='color:#88aacc'>Status:</span> <span style='color:#00ff88'>ACTIVE</span><br>
  <span style='color:#88aacc'>Clearance:</span> <span style='color:#00ff88'>LEVEL 3</span><br>
  <span style='color:#88aacc'>Mission:</span> MIT 2031<br>
  <span style='color:#88aacc'>Modules:</span> ATS-1 | Exoplanet Hunter
</div>
""", unsafe_allow_html=True)
    st.info("🛸 Searching the cosmos for the next Earth-like world.")


# (Session state initialised at the top of the file, before the sidebar)



# =============================================================================
# NASA API AUTO-SYNC  — fires once per new planet name
# =============================================================================
def _try_nasa_sync(planet_name: str):
    """
    Query NASA Exoplanet Archive for `planet_name` and update session-state
    sliders in-place.  Records sync status and the resolved planet name.

    Resolution strategy (delegated to fetch_nasa_exoplanet_data):
      1. Exact pl_name match       "Kepler-10 b"
      2. Exact hostname match      "Kepler-10"
      3. pl_name prefix LIKE       "Kepler-307%"
      4. hostname prefix LIKE      "Kepler-307%"
    """
    data = fetch_nasa_exoplanet_data(planet_name)
    if data is None:
        st.session_state.nasa_sync_status = "not_found"
        return

    updated = []
    if data["st_rad"] is not None:
        st.session_state.star_radius_solar    = float(np.clip(data["st_rad"], 0.1, 100.0))
        updated.append("R★")
    if data["pl_masse"] is not None:
        st.session_state.planet_mass_earth    = float(np.clip(data["pl_masse"], 0.01, 5000.0))
        updated.append("M♁")
    if data["st_lum"] is not None:
        st.session_state.star_luminosity_solar = float(np.clip(data["st_lum"], 0.0001, 1_000_000.0))
        updated.append("L★")
    if data["pl_orbsmax"] is not None:
        st.session_state.semi_major_axis_au   = float(np.clip(data["pl_orbsmax"], 0.001, 500.0))
        updated.append("SMA")

    st.session_state.nasa_sync_status = "ok" if len(updated) == 4 else "partial"
    st.session_state.nasa_sync_fields = updated


# Trigger sync whenever the star name changes (landing screen or results)
_current_name = st.session_state.star_name.strip()
if _current_name and _current_name != st.session_state.nasa_synced_planet:
    _try_nasa_sync(_current_name)
    st.session_state.nasa_synced_planet = _current_name


# =============================================================================
# MAIN PAGE — LANDING
# =============================================================================
st.markdown('<div class="hero-title animate-in">EXOPLANET HUNTER</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub animate-in delay-1">'
            '[ MULTI-MISSION · BOX LEAST SQUARES DETECTION ENGINE ]</div>',
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

    # ── Native autocomplete: text input drives a live-filtered selectbox ─────
    _typed = st.text_input(
        "planet_search_input",
        value=st.session_state.star_name,
        placeholder="🔭  Type planet name — e.g. Kepler-442 b, TRAPPIST-1 e …",
        label_visibility="collapsed",
    )

    _suggestions = search_planets(_typed) if _typed.strip() else []

    _selected_name = None
    if _suggestions:
        # Prepend the raw typed value so the user can also just hit Scan directly
        _opts = [_typed.strip()] + [s for s in _suggestions if s.strip().lower() != _typed.strip().lower()]
        _pick = st.selectbox(
            "autocomplete_dropdown",
            options=_opts,
            label_visibility="collapsed",
        )
        _selected_name = _pick
    else:
        # No matches yet — treat whatever is typed as the selection
        _selected_name = _typed

    # Commit whenever the resolved name differs from what's stored
    if _selected_name and _selected_name.strip() != st.session_state.star_name:
        _clean = _selected_name.strip()
        st.session_state.star_name = _clean
        st.session_state.nasa_sync_status = None
        st.session_state.nasa_synced_planet = ""
        st.rerun()

    # NASA sync status feedback on landing page
    _sync = st.session_state.nasa_sync_status
    if _sync == "ok":
        _fields = " · ".join(getattr(st.session_state, "nasa_sync_fields", []))
        st.markdown(f"""
<div style='margin:-4px 0 14px;padding:9px 14px;border-radius:8px;
            background:rgba(0,255,136,0.07);border:1px solid rgba(0,255,136,0.32);
            font-family:Space Mono,monospace;font-size:0.68rem;color:#00ff88;'>
  ✅ &nbsp;NASA LIVE DATA LOADED — sliders synced with real archive values<br>
  <span style='color:#88ffcc;font-size:0.60rem;'>Updated: {_fields}</span>
</div>""", unsafe_allow_html=True)
    elif _sync == "partial":
        _fields = " · ".join(getattr(st.session_state, "nasa_sync_fields", []))
        st.markdown(f"""
<div style='margin:-4px 0 14px;padding:9px 14px;border-radius:8px;
            background:rgba(255,208,0,0.07);border:1px solid rgba(255,208,0,0.32);
            font-family:Space Mono,monospace;font-size:0.68rem;color:#ffd044;'>
  ⚡ &nbsp;PARTIAL NASA DATA — some fields missing in archive<br>
  <span style='color:#ffe88a;font-size:0.60rem;'>Synced: {_fields} · Rest kept at previous values</span>
</div>""", unsafe_allow_html=True)
    elif _sync == "not_found":
        st.markdown(f"""
<div style='margin:-4px 0 14px;padding:9px 14px;border-radius:8px;
            background:rgba(255,80,80,0.07);border:1px solid rgba(255,80,80,0.28);
            font-family:Space Mono,monospace;font-size:0.68rem;color:#ff7070;'>
  ⚠️ &nbsp;<b style='color:#ffaaaa'>"{st.session_state.star_name}"</b> not found in NASA Exoplanet Archive<br>
  <span style='color:#ffbbbb;font-size:0.60rem;'>
    Try the full planet name, e.g. "Kepler-10 b" · Sliders unchanged
  </span>
</div>""", unsafe_allow_html=True)

    # Dynamic suggestions per mission
    if selected_mission == "Kepler":
        chips = ["Kepler-10", "Kepler-22", "Kepler-90", "Kepler-186"]
    else:
        chips = ["TRAPPIST-1", "TOI-700", "WASP-126", "HD 209458"]

    chip_html = " ".join(
        f'<code style="color:#00ffff;background:rgba(0,212,255,0.09);'
        f'padding:2px 9px;border-radius:5px;border:1px solid rgba(0,212,255,0.22);">{c}</code>'
        for c in chips
    )
    st.markdown(f"""
<div style="display:flex;gap:12px;justify-content:center;
            margin-top:-8px;margin-bottom:22px;flex-wrap:wrap;">
  <span style="font-size:0.78rem;color:#00ccaa;align-self:center;">SUGGESTIONS:</span>
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
<div class="status-text animate-in" style='margin-bottom:0.6rem;'>
  > ANALYSING &nbsp;<span style='color:#00ffff'>{st.session_state.star_name.upper()}</span>
  &nbsp;·&nbsp; {selected_mission.upper()} {time_label} {time_segment}
</div>""", unsafe_allow_html=True)

# ── NASA Archive sync status banner (results page) ────────────────────────────
_r_sync = st.session_state.nasa_sync_status
if _r_sync == "ok":
    _r_fields = " · ".join(getattr(st.session_state, "nasa_sync_fields", []))
    st.markdown(f"""
<div style='margin-bottom:1rem;padding:7px 14px;border-radius:8px;
            background:rgba(0,255,136,0.06);border:1px solid rgba(0,255,136,0.28);
            font-family:Space Mono,monospace;font-size:0.65rem;color:#00ff88;'>
  ✅ &nbsp;NASA EXOPLANET ARCHIVE SYNC COMPLETE &nbsp;·&nbsp;
  <span style='color:#88ffcc;'>{_r_fields}</span>
</div>""", unsafe_allow_html=True)
elif _r_sync == "partial":
    _r_fields = " · ".join(getattr(st.session_state, "nasa_sync_fields", []))
    st.markdown(f"""
<div style='margin-bottom:1rem;padding:7px 14px;border-radius:8px;
            background:rgba(255,208,0,0.06);border:1px solid rgba(255,208,0,0.28);
            font-family:Space Mono,monospace;font-size:0.65rem;color:#ffd044;'>
  ⚡ &nbsp;PARTIAL NASA SYNC &nbsp;·&nbsp;
  <span style='color:#ffe88a;'>{_r_fields} loaded · remaining fields at manual values</span>
</div>""", unsafe_allow_html=True)
elif _r_sync == "not_found":
    st.warning(
        f"⚠️ **NASA Archive:** No entry found for *\"{st.session_state.star_name}\"*. "
        "Sliders are at manually set values. "
        "Try the full planet name (e.g. `Kepler-10 b`) for automatic data sync.",
        icon=None
    )


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

# ── Pre-compute all derived science needed for the Profile Card ──────────────
_planet_radius_earth = estimate_planet_radius_earth(best_depth, star_radius_solar)
_comp                = get_planet_composition(planet_mass_earth, _planet_radius_earth)

_star_mass_solar     = star_luminosity_solar ** 0.25
_sma_kepler_au       = (best_period / 365.25) ** (2.0 / 3.0) * _star_mass_solar ** (1.0 / 3.0)
_effective_sma       = semi_major_axis_au if abs(semi_major_axis_au - 1.0) > 0.001 \
                       else _sma_kepler_au

_hz = calculate_habitability_index(
    semi_major_axis_au    = _effective_sma,
    star_luminosity_solar = star_luminosity_solar,
    planet_radius_earth   = _planet_radius_earth,
    planet_mass_earth     = planet_mass_earth,
)

_atm = get_atmosphere_potential(
    planet_radius_earth = _planet_radius_earth,
    planet_mass_earth   = planet_mass_earth,
    density_gcc         = _comp["density_gcc"],
    hz_zone_label       = _hz["zone_label"],
    flux_ratio          = _hz["flux_ratio"],
)

# ── PLANET PROFILE CARD ───────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-1">◈  PLANET PROFILE</div>',
            unsafe_allow_html=True)

_profile_html = build_planet_profile_card(
    star_name             = st.session_state.star_name,
    best_period           = best_period,
    best_depth            = best_depth,
    best_duration         = best_duration,
    planet_radius_earth   = _planet_radius_earth,
    planet_mass_earth     = planet_mass_earth,
    comp                  = _comp,
    hz                    = _hz,
    atm                   = _atm,
    effective_sma         = _effective_sma,
    star_luminosity_solar = star_luminosity_solar,
    star_radius_solar     = star_radius_solar,
    n_transits            = n_transits,
    snr                   = snr,
)
st.markdown(_profile_html, unsafe_allow_html=True)

# ── Stat cards ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-1">★  DETECTED PLANET PARAMETERS</div>',
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

# ── Planet Composition ────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-2">★  PLANET COMPOSITION ANALYSIS</div>',
            unsafe_allow_html=True)

_comp_col1, _comp_col2, _comp_col3 = st.columns([1.1, 1.1, 1.8])

with _comp_col1:
    st.markdown(f"""
<div class="stat-card glass-in delay-2" style="border-color:rgba({int(_comp['color'][1:3],16)},{int(_comp['color'][3:5],16)},{int(_comp['color'][5:7],16)},0.55);">
  <div class="stat-label">PLANET TYPE</div>
  <div style="font-family:'Space Mono',monospace;font-size:1.55rem;font-weight:700;
              color:{_comp['color']};text-shadow:0 0 18px {_comp['color']}88;margin-top:4px;">
    {_comp['emoji']}  {_comp['category']}
  </div>
</div>""", unsafe_allow_html=True)

with _comp_col2:
    st.markdown(f"""
<div class="stat-card glass-in delay-2">
  <div class="stat-label">BULK DENSITY</div>
  <div class="stat-value">{_comp['density_gcc']:.3f}<span class="stat-unit"> g/cm³</span></div>
  <div style="font-size:0.68rem;color:#00ccaa;margin-top:4px;font-family:'Space Mono',monospace;">
    {_comp['density_rel']:.2f} × Earth &nbsp;|&nbsp;
    R<sub>p</sub> = {_planet_radius_earth:.2f} R⊕
  </div>
</div>""", unsafe_allow_html=True)

with _comp_col3:
    st.markdown(f"""
<div class="stat-card glass-in delay-2" style="border-color:rgba({int(_comp['color'][1:3],16)},{int(_comp['color'][3:5],16)},{int(_comp['color'][5:7],16)},0.35);">
  <div class="stat-label">COMPOSITION NOTE</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;
              color:#b0cce8;line-height:1.55;margin-top:5px;">
    {_comp['description']}
  </div>
  <div style="font-size:0.64rem;color:#556677;font-family:'Space Mono',monospace;margin-top:7px;">
    ⚠ Density estimate requires a planet mass (set in sidebar).
    Radius derived from transit depth × stellar radius.
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Habitability Index ────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-2">🌿  HABITABILITY INDEX  —  GOLDILOCKS ZONE ANALYSIS</div>',
            unsafe_allow_html=True)

# ── HI score gauge + zone card ───────────────────────────────────────────────
_hi_col1, _hi_col2, _hi_col3, _hi_col4 = st.columns([1.0, 1.0, 1.0, 1.8])

_hz_color = _hz["zone_color"]
_hi_val   = _hz["hi"]

# Score tier label
if _hi_val >= 70:
    _tier_label, _tier_color = "POTENTIALLY HABITABLE", "#00ff88"
elif _hi_val >= 40:
    _tier_label, _tier_color = "MARGINAL CONDITIONS", "#ffd044"
elif _hi_val >= 15:
    _tier_label, _tier_color = "UNLIKELY HABITABLE", "#ff8800"
else:
    _tier_label, _tier_color = "NOT HABITABLE", "#ff3300"

# Build a simple bar as an inline SVG progress gauge
_bar_pct   = int(_hi_val)
_bar_color = _hz_color

with _hi_col1:
    st.markdown(f"""
<div class="stat-card glass-in delay-2" style="border-color:{_hz_color}88;">
  <div class="stat-label">HABITABILITY INDEX</div>
  <div style="font-family:'Space Mono',monospace;font-size:2.1rem;font-weight:700;
              color:{_hz_color};text-shadow:0 0 22px {_hz_color}99;line-height:1.1;
              margin-top:4px;">
    {_hi_val:.1f}<span style="font-size:1rem;color:#00ccaa;"> / 100</span>
  </div>
  <div style="margin-top:8px;background:rgba(255,255,255,0.07);
              border-radius:4px;height:6px;overflow:hidden;">
    <div style="width:{_bar_pct}%;height:100%;background:linear-gradient(90deg,
         {_bar_color},{_bar_color}cc);border-radius:4px;
         box-shadow:0 0 8px {_bar_color}88;transition:width 0.8s ease;"></div>
  </div>
  <div style="font-size:0.64rem;color:{_tier_color};margin-top:5px;
              font-family:'Space Mono',monospace;letter-spacing:0.08em;">
    {_tier_label}
  </div>
</div>""", unsafe_allow_html=True)

with _hi_col2:
    st.markdown(f"""
<div class="stat-card glass-in delay-2">
  <div class="stat-label">ZONE CLASSIFICATION</div>
  <div style="font-family:'Space Mono',monospace;font-size:1.15rem;font-weight:700;
              color:{_hz_color};text-shadow:0 0 16px {_hz_color}88;margin-top:5px;">
    {_hz['zone_emoji']}  {_hz['zone_label']}
  </div>
  <div style="font-size:0.68rem;color:#00ccaa;margin-top:6px;
              font-family:'Space Mono',monospace;">
    a = {_effective_sma:.4f} AU<br>
    Flux = {_hz['flux_ratio']:.3f} S⊕
  </div>
</div>""", unsafe_allow_html=True)

with _hi_col3:
    _sb = _hz["score_breakdown"]
    st.markdown(f"""
<div class="stat-card glass-in delay-2">
  <div class="stat-label">SCORE BREAKDOWN</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.70rem;
              line-height:2.0;color:#b0cce8;margin-top:5px;">
    <span style="color:#00ffcc">HZ Position</span>
    <span style="float:right;color:#e8f4ff">{_sb['hz_position']:.1f}</span><br>
    <span style="color:#00ffcc">Planet Size</span>
    <span style="float:right;color:#e8f4ff">{_sb['planet_size']:.1f}</span><br>
    <span style="color:#00ffcc">Orbital Stability</span>
    <span style="float:right;color:#e8f4ff">{_sb['orbital']:.1f}</span><br>
    <div style="border-top:1px solid rgba(0,212,255,0.3);margin-top:3px;padding-top:3px;">
      <span style="color:#00ffff;font-weight:700;">TOTAL</span>
      <span style="float:right;color:{_hz_color};font-weight:700;">{_hi_val:.1f}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

with _hi_col4:
    st.markdown(f"""
<div class="stat-card glass-in delay-2" style="border-color:{_hz_color}44;">
  <div class="stat-label">GOLDILOCKS ZONE BOUNDARIES</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.70rem;
              line-height:2.1;color:#b0cce8;margin-top:5px;">
    <span style="color:#ffaa22">Opt. Inner (Recent Venus)</span>
    <span style="float:right;color:#e8f4ff">{_hz['hz_opt_inner_au']:.4f} AU</span><br>
    <span style="color:#88ffaa">Conservative Inner</span>
    <span style="float:right;color:#e8f4ff">{_hz['hz_inner_au']:.4f} AU</span><br>
    <span style="color:{_hz_color};font-weight:700">▶  Planet  ◀</span>
    <span style="float:right;color:{_hz_color};font-weight:700">{_effective_sma:.4f} AU</span><br>
    <span style="color:#88ffaa">Conservative Outer</span>
    <span style="float:right;color:#e8f4ff">{_hz['hz_outer_au']:.4f} AU</span><br>
    <span style="color:#4488ff">Opt. Outer (Early Mars)</span>
    <span style="float:right;color:#e8f4ff">{_hz['hz_opt_outer_au']:.4f} AU</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── Narrative description ─────────────────────────────────────────────────────
st.markdown(f"""
<div class='desc-text' style='margin-top:0.8rem;'>
  <b style='color:{_hz_color}'>{_hz['zone_emoji']}  {_hz['zone_label']} </b> —
  {_hz['description']}
  &nbsp; <span style='color:#556677;font-size:0.72rem;'>
    [ Kepler 3rd-law SMA estimate: {_sma_kepler_au:.4f} AU  ·
    Star mass est: {_star_mass_solar:.3f} M☉  ·
    Kopparapu et al. (2013) flux limits ]
  </span>
</div>""", unsafe_allow_html=True)

# ── Goldilocks Zone plot ──────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-3">04 · GOLDILOCKS ZONE DIAGRAM</div>',
            unsafe_allow_html=True)
st.markdown(f"""<div class='desc-text'>
  The <span style='color:#00ff88'>green band</span> is the Conservative Habitable Zone where liquid water
  can exist on the surface.  The <span style='color:#ffd044'>amber band</span> is the Optimistic HZ (Recent Venus → Early Mars).
  The <span style='color:{_hz_color}'>coloured dot</span> marks the detected planet at
  <b style='color:#00ffff'>{_effective_sma:.4f} AU</b> receiving
  <b style='color:#ffe66d'>{_hz['flux_ratio']:.3f}×</b> Earth's stellar flux.
</div>""", unsafe_allow_html=True)

with st.spinner("Rendering Goldilocks Zone diagram …"):
    fig_gz = plot_goldilocks_zone(
        hz_result           = _hz,
        semi_major_axis_au  = _effective_sma,
        star_luminosity_solar = star_luminosity_solar,
        planet_name         = st.session_state.star_name,
    )
st.pyplot(fig_gz, use_container_width=True); plt.close(fig_gz)

# ── Graph 1: Raw ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-2">05 · RAW LIGHT CURVE</div>',
            unsafe_allow_html=True)
st.markdown("""<div class='desc-text'>Raw stellar brightness over time.
The <span style='color:#ff6b6b'>red curve</span> is the Savitzky-Golay stellar trend —
slow variability that completely hides the tiny planet transits.</div>""",
            unsafe_allow_html=True)
fig_raw = plot_raw(raw_t, raw_f, raw_fe, trend_t, trend_f)
st.pyplot(fig_raw, use_container_width=True); plt.close(fig_raw)

# ── Graph 2: Flat ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-2">06 · CLEANED &amp; FLATTENED LIGHT CURVE</div>',
            unsafe_allow_html=True)
st.markdown(f"""<div class='desc-text'>Top: stellar trend removed.
Bottom: outlier spikes clipped. Noise floor ≈
<b style='color:#a8ff78'>{noise_ppm:.0f} ppm</b> — BLS search input.</div>""",
            unsafe_allow_html=True)
fig_flat = plot_flat(flat_t, flat_f, clean_t, clean_f)
st.pyplot(fig_flat, use_container_width=True); plt.close(fig_flat)

# ── Graph 3: BLS ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-3">07 · BLS PERIODOGRAM &amp; PHASE-FOLDED TRANSIT</div>',
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
    if st.button("🔄  SEARCH ANOTHER STAR", use_container_width=True):
        st.session_state.search_btn = False
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div class='status-text' style='text-align:center;padding:1rem 0 2rem;'>
  > {st.session_state.star_name.upper()} · {selected_mission} {time_label[0]}{time_segment}
  · {len(clean_t):,} CADENCES · {t_span:.1f} D · NOISE {noise_ppm:.0f} PPM
  · P = {best_period:.5f} D · DEPTH {best_depth*1e6:.0f} PPM
  · NASA MAST ARCHIVE
</div>""", unsafe_allow_html=True)