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

# set_page_config MUST be the absolute first Streamlit call in the file
st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="Exohunt.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    background-color: #02030a !important;
    background-image: url('BG.png');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    /* Three horizontal strips: EXOPLANET (top 33.33%), HUNTER (mid 33.33%), SPACE (bottom 33.33%) */
    background-image:
        url("data:image/png;base64,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"),
        url("data:image/png;base64,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"),
        url("data:image/png;base64,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") !important;
    background-size: 100% 33.34%, 100% 33.34%, 100% 33.34% !important;
    background-position: center top, center 33.33%, center bottom !important;
    background-repeat: no-repeat, no-repeat, no-repeat !important;
    background-attachment: scroll !important;
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