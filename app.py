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
    top: -8%;
    left: -8%;
    width: 116%;
    height: 116%;
    z-index: -2;
    pointer-events: none;
    animation: cosmicZoom 28s ease-in-out infinite alternate;
    will-change: transform;
    background-color: #02030a !important;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.68' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23g)' opacity='0.04'/%3E%3C/svg%3E"),
        radial-gradient(ellipse 120% 120% at 50% 50%, transparent 28%, rgba(1,2,10,0.80) 72%, rgba(1,2,8,0.97) 100%),
        radial-gradient(ellipse 80% 65% at 52% 48%, rgba(8,18,55,0.55) 0%, transparent 70%),
        radial-gradient(ellipse 52% 38% at 82% 82%, rgba(210,95,12,0.30) 0%, rgba(160,55,8,0.14) 45%, transparent 72%),
        radial-gradient(ellipse 48% 42% at 80% 14%, rgba(0,185,230,0.34) 0%, rgba(0,100,175,0.16) 50%, transparent 78%),
        radial-gradient(ellipse 68% 58% at 16% 52%, rgba(105,22,165,0.44) 0%, rgba(55,10,95,0.22) 48%, transparent 78%),
        linear-gradient(158deg, #04060f 0%, #020309 55%, #030509 100%) !important;
    background-size: 400px 400px, 100% 100%, 100% 100%, 100% 100%, 100% 100%, 100% 100%, 100% 100% !important;
    background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat, no-repeat !important;
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
        radial-gradient(1.0px 1.0px at 73%  43%, rgba(200,240,255,0.42) 0%, transparent 100%),
        radial-gradient(0.8px 0.8px at 91%  33%, rgba(255,220,180,0.48) 0%, transparent 100%),
        radial-gradient(1.1px 1.1px at 47%  67%, rgba(180,200,255,0.44) 0%, transparent 100%),
        radial-gradient(0.9px 0.9px at 63%  88%, rgba(255,255,210,0.52) 0%, transparent 100%);
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
   BASE CONTAINERS — transparent so the animated background shows through
════════════════════════════════════════════════════════════════════════════ */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stHeader"], [data-testid="stBottom"],
[data-testid="stDecoration"], section[data-testid="stSidebar"] ~ div, .main {
    background: transparent !important;
    background-color: transparent !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — MAIN CONTENT PANEL
════════════════════════════════════════════════════════════════════════════ */
.main .block-container {
    background: rgba(3, 7, 22, 0.46) !important;
    backdrop-filter:          blur(22px) saturate(170%) brightness(0.95) !important;
    -webkit-backdrop-filter:  blur(22px) saturate(170%) brightness(0.95) !important;
    border-radius: 20px !important;
    padding: 2rem 2.5rem !important;
    margin-top: 0.6rem !important;
    border: 1px solid rgba(0, 212, 255, 0.15) !important;
    box-shadow:
        0 12px 48px rgba(0, 0, 0, 0.60),
        inset 0  1px 0 rgba(255, 255, 255, 0.07),
        inset 0 -1px 0 rgba(0, 212, 255, 0.1) !important;
}

/* ════════════════════════════════════════════════════════════════════════════
   GLASSMORPHISM — SIDEBAR
════════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: rgba(3, 7, 22, 0.72) !important;
    backdrop-filter:         blur(28px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(180%) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.14) !important;
    box-shadow:
        4px 0 36px rgba(0, 0, 0, 0.60),
        inset -1px 0 0 rgba(0, 212, 255, 0.08) !important;
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
    background: rgba(5, 12, 30, 0.62) !important;
    backdrop-filter:         blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(0, 212, 255, 0.22) !important;
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
════════════════════════════════════════════════════════════════════════════ */
.stat-card {
    background:              rgba(5, 12, 32, 0.58);
    backdrop-filter:         blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
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
    background:              rgba(4, 11, 30, 0.75) !important;
    backdrop-filter:         blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
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
    background:              rgba(0, 212, 255, 0.07) !important;
    backdrop-filter:         blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
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
    background:              rgba(8, 20, 50, 0.58) !important;
    backdrop-filter:         blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
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
    background: rgba(5, 12, 32, 0.60) !important;
    backdrop-filter: blur(10px) !important;
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
</style>

<!-- KEN BURNS DRIFT LAYER — the animated cosmic background -->
<div class="cosmic-drift"></div>
<div class="star-drift"></div>
<!-- Optional CRT scanline effect (uncomment to enable) -->
<!-- <div class="scanline"></div> -->
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
        min_value=0.1, max_value=100.0, value=1.0, step=0.05,
        format="%.2f",
        label_visibility="collapsed",
        help="Host star radius in solar radii — used to convert transit depth → planet radius.",
    )
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
        min_value=0.01, max_value=5000.0, value=1.0, step=0.5,
        format="%.2f",
        label_visibility="collapsed",
        help="Planet mass in Earth masses (M⊕). Use radial-velocity or TTV data, or keep default for a rough estimate.",
    )
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
        min_value=0.0001, max_value=1_000_000.0, value=1.0, step=0.1,
        format="%.4f",
        label_visibility="collapsed",
        help="Host star luminosity in solar units (L☉). Sun = 1.0. "
             "Red dwarfs ≈ 0.001–0.08; F-type ≈ 2–5; giants > 100.",
    )
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
        min_value=0.001, max_value=500.0, value=1.0, step=0.01,
        format="%.4f",
        label_visibility="collapsed",
        help="Orbital semi-major axis in Astronomical Units (AU). "
             "Earth = 1.0 AU. Derived from BLS period via Kepler's 3rd law if left at default.",
    )
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
    st.session_state.star_name = st.text_input(
        "star_input",
        value=st.session_state.star_name,
        placeholder="> ENTER_STAR_DESIGNATION (e.g. Kepler-10, TRAPPIST-1)",
        label_visibility="collapsed",
    )

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
<div class="status-text animate-in" style='margin-bottom:1.2rem;'>
  > ANALYSING &nbsp;<span style='color:#00ffff'>{st.session_state.star_name.upper()}</span>
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

_planet_radius_earth = estimate_planet_radius_earth(best_depth, star_radius_solar)
_comp = get_planet_composition(planet_mass_earth, _planet_radius_earth)

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
# Kepler's 3rd law: a (AU) = (P_days / 365.25)^(2/3) * M_star^(1/3)
# We assume M_star ≈ L_star^0.25 (main-sequence mass–luminosity relation)
_star_mass_solar = star_luminosity_solar ** 0.25
_sma_kepler_au   = (best_period / 365.25) ** (2.0 / 3.0) * _star_mass_solar ** (1.0 / 3.0)

# If the user left the default (1.0 AU) use the BLS-derived estimate
_effective_sma = semi_major_axis_au if abs(semi_major_axis_au - 1.0) > 0.001 \
                 else _sma_kepler_au

_hz = calculate_habitability_index(
    semi_major_axis_au    = _effective_sma,
    star_luminosity_solar = star_luminosity_solar,
    planet_radius_earth   = _planet_radius_earth,
    planet_mass_earth     = planet_mass_earth,
)

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
      <!-- progress bar -->
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