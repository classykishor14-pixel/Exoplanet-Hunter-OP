"""
==============================================================================
  EXOPLANET DETECTION SYSTEM — Streamlit Web Application
  app.py
==============================================================================

HOW TO RUN
----------
    pip install streamlit lightkurve astropy matplotlib numpy
    streamlit run app.py

The app will open automatically at http://localhost:8501
"""

import warnings
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import streamlit as st
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
import astropy.units as u

warnings.filterwarnings("ignore")

# ── BLS constants ─────────────────────────────────────────────────────────────
BLS_MIN_PERIOD   = 0.5
BLS_MAX_PERIOD   = 20.0
BLS_MIN_DURATION = 0.01
BLS_MAX_DURATION = 0.25
BLS_FREQ_FACTOR  = 10
N_BINS           = 60
SG_WINDOW        = 101
SG_POLY          = 3

# ── Plot theme ─────────────────────────────────────────────────────────────────
BG_DARK   = "#060a14"
BG_PANEL  = "#0a0f1e"
C_GRID    = "#1a2a44"
C_TICK    = "#6a85b0"
C_RAW     = "#7ecfff"
C_TREND   = "#ff6b6b"
C_FLAT    = "#ffe66d"
C_CLEAN   = "#a8ff78"
C_FOLD    = "#4a7cff"
C_BIN     = "#ffffff"
C_PEAK    = "#ff4f6e"
C_PERI    = "#00d4ff"
C_ANNO    = "#ffe66d"


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Exoplanet Hunter",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Exo+2:wght@300;400;600;800&display=swap');

  .stApp { background: #060a14; }
  section[data-testid="stSidebar"] {
      background: #0a0f1e !important;
      border-right: 1px solid #1e2d50;
  }
  html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; color: #c8d8f0; }

  .hero-title {
      font-family: 'Space Mono', monospace;
      font-size: 2.6rem;
      font-weight: 700;
      background: linear-gradient(135deg, #00d4ff 0%, #a8ff78 50%, #ffe66d 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -1px;
      line-height: 1.1;
      margin-bottom: 0.2rem;
  }
  .hero-sub {
      font-family: 'Exo 2', sans-serif;
      font-size: 0.95rem;
      color: #4a6a9a;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 2rem;
  }
  .sidebar-label {
      font-family: 'Space Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #4a6a9a;
      margin-bottom: 0.3rem;
  }
  .stat-grid { display: flex; gap: 12px; flex-wrap: wrap; margin: 1.2rem 0; }
  .stat-card {
      background: #0d1a2e;
      border: 1px solid #1e2d50;
      border-radius: 10px;
      padding: 14px 20px;
      flex: 1;
      min-width: 140px;
      position: relative;
      overflow: hidden;
  }
  .stat-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
  }
  .stat-card.blue::before  { background: #00d4ff; }
  .stat-card.green::before { background: #a8ff78; }
  .stat-card.gold::before  { background: #ffe66d; }
  .stat-card.pink::before  { background: #ff4f6e; }
  .stat-card.cyan::before  { background: #4a7cff; }
  .stat-label {
      font-family: 'Space Mono', monospace;
      font-size: 0.65rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: #4a6a9a;
      margin-bottom: 4px;
  }
  .stat-value {
      font-family: 'Space Mono', monospace;
      font-size: 1.25rem;
      font-weight: 700;
      color: #e8f4ff;
  }
  .stat-unit { font-size: 0.7rem; color: #4a6a9a; margin-left: 3px; }
  .section-header {
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: #4a6a9a;
      border-bottom: 1px solid #1e2d50;
      padding-bottom: 8px;
      margin: 2rem 0 1rem 0;
  }
  @keyframes fadeUp {
      from { opacity: 0; transform: translateY(18px); }
      to   { opacity: 1; transform: translateY(0); }
  }
  .animate-in { animation: fadeUp 0.55s ease both; }
  .delay-1 { animation-delay: 0.1s; }
  .delay-2 { animation-delay: 0.22s; }
  .delay-3 { animation-delay: 0.34s; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS — MATPLOTLIB THEME
# =============================================================================

def apply_dark_theme(ax):
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d50")
        sp.set_linewidth(0.8)
    ax.tick_params(colors=C_TICK, labelsize=8.5)
    ax.grid(True, linestyle=":", linewidth=0.35, color=C_GRID, alpha=0.9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))


def make_fig(w=13, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG_DARK)
    apply_dark_theme(ax)
    return fig, ax


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def clear_cache():
    for path in [Path.home() / ".lightkurve" / "cache",
                 Path.home() / ".lightkurve-cache"]:
        if path.exists():
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def fetch_and_clean(target: str, quarter: int):
    """
    Full Stage 1+2 pipeline cached by Streamlit.
    Returns (lc_raw_normed, lc_flat, lc_trend, lc_clean).
    Uses plain numpy arrays internally so Streamlit's cache serialiser works.
    """
    result = lk.search_lightcurve(
        target, mission="Kepler", quarter=quarter,
        cadence="long", author="Kepler",
    )
    if len(result) == 0:
        raise ValueError(
            f"No Kepler data found for '{target}' Q{quarter}. "
            "Check the star name (e.g. 'Kepler-22') or try a different quarter."
        )

    lc = result.download_all().stitch()

    cols_lower = [c.lower() for c in lc.columns]
    if "pdcsap_flux" in cols_lower:
        lc = lc.select_flux("pdcsap_flux")
    elif "sap_flux" in cols_lower:
        lc = lc.select_flux("sap_flux")

    lc_raw_normed = lc.remove_nans().normalize()
    lc_flat, lc_trend = lc_raw_normed.flatten(
        window_length=SG_WINDOW, polyorder=SG_POLY,
        return_trend=True, break_tolerance=5, niters=3, sigma=3,
    )
    lc_clean = lc_flat.remove_outliers(sigma=4.0, sigma_lower=1e5)
    return lc_raw_normed, lc_flat, lc_trend, lc_clean


@st.cache_data(show_spinner=False)
def run_bls_cached(time_arr, flux_arr, err_arr):
    """
    BLS on plain numpy arrays (hashable → cache-safe).
    Returns (periods_arr, power_arr, best_period, best_t0,
             best_duration, best_depth).
    """
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
        np.array(pg.period.value),
        np.array(pg.power),
        float(pg.period[best_idx].value),
        float(pg.transit_time[best_idx].value),
        float(pg.duration[best_idx].value),
        float(pg.depth[best_idx]),
    )


def phase_fold_arrays(time, flux, flux_err, period, t0):
    """Fold and bin using lightkurve, return plain arrays."""
    from astropy.time import Time
    time_obj = Time(time, format="bkjd")
    
    lc_tmp = lk.LightCurve(
        time=time_obj, 
        flux=flux, 
        flux_err=flux_err
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
            errs[i] = (1.48 * np.median(np.abs(fv[m] - meds[i]))
                       / np.sqrt(m.sum()))
    return fv, phase_hours, centres, meds, errs


# =============================================================================
# PLOT BUILDERS
# =============================================================================

def plot_raw(lc_raw, lc_trend):
    fig, ax = make_fig(h=4)
    t, f, fe = lc_raw.time.value, lc_raw.flux.value, lc_raw.flux_err.value

    ax.plot(t, f, color=C_RAW, lw=0.5, alpha=0.7, zorder=2, label="Raw flux")
    ax.fill_between(t, f - fe, f + fe, color=C_RAW, alpha=0.10, zorder=1)
    ax.plot(lc_trend.time.value, lc_trend.flux.value,
            color=C_TREND, lw=1.8, alpha=0.9, zorder=3,
            label=f"Savitzky-Golay trend  (window={SG_WINDOW}, poly={SG_POLY})")
    ax.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Time  [BKJD]", color=C_TICK, fontsize=9)
    ax.set_ylabel("Normalised Flux", color=C_RAW, fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.25,
              facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    ax.set_title("Raw Light Curve  +  Stellar Trend (red = what gets removed)",
                 color=C_TICK, fontsize=9, loc="left", pad=6)
    fig.tight_layout(pad=1.5)
    return fig


def plot_flat(lc_flat, lc_clean):
    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor(BG_DARK)
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    apply_dark_theme(ax1); apply_dark_theme(ax2)

    ax1.plot(lc_flat.time.value, lc_flat.flux.value,
             color=C_FLAT, lw=0.5, alpha=0.65, zorder=2,
             label="Flattened flux  (stellar trend removed)")
    ax1.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Flux", color=C_FLAT, fontsize=9)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.25,
               facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    ax1.set_title("Flattened  (top)  →  After outlier removal  (bottom)",
                  color=C_TICK, fontsize=9, loc="left", pad=6)

    ax2.plot(lc_clean.time.value, lc_clean.flux.value,
             color=C_CLEAN, lw=0.5, alpha=0.65, zorder=2,
             label="Planet-search ready  (outliers removed)")
    ax2.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    ax2.set_xlabel("Time  [BKJD]", color=C_TICK, fontsize=9)
    ax2.set_ylabel("Flux", color=C_CLEAN, fontsize=9)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.25,
               facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    noise = float(np.std(lc_clean.flux.value)) * 1e6
    ax2.text(0.01, 0.08, f"Noise floor ≈ {noise:.0f} ppm",
             transform=ax2.transAxes, fontsize=7.5, color=C_CLEAN, style="italic")

    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    fig.tight_layout(pad=1.5)
    return fig


def plot_bls(periods, power, lc_clean,
             best_period, best_t0, best_duration, best_depth):
    dur_h       = best_duration * 24.0
    half_dur_h  = dur_h / 2.0
    margin      = max(best_depth * 6, 0.0015)

    fold_flux, phase_hours, bin_c, bin_m, bin_e = phase_fold_arrays(
        lc_clean.time.value, lc_clean.flux.value, lc_clean.flux_err.value,
        best_period, best_t0,
    )

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG_DARK)
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.42,
                              top=0.90, bottom=0.07, left=0.07, right=0.97)
    top_ax  = fig.add_subplot(outer[0])
    bot     = gridspec.GridSpecFromSubplotSpec(
                  1, 2, subplot_spec=outer[1],
                  wspace=0.32, width_ratios=[1, 1.1])
    fold_ax = fig.add_subplot(bot[0])
    zoom_ax = fig.add_subplot(bot[1])
    for ax in (top_ax, fold_ax, zoom_ax):
        apply_dark_theme(ax)

    # ── Periodogram ──────────────────────────────────────────────────────────
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
        top_ax.text(harmonic, peak_pwr * 0.04, " P/2",
                    color="#444", fontsize=7, va="bottom")
    top_ax.set_xlabel("Trial Period  [days]", color=C_TICK, fontsize=9.5)
    top_ax.set_ylabel("BLS Power", color=C_PERI, fontsize=9.5)
    top_ax.set_xlim(BLS_MIN_PERIOD, BLS_MAX_PERIOD)
    top_ax.set_ylim(0, peak_pwr * 1.18)
    top_ax.legend(loc="upper right", fontsize=8.5, framealpha=0.25,
                  facecolor=BG_PANEL, edgecolor="#1e2d50", labelcolor="white")
    top_ax.set_title("① BLS Periodogram — tallest spike = planet orbital period",
                     color=C_TICK, fontsize=9, pad=6, loc="left")

    # ── Full fold ─────────────────────────────────────────────────────────────
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
    fold_ax.set_title("② Full Phase-Folded Curve",
                      color=C_TICK, fontsize=9, pad=5, loc="left")

    # ── Zoomed transit ────────────────────────────────────────────────────────
    zoom_h = dur_h * 4
    zs = np.abs(phase_hours) <= zoom_h
    zb = valid & (np.abs(bin_c) <= zoom_h)
    bottom_y = 1.0 - best_depth
    arr_x    = dur_h * 1.9

    zoom_ax.scatter(phase_hours[zs], fold_flux[zs],
                    color=C_FOLD, s=5, alpha=0.45, zorder=2)
    zoom_ax.errorbar(bin_c[zb], bin_m[zb], yerr=bin_e[zb],
                     fmt="o", color=C_BIN, ms=5, lw=1.0,
                     elinewidth=0.9, capsize=2.5, zorder=4)
    zoom_ax.axvspan(-half_dur_h, half_dur_h,
                    color=C_PEAK, alpha=0.12, zorder=1)
    zoom_ax.axhline(1.0, color="#334", lw=0.7, ls="--", alpha=0.5)
    zoom_ax.annotate(
        "", xy=(arr_x, bottom_y), xytext=(arr_x, 1.0),
        arrowprops=dict(arrowstyle="<->", color=C_ANNO, lw=1.3, mutation_scale=10),
    )
    zoom_ax.text(arr_x * 1.05, (1.0 + bottom_y) / 2,
                 f"Δ = {best_depth*1e6:.0f} ppm",
                 color=C_ANNO, fontsize=8, va="center")
    zoom_ax.annotate(
        "", xy=(-half_dur_h, bottom_y * 0.9999),
        xytext=(half_dur_h, bottom_y * 0.9999),
        arrowprops=dict(arrowstyle="<->", color="#88aaff", lw=1.0, mutation_scale=8),
    )
    zoom_ax.text(0, bottom_y - margin * 0.35,
                 f"Duration = {dur_h:.2f} h",
                 color="#88aaff", fontsize=7.5, ha="center")
    zoom_ax.set_xlim(-zoom_h, zoom_h)
    zoom_ax.set_ylim(1.0 - margin, 1.0 + margin * 0.4)
    zoom_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    zoom_ax.set_xlabel("Phase  [hours]", color=C_TICK, fontsize=9.5)
    zoom_ax.set_ylabel("Normalised Flux", color=C_FOLD, fontsize=9.5)
    zoom_ax.set_title("③ Zoomed Transit  — U-shaped dip",
                      color=C_TICK, fontsize=9, pad=5, loc="left")

    fig.suptitle("BLS Planet Detection", color="white",
                 fontsize=13, fontweight="bold", y=0.96)
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.6rem'>
      <div style='font-family:Space Mono,monospace;font-size:1.05rem;
                  color:#00d4ff;font-weight:700;letter-spacing:-0.5px;'>
        🔭 EXOPLANET<br>HUNTER
      </div>
      <div style='font-size:0.7rem;color:#2a4a6a;
                  text-transform:uppercase;letter-spacing:0.12em;margin-top:4px;'>
        Kepler Mission · BLS Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Target Star</div>',
                unsafe_allow_html=True)
    star_name = st.text_input(
        label="star", value="Kepler-10",
        label_visibility="collapsed",
        placeholder="e.g. Kepler-10, Kepler-22 …",
    )

    st.markdown('<div class="sidebar-label" style="margin-top:14px">'
                'Kepler Quarter (0–17)</div>', unsafe_allow_html=True)
    quarter = st.slider("q", min_value=0, max_value=17,
                        value=3, label_visibility="collapsed")

    search_btn = st.button("⚡  SEARCH FOR PLANETS",
                           use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#2a4a6a;line-height:1.9;'>
      <b style='color:#3a5a8a'>Detection Pipeline</b><br>
      ① Download from MAST<br>
      ② NaN removal + normalise<br>
      ③ Savitzky-Golay flatten<br>
      ④ Outlier sigma-clip  (4σ)<br>
      ⑤ BLS periodogram<br>
      ⑥ Phase-fold &amp; bin
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.7rem;color:#1e3a5a;line-height:1.7;'>
      Try other stars:<br>
      <span style='color:#2a5a8a'>Kepler-22 · Kepler-16<br>
      Kepler-62 · Kepler-186</span><br><br>
      Data: NASA MAST Archive<br>
      Powered by lightkurve + astropy
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN PAGE
# =============================================================================

st.markdown('<div class="hero-title animate-in">EXOPLANET HUNTER</div>',
            unsafe_allow_html=True)
st.markdown('<div class="hero-sub animate-in delay-1">'
            'Kepler Space Telescope · Box Least Squares Detection Engine'
            '</div>', unsafe_allow_html=True)

# ── Landing state ─────────────────────────────────────────────────────────────
if not search_btn:
    c1, c2, c3 = st.columns(3)
    for col, color, label, value in [
        (c1, "blue",  "MISSION",     "Kepler Space Telescope"),
        (c2, "green", "METHOD",      "Box Least Squares (BLS)"),
        (c3, "gold",  "DATA SOURCE", "NASA MAST Archive"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card {color} animate-in delay-2">
              <div class="stat-label">{label}</div>
              <div style='font-family:Space Mono,monospace;font-size:0.9rem;
                          color:#e8f4ff;margin-top:4px;'>{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:3rem;text-align:center;color:#1e3a5a;
                font-family:Space Mono,monospace;font-size:0.8rem;'>
      ← Enter a star name in the sidebar and click SEARCH
    </div>""", unsafe_allow_html=True)
    st.stop()


# =============================================================================
# RUN PIPELINE
# =============================================================================

st.markdown(f"""
<div class="animate-in" style='font-family:Space Mono,monospace;
     font-size:0.8rem;color:#4a6a9a;margin-bottom:1rem;'>
  ANALYSING  <span style='color:#00d4ff'>{star_name.upper()}</span>
  &nbsp;·&nbsp; QUARTER {quarter}
</div>""", unsafe_allow_html=True)

# Step 1-2: Fetch + clean
with st.spinner("📡 Contacting NASA MAST archive and cleaning data …"):
    try:
        lc_raw, lc_flat, lc_trend, lc_clean = fetch_and_clean(star_name, quarter)
    except Exception as e:
        st.error(
            f"**Download failed:** {e}\n\n"
            "**Common fixes:**\n"
            "- Check the star name (e.g. `Kepler-22`, `Kepler-16`)\n"
            "- Try a different quarter (0–17)\n"
            "- If you see an **SSL error**, your ISP is blocking NASA's server "
            "— enable a VPN and try again"
        )
        st.stop()

# Step 3: BLS
with st.spinner("🔍 Running BLS periodogram — searching all periods "
                f"{BLS_MIN_PERIOD}–{BLS_MAX_PERIOD} d (30–60 s) …"):
    try:
        periods, power, best_period, best_t0, best_duration, best_depth = \
            run_bls_cached(
                np.array(lc_clean.time.value),
                np.array(lc_clean.flux.value),
                np.array(lc_clean.flux_err.value),
            )
    except Exception as e:
        st.error(f"BLS failed: {e}")
        st.stop()

# Derived stats
t_span     = lc_clean.time.value.max() - lc_clean.time.value.min()
n_transits = int(np.floor(t_span / best_period))
noise_ppm  = float(np.std(lc_clean.flux.value)) * 1e6
snr        = float(power.max()) / float(np.median(power)) if np.median(power) > 0 else 0.0

# =============================================================================
# STAT CARDS
# =============================================================================

st.markdown('<div class="section-header animate-in delay-1">'
            '★ Detected Planet Parameters</div>', unsafe_allow_html=True)

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
        <div class="stat-card {color} animate-in delay-2">
          <div class="stat-label">{label}</div>
          <div class="stat-value">{value}
            <span class="stat-unit">{unit}</span>
          </div>
        </div>""", unsafe_allow_html=True)

# =============================================================================
# GRAPH 1 — RAW LIGHT CURVE
# =============================================================================

st.markdown('<div class="section-header animate-in delay-2">'
            '01 · Raw Light Curve</div>', unsafe_allow_html=True)
st.markdown("""
<div style='font-size:0.82rem;color:#2a4a6a;margin-bottom:0.8rem;'>
Raw stellar brightness over time. The <span style='color:#ff6b6b'>red curve</span>
is the Savitzky-Golay fitted stellar trend — slow variability that completely hides
the tiny planet transits beneath it.
</div>""", unsafe_allow_html=True)

fig_raw = plot_raw(lc_raw, lc_trend)
st.pyplot(fig_raw, use_container_width=True)
plt.close(fig_raw)

# =============================================================================
# GRAPH 2 — CLEANED & FLATTENED
# =============================================================================

st.markdown('<div class="section-header animate-in delay-2">'
            '02 · Cleaned &amp; Flattened Light Curve</div>', unsafe_allow_html=True)
st.markdown(f"""
<div style='font-size:0.82rem;color:#2a4a6a;margin-bottom:0.8rem;'>
Top panel: after dividing out the stellar trend. Bottom panel: after
sigma-clipping outlier spikes (flares &amp; cosmic rays).
Noise floor ≈ <b style='color:#a8ff78'>{noise_ppm:.0f} ppm</b> —
this is the input to the planet-search engine.
</div>""", unsafe_allow_html=True)

fig_flat = plot_flat(lc_flat, lc_clean)
st.pyplot(fig_flat, use_container_width=True)
plt.close(fig_flat)

# =============================================================================
# GRAPH 3 — BLS + FOLDED TRANSIT
# =============================================================================

st.markdown('<div class="section-header animate-in delay-3">'
            '03 · BLS Periodogram &amp; Phase-Folded Transit</div>',
            unsafe_allow_html=True)
st.markdown(f"""
<div style='font-size:0.82rem;color:#2a4a6a;margin-bottom:0.8rem;'>
The BLS algorithm tested every period from {BLS_MIN_PERIOD}–{BLS_MAX_PERIOD} days.
The tallest spike = the planet's orbital period
(<b style='color:#ffe66d'>P = {best_period:.5f} d</b>).
The bottom panels show all {n_transits} transits stacked on top of each other —
the clean U-shaped dip is the shadow of the exoplanet crossing its star.
</div>""", unsafe_allow_html=True)

with st.spinner("Rendering BLS detection plot …"):
    fig_bls = plot_bls(periods, power, lc_clean,
                       best_period, best_t0, best_duration, best_depth)
st.pyplot(fig_bls, use_container_width=True)
plt.close(fig_bls)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align:center;font-family:Space Mono,monospace;
     font-size:0.7rem;color:#1e3a5a;padding:1rem 0 2rem;'>
  {star_name.upper()} · Q{quarter} ·
  {len(lc_clean):,} cadences ·
  {t_span:.1f} d baseline ·
  Noise {noise_ppm:.0f} ppm ·
  Period {best_period:.5f} d ·
  Depth {best_depth*1e6:.0f} ppm ·
  Data: NASA MAST / Kepler Mission
</div>""", unsafe_allow_html=True)
# --- PERSONAL BRANDING & CLASSIFICATION ---
st.sidebar.markdown("---")
st.sidebar.write("🚀 **Developed by Kishore (Kai)**")
st.sidebar.write("Target: MIT Class of 2031 | Astrophysics & AeroEng")

# This part assumes you have a variable named 'planet_radius_earth' 
# If not, we calculate it here (Star Radius * sqrt(depth) * 109.2)
try:
    # Most lightkurve objects have star metadata
    st_radius = lc_clean.meta.get('RADIUS', 1.0) 
    # 'depth' comes from your BLS result
    p_radius = np.sqrt(results.transit_depth[0]) * st_radius * 109.2
    
    st.sidebar.subheader("🪐 Planet Stats")
    st.sidebar.metric("Estimated Size", f"{p_radius:.2f} x Earth")

    # Classification Logic
    if p_radius < 1.25:
        st.sidebar.success("Category: Rocky Planet (Earth-like size)")
    elif 1.25 <= p_radius < 2.0:
        st.sidebar.info("Category: Super-Earth")
    elif 2.0 <= p_radius < 4.0:
        st.sidebar.warning("Category: Mini-Neptune (Gas/Water World)")
    else:
        st.sidebar.error("Category: Gas Giant (Jupiter-class)")
        
except Exception as e:
    st.sidebar.write("Search for a star to see Planet Stats")

    # --- PERSONAL BRANDING & CLASSIFICATION ---
st.sidebar.markdown("---")
st.sidebar.write("🚀 **Developed by Kishore (Kai)**")
st.sidebar.write("Target: MIT Class of 2031 | Astrophysics & AeroEng")

# Planet Classifier Logic
try:
    # This pulls the star radius from the NASA data metadata
    st_radius = lc_clean.meta.get('RADIUS', 1.0) 
    
    # This calculates the radius based on how much light the planet blocks
    # results.transit_depth[0] is the depth of the tallest peak in the BLS periodogram
    p_radius = np.sqrt(results.transit_depth[0]) * st_radius * 109.2
    
    st.sidebar.subheader("🪐 Planet Stats")
    st.sidebar.metric("Estimated Size", f"{p_radius:.2f} x Earth")

    # Classification logic for your portfolio
    if p_radius < 1.25:
        st.sidebar.success("Category: Rocky Planet")
    elif 1.25 <= p_radius < 2.0:
        st.sidebar.info("Category: Super-Earth")
    elif 2.0 <= p_radius < 4.0:
        st.sidebar.warning("Category: Mini-Neptune")
    else:
        st.sidebar.error("Category: Gas Giant")
        
except:
    # This shows before the user hits "Search"
    st.sidebar.info("Search for a star to see Planet Stats")