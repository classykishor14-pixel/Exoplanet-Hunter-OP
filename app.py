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
  • Planet composition classification (Gas Giant/Super-Earth/Rocky/Lava World)
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

# ── Planetary constants for composition classification ───────────────────────
# Density thresholds in g/cm³
DENSITY_GAS_GIANT      = 1.5   # < 1.5 g/cm³ → Gas Giant
DENSITY_SUPER_EARTH     = 4.0   # 1.5–4.0 g/cm³ → Super-Earth
DENSITY_ROCKY          = 6.0   # 4.0–6.0 g/cm³ → Rocky
# > 6.0 g/cm³ → Lava World (Iron-rich/Mercury-like)

# Earth values for reference (not used directly but helpful for understanding)
EARTH_RADIUS_KM = 6371.0
EARTH_MASS_KG   = 5.972e24
EARTH_DENSITY   = 5.514  # g/cm³

# Jupiter values
JUPITER_RADIUS_KM = 69911.0
JUPITER_MASS_KG   = 1.898e27
JUPITER_DENSITY   = 1.326  # g/cm³

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

# Planet composition colors for UI
PLANET_COLORS = {
    "Gas Giant": "#ff6b9d",
    "Super-Earth": "#4ecdc4",
    "Rocky": "#ffe66d",
    "Lava World": "#ff4757",
    "Unknown": "#dfe6e9"
}


# =============================================================================
# ░░░  PLANET COMPOSITION CLASSIFICATION FUNCTION  ░░░
# =============================================================================

def get_planet_composition(planet_mass, planet_radius, mass_unit="earth", radius_unit="earth"):
    """
    Classify a planet as 'Gas Giant', 'Super-Earth', 'Rocky', or 'Lava World'
    based on its density using NASA Exoplanet Archive logic.
    
    Parameters:
    -----------
    planet_mass : float
        Planet mass value
    planet_radius : float
        Planet radius value
    mass_unit : str
        Unit of mass ('earth' for Earth masses, 'jupiter' for Jupiter masses, 'kg' for kg)
    radius_unit : str
        Unit of radius ('earth' for Earth radii, 'jupiter' for Jupiter radii, 'km' for km)
    
    Returns:
    --------
    dict : {
        'classification': str,  # One of: Gas Giant, Super-Earth, Rocky, Lava World
        'density': float,       # Density in g/cm³
        'confidence': str,      # Confidence level based on data quality
        'description': str      # Brief scientific description
    }
    
    NASA Exoplanet Archive Classification Logic:
    --------------------------------------------
    - Gas Giant:   Density < 1.5 g/cm³ (like Jupiter, Saturn)
    - Super-Earth: Density 1.5–4.0 g/cm³ (intermediate, likely water-rich)
    - Rocky:       Density 4.0–6.0 g/cm³ (Earth-like, silicate composition)
    - Lava World:  Density > 6.0 g/cm³ (Iron-rich, Mercury-like, extreme surface)
    """
    
    # Conversion factors to SI units (kg, m)
    # Earth values
    EARTH_MASS_KG = 5.972e24
    EARTH_RADIUS_M = 6.371e6
    
    # Jupiter values
    JUPITER_MASS_KG = 1.898e27
    JUPITER_RADIUS_M = 6.9911e7
    
    # Convert to kg and meters
    if mass_unit.lower() == "earth":
        mass_kg = planet_mass * EARTH_MASS_KG
    elif mass_unit.lower() == "jupiter":
        mass_kg = planet_mass * JUPITER_MASS_KG
    elif mass_unit.lower() == "kg":
        mass_kg = planet_mass
    else:
        raise ValueError(f"Unknown mass unit: {mass_unit}. Use 'earth', 'jupiter', or 'kg'")
    
    if radius_unit.lower() == "earth":
        radius_m = planet_radius * EARTH_RADIUS_M
    elif radius_unit.lower() == "jupiter":
        radius_m = planet_radius * JUPITER_RADIUS_M
    elif radius_unit.lower() == "km":
        radius_m = planet_radius * 1000
    else:
        raise ValueError(f"Unknown radius unit: {radius_unit}. Use 'earth', 'jupiter', or 'km'")
    
    # Calculate volume (assuming sphere) in m³
    volume_m3 = (4.0/3.0) * np.pi * (radius_m ** 3)
    
    # Calculate density in kg/m³
    density_kg_m3 = mass_kg / volume_m3
    
    # Convert to g/cm³ (1 g/cm³ = 1000 kg/m³)
    density_g_cm3 = density_kg_m3 / 1000.0
    
    # Classify based on density thresholds
    if density_g_cm3 < DENSITY_GAS_GIANT:
        classification = "Gas Giant"
        confidence = "High" if density_g_cm3 < 0.8 else "Medium"
        description = (
            "Low-density planet primarily composed of hydrogen and helium. "
            "Similar to Jupiter and Saturn. Likely has no solid surface."
        )
    elif density_g_cm3 < DENSITY_SUPER_EARTH:
        classification = "Super-Earth"
        confidence = "Medium"
        description = (
            "Intermediate density planet. May have a thick atmosphere and "
            "potentially water-rich composition. Larger than Earth but smaller than Neptune."
        )
    elif density_g_cm3 < DENSITY_ROCKY:
        classification = "Rocky"
        confidence = "High"
        description = (
            "Earth-like density. Composed primarily of silicates and metals. "
            "Likely has a solid surface and potentially a thin atmosphere."
        )
    else:
        classification = "Lava World"
        confidence = "Medium" if density_g_cm3 < 10.0 else "High"
        description = (
            "Iron-rich, ultra-dense planet. Mercury-like composition. "
            "Extreme surface temperatures, likely covered in molten lava."
        )
    
    return {
        'classification': classification,
        'density': round(density_g_cm3, 3),
        'confidence': confidence,
        'description': description,
        'mass_earth': round(mass_kg / EARTH_MASS_KG, 2),
        'radius_earth': round(radius_m / EARTH_RADIUS_M, 2)
    }


def estimate_planet_mass_from_depth(depth_ppm, star_mass_solar=1.0, star_radius_solar=1.0):
    """
    Estimate planet radius from transit depth, then estimate mass based on density assumption.
    
    Parameters:
    -----------
    depth_ppm : float
        Transit depth in parts per million
    star_mass_solar : float
        Star mass in solar masses (default 1.0)
    star_radius_solar : float
        Star radius in solar radii (default 1.0)
    
    Returns:
    --------
    tuple : (radius_earth, mass_estimate_earth)
        Estimated radius (Earth radii) and mass (Earth masses)
    """
    # Solar radius in km
    SOLAR_RADIUS_KM = 695700.0
    EARTH_RADIUS_KM = 6371.0
    
    # Transit depth = (Rp/Rs)²
    # So Rp = Rs * sqrt(depth)
    depth_ratio = depth_ppm / 1e6
    star_radius_km = star_radius_solar * SOLAR_RADIUS_KM
    planet_radius_km = star_radius_km * np.sqrt(depth_ratio)
    planet_radius_earth = planet_radius_km / EARTH_RADIUS_KM
    
    # Estimate mass based on typical density (5.5 g/cm³ for rocky planets)
    # If radius < 2 Earth radii, assume rocky (density ~5.5)
    # If radius between 2-4, assume super-earth (density ~3.5)
    # If radius > 4, assume gas giant (density ~1.3)
    if planet_radius_earth < 2.0:
        assumed_density = 5.5  # g/cm³ (Earth-like)
    elif planet_radius_earth < 4.0:
        assumed_density = 3.5  # g/cm³ (Super-Earth)
    else:
        assumed_density = 1.3  # g/cm³ (Gas Giant)
    
    # Mass = density * volume
    # Planet mass in Earth masses = (density / Earth_density) * (R/R_Earth)³
    planet_mass_earth = (assumed_density / EARTH_DENSITY) * (planet_radius_earth ** 3)
    
    return planet_radius_earth, planet_mass_earth


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
    """Phase-fold and bin. Uses astropy.time.Time directly."""
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
    st.markdown("""
    <div style='font-size:0.71rem;color:#00ccaa;line-height:2.0;'>
      <b style='color:#00ffff'>Detection Pipeline</b><br>
      ① Download from MAST<br>
      ② NaN removal + normalise<br>
      ③ Savitzky-Golay flatten<br>
      ④ Outlier sigma-clip (4σ)<br>
      ⑤ BLS periodogram<br>
      ⑥ Phase-fold &amp; bin<br>
      ⑦ Planet composition analysis
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

# ── Estimate planet radius and mass from transit depth ───────────────────────
# Assuming a Sun-like star for estimation (user could adjust in future)
planet_radius_earth, planet_mass_earth = estimate_planet_mass_from_depth(best_depth)

# ── Get planet composition using NASA Exoplanet Archive logic ─────────────────
composition = get_planet_composition(
    planet_mass_earth, 
    planet_radius_earth, 
    mass_unit="earth", 
    radius_unit="earth"
)

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

# ── Planet composition card (special highlight) ──────────────────────────────
st.markdown('<div class="section-header animate-in delay-2">🌍  PLANET COMPOSITION ANALYSIS (NASA Exoplanet Archive Standard)</div>',
            unsafe_allow_html=True)

# Display composition in a beautiful card
comp_color = PLANET_COLORS.get(composition['classification'], "#ffffff")
st.markdown(f"""
<div style='background: rgba(5, 12, 32, 0.65); backdrop-filter: blur(20px); 
            border: 1px solid rgba(0, 212, 255, 0.3); border-radius: 16px; 
            padding: 1.2rem 1.8rem; margin: 0.5rem 0 1rem 0;'>
  <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;'>
    <div>
      <div style='font-family: Space Mono, monospace; font-size: 0.7rem; letter-spacing: 0.2em; 
                  color: #00ccaa; text-transform: uppercase;'>Classification</div>
      <div style='font-family: Space Mono, monospace; font-size: 2.2rem; font-weight: 700; 
                  color: {comp_color}; text-shadow: 0 0 20px rgba(0,212,255,0.3);'>
        {composition['classification']}
      </div>
    </div>
    <div>
      <div style='font-family: Space Mono, monospace; font-size: 0.7rem; letter-spacing: 0.2em; 
                  color: #00ccaa;'>Density</div>
      <div style='font-family: Space Mono, monospace; font-size: 1.5rem; font-weight: 700; 
                  color: #e8f4ff;'>
        {composition['density']} <span style='font-size: 0.9rem;'>g/cm³</span>
      </div>
    </div>
    <div>
      <div style='font-family: Space Mono, monospace; font-size: 0.7rem; letter-spacing: 0.2em; 
                  color: #00ccaa;'>Est. Radius</div>
      <div style='font-family: Space Mono, monospace; font-size: 1.5rem; font-weight: 700; 
                  color: #e8f4ff;'>
        {composition['radius_earth']} <span style='font-size: 0.9rem;'>R_Earth</span>
      </div>
    </div>
    <div>
      <div style='font-family: Space Mono, monospace; font-size: 0.7rem; letter-spacing: 0.2em; 
                  color: #00ccaa;'>Est. Mass</div>
      <div style='font-family: Space Mono, monospace; font-size: 1.5rem; font-weight: 700; 
                  color: #e8f4ff;'>
        {composition['mass_earth']} <span style='font-size: 0.9rem;'>M_Earth</span>
      </div>
    </div>
  </div>
  <div style='margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(0,212,255,0.2);'>
    <div style='font-family: Space Mono, monospace; font-size: 0.75rem; color: #b8d0ff; line-height: 1.5;'>
      📖 {composition['description']}
    </div>
    <div style='margin-top: 8px;'>
      <span style='background: rgba(0,212,255,0.15); padding: 2px 8px; border-radius: 12px; 
                   font-family: monospace; font-size: 0.65rem; color: #00ffff;'>
        Confidence: {composition['confidence']}
      </span>
      <span style='background: rgba(0,212,255,0.1); padding: 2px 8px; border-radius: 12px; 
                   font-family: monospace; font-size: 0.65rem; margin-left: 8px;'>
        Δ depth = {best_depth*1e6:.0f} ppm
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Graph 1: Raw ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-3">01 · RAW LIGHT CURVE</div>',
            unsafe_allow_html=True)
st.markdown("""<div class='desc-text'>Raw stellar brightness over time.
The <span style='color:#ff6b6b'>red curve</span> is the Savitzky-Golay stellar trend —
slow variability that completely hides the tiny planet transits.</div>""",
            unsafe_allow_html=True)
fig_raw = plot_raw(raw_t, raw_f, raw_fe, trend_t, trend_f)
st.pyplot(fig_raw, use_container_width=True); plt.close(fig_raw)

# ── Graph 2: Flat ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-3">02 · CLEANED &amp; FLATTENED LIGHT CURVE</div>',
            unsafe_allow_html=True)
st.markdown(f"""<div class='desc-text'>Top: stellar trend removed.
Bottom: outlier spikes clipped. Noise floor ≈
<b style='color:#a8ff78'>{noise_ppm:.0f} ppm</b> — BLS search input.</div>""",
            unsafe_allow_html=True)
fig_flat = plot_flat(flat_t, flat_f, clean_t, clean_f)
st.pyplot(fig_flat, use_container_width=True); plt.close(fig_flat)

# ── Graph 3: BLS ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header animate-in delay-3">03 · BLS PERIODOGRAM &amp; PHASE-FOLDED TRANSIT</div>',
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
  · CLASSIFICATION: {composition['classification'].upper()}
  · NASA MAST ARCHIVE
</div>""", unsafe_allow_html=True)