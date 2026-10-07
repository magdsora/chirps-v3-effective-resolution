"""
Render submission-ready versions of Figure 1 and Figure 2:
  - one colorblind-safe sequential blue scale shared by both figures
    (light = little or no gauge support, dark = strong gauge support),
    orange/blue time-series lines, and a neutral grey ocean
  - 300 dpi
  - no embedded suptitles (the manuscript captions carry that text)

Outputs to figures/submission/. Does not modify the original scripts or their
published outputs; overrides are applied here via config replace + monkeypatch.
Run from the project root:  python3 make_submission_figs.py
"""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes as geoaxes

OUT_DIR = Path("figures/submission")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- global overrides: force 300 dpi, suppress figure-level suptitles -------
_orig_savefig = matplotlib.figure.Figure.savefig

def _savefig_300(self, *args, **kwargs):
    kwargs["dpi"] = 300
    return _orig_savefig(self, *args, **kwargs)

matplotlib.figure.Figure.savefig = _savefig_300
matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

# ---- shared colours ---------------------------------------------------------
# One sequential blue ramp (light -> dark) for both figures' maps. Under a
# Machado et al. (2009) protan/deutan simulation, 0% vs 100% coverage and
# R² 0.25 vs 0.85 stay well separated, unlike the red-yellow-green originals.
BLUE = ["#e4effc", "#b7d3f6", "#86b6ef", "#5598e7",
        "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
OCEAN = "#e6e6e3"  # neutral grey, so the sea is not read as a light blue value

_orig_add_feature = geoaxes.GeoAxes.add_feature

def _add_feature(self, feature, **kwargs):
    if feature is cfeature.OCEAN:
        kwargs["facecolor"] = OCEAN
    return _orig_add_feature(self, feature, **kwargs)

geoaxes.GeoAxes.add_feature = _add_feature

# ---------------------------------------------------------------- Figure 1 --
import fig3_effective_resolution as f3

f3.CUSTOM_CMAPS["blue_seq"] = [(i / (len(BLUE) - 1), c) for i, c in enumerate(BLUE)]

fig1_cfg = replace(
    f3.CONFIG,
    bg_cmap_name="blue_seq",
    out_path=OUT_DIR / "Fig1_station_coverage.png",
)
print("== Figure 1 ==")
f3.main(fig1_cfg)

# ---------------------------------------------------------------- Figure 2 --
import r2_diagnostic as r2

# time-series lines: orange vs blue stay distinct for red-green colour-blind readers
r2.STYLE["West Africa"]["color"] = "#eb6834"
r2.STYLE["NW US"]["color"] = "#2a78d6"

BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list("blue_seq", BLUE, N=256)


def plot_spatial(ax, grid, transform, region, panel_label, subtitle, title_pad=6):
    """r2.plot_spatial with the blue ramp on a 0.20-0.85 scale (the published
    R² range), so 0.25 (satellite-only) sits at the light end."""
    h, w = grid.shape
    left, top = transform.c, transform.f
    extent = (left, left + transform.a * w, top + transform.e * h, top)
    im = ax.imshow(grid, extent=extent, origin="upper", cmap=BLUE_CMAP,
                   norm=mcolors.Normalize(vmin=0.2, vmax=0.85),
                   transform=ccrs.PlateCarree(), interpolation="none", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN, edgecolor="none", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#333333", zorder=4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#333333", zorder=4)
    ax.set_extent([region.west, region.east, region.south, region.north],
                  crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#777777", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax.set_title(f"{panel_label}. {subtitle}", fontsize=11, pad=title_pad)
    return im


r2.plot_spatial = plot_spatial

fig2_cfg = replace(
    r2.CONFIG,
    out_combined=OUT_DIR / "Fig2_r2_combined.png",
    timeseries_csv=OUT_DIR / "r2_timeseries.csv",
    out_timeseries=OUT_DIR / "Fig2a_unused.png",
    out_spatial=OUT_DIR / "Fig2b_unused.png",
)
print("== Figure 2 ==")
r2.main(fig2_cfg)

print("done ->", OUT_DIR)
