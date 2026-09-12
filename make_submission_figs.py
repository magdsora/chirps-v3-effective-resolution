"""
Render submission-ready versions of Figure 1 and Figure 2:
  - colorblind-safe palettes (orange = un-anchored, purple = station-anchored;
    no red-green), consistent across both figures
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
import matplotlib.figure
import matplotlib.pyplot as plt

OUT_DIR = Path("figures/submission")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- global overrides: force 300 dpi, suppress figure-level suptitles -------
_orig_savefig = matplotlib.figure.Figure.savefig

def _savefig_300(self, *args, **kwargs):
    kwargs["dpi"] = 300
    return _orig_savefig(self, *args, **kwargs)

matplotlib.figure.Figure.savefig = _savefig_300
matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

# ---------------------------------------------------------------- Figure 1 --
import fig3_effective_resolution as f3

# pastel orange -> off-white -> pastel purple (CVD-safe; distinct from ocean blue)
f3.CUSTOM_CMAPS["light_orpu"] = [
    (0.0, "#F2B279"), (0.5, "#F7F3EA"), (1.0, "#BFA0DC"),
]

fig1_cfg = replace(
    f3.CONFIG,
    bg_cmap_name="light_orpu",
    out_path=OUT_DIR / "Fig1_station_coverage.png",
)
print("== Figure 1 ==")
f3.main(fig1_cfg)

# ---------------------------------------------------------------- Figure 2 --
import r2_diagnostic as r2

# reference line: green -> purple (ties to purple = well-anchored in the maps)
r2.STYLE["NW US"]["color"] = "#5B4B8A"

# spatial maps: RdYlGn -> PuOr (orange = satellite-only floor,
# purple = station-anchored; matches Figure 1's semantics).
# Patch get_cmap only for this run.
_orig_get_cmap = plt.get_cmap
plt.get_cmap = lambda name=None, *a, **k: _orig_get_cmap(
    "PuOr" if name == "RdYlGn" else name, *a, **k
)

fig2_cfg = replace(
    r2.CONFIG,
    out_combined=OUT_DIR / "Fig2_r2_combined.png",
    timeseries_csv=OUT_DIR / "r2_timeseries.csv",
    out_timeseries=OUT_DIR / "Fig2a_unused.png",
    out_spatial=OUT_DIR / "Fig2b_unused.png",
)
print("== Figure 2 ==")
r2.main(fig2_cfg)

plt.get_cmap = _orig_get_cmap
print("done ->", OUT_DIR)
