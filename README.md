# Effective resolution of CHIRPS v3.0 in West Africa

Code accompanying the manuscript *"What Does '5.5 km Daily' Actually Mean?
Resolution Claims of CHIRPS v3.0 in West Africa."*

CHIRPS v3.0 is distributed on a 0.05° (≈5.5 km) grid with daily files, and
downstream work routinely cites both numbers as the dataset's resolution.
Neither number describes how much of a given pixel is actually anchored to a
rain gauge. Two diagnostics that *do* describe it are published by the Climate
Hazards Center alongside the data, and the scripts here read them.

## What the scripts produce

| Script | Output | What it shows |
| --- | --- | --- |
| `fig3_effective_resolution.py` | Figure 1 | Share of months in which each CHIRPS pixel contained at least one reporting gauge, on the native 0.05° grid and aggregated to 80 km cells, over four ~11-year periods |
| `r2_diagnostic.py` | Figure 2 | Monthly per-pixel station-blending R² (Funk et al. 2026, Eq. 10): July and January time series for West Africa against a densely gauged reference (NW US), plus mean July maps for 1981–1985 and 2021–2025 |
| `make_submission_figs.py` | Both, 300 dpi | Re-renders Figures 1 and 2 with colourblind-safe palettes and no embedded titles, into `figures/submission/` |

Rendered versions are committed under [`figures/submission/`](figures/submission).

## Domains

Both are configurable at the top of each script via the `Config` dataclass.

- **Analysis domain** (`r2_diagnostic.py`, and the station counts quoted in the
  text): 6°W–17°E, 3–25°N.
- **Figure 1 map window** (`fig3_effective_resolution.py`): 3°W–12°E, 4–16°N,
  the most densely gauged part of the domain.
- **Reference region**: NW US, 125–110°W, 40–50°N.

## Data

No raster data is committed. Both diagnostics are public:

- Station density (monthly GeoTIFFs, gauge count per 0.05° pixel):
  <https://data.chc.ucsb.edu/products/CHIRPS/v3.0/diagnostics/station_density/>
- Station-blending R² (monthly GeoTIFFs):
  <https://data.chc.ucsb.edu/products/CHIRPS/v3.0/diagnostics/monthly.Rsquared.estimate/tifs/>

`r2_diagnostic.py` downloads the R² tiles it needs into `r2_cache/` on first
run. The station-density tiles are expected to be present locally in
`figures/station_density_tifs/`; fetch them with:

```sh
mkdir -p figures/station_density_tifs
wget -r -np -nd -A 'v3.stn_density.*.tif' \
  -P figures/station_density_tifs \
  https://data.chc.ucsb.edu/products/CHIRPS/v3.0/diagnostics/station_density/
```

The full archive is roughly 540 monthly files from January 1981 onward. Recent
months are still being populated and should not be treated as complete.

## Running

```sh
pip install -r requirements.txt
python3 fig3_effective_resolution.py    # Figure 1
python3 r2_diagnostic.py                # Figure 2
python3 make_submission_figs.py         # both, submission-ready
```

Run from the repository root; the scripts resolve paths relative to it.
`cartopy` is the one awkward dependency — on macOS it is usually easiest via
`conda install -c conda-forge cartopy`.

## Reading the R² diagnostic

R² is the variance explained by the CHIRPS blending scheme, combining a
station-adjusted estimate with a satellite-only estimate assumed to correlate
0.5 with true precipitation. It is an analytic function of distance to the
nearest reporting gauge and a local decorrelation slope, not a
cross-validation statistic, so it measures how much of a pixel the scheme
*believes* is anchored to stations, not accuracy.

- **0.25** — no gauge within reach; the pixel carries the satellite-only
  estimate. This is the most common value across the West African domain.
- **below 0.25** — a gauge at intermediate distance, not an absence of gauges.
  The expression is not monotonic and dips to 0.207.
- **≈0.85** — a gauge sits in or beside the pixel.

## Citing

The CHIRPS v3.0 dataset and both diagnostics are the work of the Climate
Hazards Center at UC Santa Barbara:

> Funk, C., Peterson, P., Harrison, L., et al. (2026) 'The Climate Hazards
> Center Infrared Precipitation with Stations, Version 3', *Scientific Data*
> 13, 718. <https://doi.org/10.1038/s41597-026-07096-4>

## License

MIT — see [LICENSE](LICENSE).
