"""
Figure 3 — Station coverage reliability on the native CHIRPS grid.

Uses the exact CHIRPS 0.05° (~5.5 km) cells. The CHC monthly station-density
GeoTIFFs are already aligned to the CHIRPS grid, so no resampling or
aggregation is applied — each output cell IS a CHIRPS cell.

For each cell, shows the percentage of months in the study period that had
at least one reporting station in that cell. The map directly compares
CHIRPS' nominal 5.5 km resolution to where ground observations actually
exist over time. Vast red expanse = nominal resolution unsupported by data.
Sparse green = pixels actually anchored to ground truth.

`grid_size_km` in Config can be raised (e.g. 50, 100) to aggregate to a
coarser grid for different views.

Inputs: CHC monthly station-density GeoTIFFs.
    v3: https://data.chc.ucsb.edu/products/CHIRPS/v3.0/diagnostics/global_monthly_station_density/tifs/p05/
    v2: https://data.chc.ucsb.edu/products/CHIRPS-2.0/diagnostics/global_monthly_station_density/tifs/p05/

To apply this elsewhere, edit the CONFIG block below — AOI, study period,
grid size, and CHIRPS version are all knobs.

Dependencies: numpy, rasterio, matplotlib, cartopy.
In Colab: !pip install rasterio cartopy
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import cartopy.crs as ccrs
import cartopy.feature as cfeature


@dataclass(frozen=True)
class Config:
    aoi_west: float = -3.0
    aoi_east: float = 12.0
    aoi_south: float = 4.0
    aoi_north: float = 16.0
    study_start: str = "2015-01"
    study_end: str = "2024-12"
    grid_size_km: float = 5.5
    pixel_resolution_deg: float = 0.05
    version: str = "v3"
    show_labels: bool = False
    metric: str = "pct_months"  # "pct_months" or "mean_stations"
    colorbar_bins_pct: tuple[float, ...] = (0, 5, 25, 50, 75, 95, 100)
    colorbar_bins_mean: tuple[float, ...] = (0, 0.5, 1, 2, 3, 5, 10)
    reference_gridline_deg: float = 1.0
    cmap_name: str = "sand_to_burgundy"
    cell_edge_color: str = "#BBBBBB"
    cell_edge_linewidth: float = 0.0
    comparison_periods: tuple[tuple[str, str], ...] | None = (
        ("1981-01", "1991-12"),
        ("1992-01", "2002-12"),
        ("2003-01", "2013-12"),
        ("2014-01", "2025-12"),
    )
    tif_dir: Path = Path("figures/station_density_tifs")
    out_path: Path = Path("figures/Fig3_station_coverage_comparison.png")
    csv_out: Path | None = None
    # Dual-layer mode: 110 km cells colored light-red→light-green;
    # native 5.5 km pixels rendered as black squares sized 5–30 km by coverage.
    dual_layer: bool = True
    bg_grid_km: float = 80.0
    marker_min_km: float = 5.0
    marker_max_km: float = 30.0
    bg_cmap_name: str = "light_rdylgn"


CONFIG = Config()


def months_between(start: str, end: str) -> list[tuple[int, int]]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out: list[tuple[int, int]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out


def read_window(path: Path, cfg: Config) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(path) as ds:
        win = from_bounds(cfg.aoi_west, cfg.aoi_south,
                          cfg.aoi_east, cfg.aoi_north, ds.transform)
        arr = ds.read(1, window=win).astype(np.float32)
        nodata = ds.nodata
        if nodata is not None:
            arr[arr == nodata] = 0
        arr[arr < 0] = 0
        return arr, ds.window_transform(win)


def km_per_pixel(cfg: Config) -> float:
    mean_lat = 0.5 * (cfg.aoi_north + cfg.aoi_south)
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(mean_lat))
    km_per_pixel_lon = cfg.pixel_resolution_deg * km_per_deg_lon
    km_per_pixel_lat = cfg.pixel_resolution_deg * 111.0
    return 0.5 * (km_per_pixel_lat + km_per_pixel_lon)


def tif_path(cfg: Config, year: int, month: int) -> Path:
    return cfg.tif_dir / f"{cfg.version}.stn_density.{year}.{month:02d}.tif"


def aggregate_blocks(arr: np.ndarray, block: int, op: str = "sum") -> np.ndarray:
    h, w = arr.shape
    h_trim = (h // block) * block
    w_trim = (w // block) * block
    arr = arr[:h_trim, :w_trim]
    reshaped = arr.reshape(h_trim // block, block, w_trim // block, block)
    if op == "sum":
        return reshaped.sum(axis=(1, 3))
    if op == "max":
        return reshaped.max(axis=(1, 3))
    raise ValueError(f"Unknown op: {op}")


def grid_setup(cfg: Config) -> tuple[list[tuple[int, int]], int, float]:
    months = months_between(cfg.study_start, cfg.study_end)
    if not months:
        raise ValueError(f"Empty study period: {cfg.study_start} to {cfg.study_end}")
    kpp = km_per_pixel(cfg)
    block_px = max(1, int(round(cfg.grid_size_km / kpp)))
    print(f"AOI mean km/pixel: {kpp:.2f}")
    print(f"Grid size {cfg.grid_size_km:.0f} km -> block {block_px} px "
          f"(~{block_px * kpp:.0f} km per cell side)")
    print(f"Study period: {len(months)} months, {months[0]} to {months[-1]}")
    return months, block_px, kpp


def coarse_transform_for(transform: rasterio.Affine, block_px: int) -> rasterio.Affine:
    return rasterio.Affine(
        transform.a * block_px, transform.b, transform.c,
        transform.d, transform.e * block_px, transform.f,
    )


def mean_stations_per_grid(cfg: Config) -> tuple[np.ndarray, rasterio.Affine, int]:
    months, block_px, _ = grid_setup(cfg)
    monthly_sum = None
    transform = None
    for year, month in months:
        path = tif_path(cfg, year, month)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name} at {path.absolute()}. "
                f"See file header for download URLs."
            )
        arr, tr = read_window(path, cfg)
        if monthly_sum is None:
            monthly_sum = arr.copy()
            transform = tr
        else:
            monthly_sum += arr

    coarse_total = aggregate_blocks(monthly_sum, block_px, op="sum")
    mean_grid = coarse_total / len(months)
    return mean_grid, coarse_transform_for(transform, block_px), len(months)


def pct_months_with_station_per_grid(
    cfg: Config,
) -> tuple[np.ndarray, rasterio.Affine, int]:
    months, block_px, _ = grid_setup(cfg)
    coverage_count = None
    transform = None
    for year, month in months:
        path = tif_path(cfg, year, month)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name} at {path.absolute()}. "
                f"See file header for download URLs."
            )
        arr, tr = read_window(path, cfg)
        binary_pixel = (arr > 0).astype(np.float32)
        cell_binary = aggregate_blocks(binary_pixel, block_px, op="max")
        if coverage_count is None:
            coverage_count = cell_binary
            transform = tr
        else:
            coverage_count += cell_binary

    pct = (coverage_count / len(months)) * 100.0
    return pct, coarse_transform_for(transform, block_px), len(months)


def format_label(v: float, metric: str) -> str:
    if metric == "pct_months":
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def label_threshold(metric: str) -> float:
    return 1.0 if metric == "pct_months" else 0.1


def dark_threshold(metric: str) -> float:
    return 60.0 if metric == "pct_months" else 3.0


def cell_label_text(cfg: Config) -> str:
    if cfg.grid_size_km <= 6.0:
        return "0.05° CHIRPS cell"
    return f"~{cfg.grid_size_km:.0f} km cell"


def colorbar_label(cfg: Config) -> str:
    cell = cell_label_text(cfg)
    if cfg.metric == "pct_months":
        return (f"Months with ≥1 station per {cell} (%)\n"
                f"{cfg.study_start} to {cfg.study_end}")
    return (f"Mean stations per {cell} per month\n"
            f"{cfg.study_start} to {cfg.study_end}")


def plot_title(cfg: Config, n_months: int) -> str:
    cell = cell_label_text(cfg)
    if cfg.metric == "pct_months":
        return (f"Station-coverage reliability — CHIRPS {cfg.version} "
                f"native grid, West Africa\n"
                f"% of months with ≥1 station per {cell}, "
                f"{cfg.study_start} to {cfg.study_end} ({n_months} months).")
    return (f"Station density per {cell} — CHIRPS {cfg.version}, "
            f"West Africa\n"
            f"Mean monthly station count, {cfg.study_start} to "
            f"{cfg.study_end} ({n_months} months).")


CUSTOM_CMAPS: dict[str, list[tuple[float, str]]] = {
    # Original tries (kept for reference)
    "light_rdylgn": [(0.0, "#F4A6A6"), (0.5, "#FFF5B7"), (1.0, "#A6E3A6")],
    "muted_rdylgn": [(0.0, "#B23A48"), (0.5, "#F2D096"), (1.0, "#2D6A4F")],
    "soft_red_to_green": [(0.0, "#FBE4E4"), (0.5, "#F2D096"), (1.0, "#2D6A4F")],
    # Single-hue and cool-tone alternatives
    "cream_to_navy":   [(0.0, "#FBF5E5"), (0.5, "#79A8A9"), (1.0, "#1A3A52")],
    "ivory_to_forest": [(0.0, "#FAF7E8"), (0.5, "#A8C0A2"), (1.0, "#2E5339")],
    "blush_to_plum":   [(0.0, "#FBEDEA"), (0.5, "#C29CAA"), (1.0, "#4A2540")],
    "sand_to_burgundy":[(0.0, "#F8EFD8"), (0.5, "#C97B63"), (1.0, "#5A1F2B")],
    # New paper-grade single-hue
    "linen_to_indigo": [(0.0, "#F8F2E4"), (1.0, "#1E3A5F")],
    "snow_to_charcoal":[(0.0, "#F5F5F2"), (1.0, "#2C2C2C")],
    "wheat_to_olive":  [(0.0, "#F4ECD2"), (1.0, "#4A5A2C")],
}


def get_cmap(cfg: Config) -> mcolors.Colormap:
    if cfg.cmap_name in CUSTOM_CMAPS:
        return mcolors.LinearSegmentedColormap.from_list(
            cfg.cmap_name, CUSTOM_CMAPS[cfg.cmap_name], N=256,
        )
    return plt.get_cmap(cfg.cmap_name)


def get_norm_and_ticks(cfg: Config) -> tuple[mcolors.Normalize, list, str]:
    if cfg.metric == "pct_months":
        return mcolors.Normalize(vmin=0, vmax=100), [0, 25, 50, 75, 100], "neither"
    bounds = list(cfg.colorbar_bins_mean)
    cmap_for_norm = plt.get_cmap("YlOrRd", len(bounds) - 1)
    return mcolors.BoundaryNorm(bounds, cmap_for_norm.N), bounds, "max"


def prepare_plot_data(grid: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.metric == "pct_months":
        return grid
    return np.where(grid > 0, grid, np.nan)


def extent_from(transform: rasterio.Affine, shape: tuple[int, int]) -> tuple:
    h, w = shape
    left = transform.c
    top = transform.f
    right = left + transform.a * w
    bottom = top + transform.e * h
    return (left, right, bottom, top)


def add_basemap(ax, cfg: Config) -> None:
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#333333")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#333333")
    ax.set_extent([cfg.aoi_west, cfg.aoi_east,
                   cfg.aoi_south, cfg.aoi_north],
                  crs=ccrs.PlateCarree())
    step = cfg.reference_gridline_deg
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="#777777", alpha=0.4,
        xlocs=np.arange(np.floor(cfg.aoi_west),
                        np.ceil(cfg.aoi_east) + step, step),
        ylocs=np.arange(np.floor(cfg.aoi_south),
                        np.ceil(cfg.aoi_north) + step, step),
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": "#444444"}
    gl.ylabel_style = {"size": 8, "color": "#444444"}


def add_cell_labels(ax, grid: np.ndarray, transform: rasterio.Affine,
                    cfg: Config) -> None:
    if not cfg.show_labels:
        return
    h, w = grid.shape
    lo = label_threshold(cfg.metric)
    hi = dark_threshold(cfg.metric)
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v < lo:
                continue
            lon = transform.c + (c + 0.5) * transform.a
            lat = transform.f + (r + 0.5) * transform.e
            ax.text(lon, lat, format_label(v, cfg.metric),
                    ha="center", va="center", fontsize=6,
                    color="white" if v >= hi else "black",
                    transform=ccrs.PlateCarree())


def plot_panel(ax, grid: np.ndarray, transform: rasterio.Affine,
               cfg: Config, panel_title: str,
               cmap, norm) -> "plt.cm.ScalarMappable":
    plot_data = prepare_plot_data(grid, cfg)
    if cfg.cell_edge_linewidth > 0:
        h, w = grid.shape
        lon_edges = np.linspace(transform.c,
                                transform.c + transform.a * w, w + 1)
        lat_edges = np.linspace(transform.f,
                                transform.f + transform.e * h, h + 1)
        im = ax.pcolormesh(
            lon_edges, lat_edges, plot_data,
            cmap=cmap, norm=norm,
            edgecolors=cfg.cell_edge_color,
            linewidth=cfg.cell_edge_linewidth,
            transform=ccrs.PlateCarree(),
            shading="flat",
        )
    else:
        extent = extent_from(transform, grid.shape)
        im = ax.imshow(
            plot_data, extent=extent, origin="upper",
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            interpolation="none",
        )
    add_basemap(ax, cfg)
    add_cell_labels(ax, grid, transform, cfg)
    ax.set_title(panel_title, fontsize=11, pad=4)
    return im


def plot_single(grid: np.ndarray, transform: rasterio.Affine,
                cfg: Config, n_months: int) -> None:
    cmap = get_cmap(cfg)
    norm, ticks, extend = get_norm_and_ticks(cfg)

    fig, ax = plt.subplots(
        figsize=(11, 11),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    panel_title = f"{cfg.study_start} to {cfg.study_end} ({n_months} months)"
    im = plot_panel(ax, grid, transform, cfg, panel_title, cmap, norm)

    cb = fig.colorbar(im, ax=ax, orientation="vertical",
                      shrink=0.7, pad=0.04,
                      ticks=ticks, extend=extend)
    cb.set_label(colorbar_label(cfg), fontsize=10)
    fig.suptitle(plot_title(cfg, n_months), fontsize=12, y=0.98)
    cfg.out_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(cfg.out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {cfg.out_path}")


def plot_comparison(panels: list[tuple[np.ndarray, rasterio.Affine, int, str]],
                    cfg: Config) -> None:
    cmap = get_cmap(cfg)
    norm, ticks, extend = get_norm_and_ticks(cfg)

    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(7.5 * n, 6.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.10,
                                wspace=0.0, hspace=0.0)
    if n == 1:
        axes = [axes]
    last_im = None
    for ax, (grid, transform, n_months, label) in zip(axes, panels):
        last_im = plot_panel(
            ax, grid, transform, cfg,
            f"{label}  ({n_months} months)",
            cmap, norm,
        )

    cb = fig.colorbar(
        last_im, ax=axes, orientation="horizontal",
        shrink=0.55, pad=0.05, aspect=45,
        ticks=ticks, extend=extend,
    )
    cb.set_label(comparison_colorbar_label(cfg), fontsize=10)

    fig.suptitle(comparison_suptitle(cfg), fontsize=12)
    cfg.out_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(cfg.out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {cfg.out_path}")


def comparison_colorbar_label(cfg: Config) -> str:
    cell = cell_label_text(cfg)
    if cfg.metric == "pct_months":
        return f"% of months with ≥1 station per {cell}"
    return f"Mean stations per {cell} per month"


def comparison_suptitle(cfg: Config) -> str:
    cell = cell_label_text(cfg)
    if cfg.metric == "pct_months":
        return (f"Station-coverage reliability — CHIRPS {cfg.version} "
                f"native grid, West Africa\n"
                f"% of months with ≥1 station per {cell}, by period")
    return (f"Station density per {cell} — CHIRPS {cfg.version}, "
            f"West Africa, by period")


def save_csv(grid: np.ndarray, transform: rasterio.Affine, cfg: Config) -> None:
    if cfg.csv_out is None:
        return
    h, w = grid.shape
    column = "pct_months_with_station" if cfg.metric == "pct_months" else "mean_stations_per_month"
    cfg.csv_out.parent.mkdir(exist_ok=True, parents=True)
    with cfg.csv_out.open("w") as f:
        f.write(f"cell_lon_center,cell_lat_center,{column}\n")
        for r in range(h):
            for c in range(w):
                lon = transform.c + (c + 0.5) * transform.a
                lat = transform.f + (r + 0.5) * transform.e
                f.write(f"{lon:.4f},{lat:.4f},{grid[r,c]:.4f}\n")
    print(f"Saved {cfg.csv_out}")


def compute_grid(cfg: Config) -> tuple[np.ndarray, rasterio.Affine, int]:
    if cfg.metric == "pct_months":
        return pct_months_with_station_per_grid(cfg)
    if cfg.metric == "mean_stations":
        return mean_stations_per_grid(cfg)
    raise ValueError(
        f"Unknown metric: {cfg.metric}. Use 'pct_months' or 'mean_stations'."
    )


def compute_native_pct_grid(cfg: Config) -> tuple[np.ndarray, rasterio.Affine, int]:
    return pct_months_with_station_per_grid(cfg)


def compute_coarse_pct_grid(cfg: Config) -> tuple[np.ndarray, rasterio.Affine, int]:
    coarse_cfg = replace(cfg, grid_size_km=cfg.bg_grid_km)
    return pct_months_with_station_per_grid(coarse_cfg)


def get_bg_cmap(cfg: Config) -> mcolors.Colormap:
    return mcolors.LinearSegmentedColormap.from_list(
        cfg.bg_cmap_name, CUSTOM_CMAPS[cfg.bg_cmap_name], N=256,
    )


def cell_step_deg(cfg: Config) -> float:
    mean_lat = 0.5 * (cfg.aoi_north + cfg.aoi_south)
    kpp = 0.5 * (
        cfg.pixel_resolution_deg * 111.0
        + cfg.pixel_resolution_deg * 111.0 * np.cos(np.deg2rad(mean_lat))
    )
    block_px = max(1, int(round(cfg.bg_grid_km / kpp)))
    return block_px * cfg.pixel_resolution_deg


def add_basemap_dual(ax, cfg: Config,
                     row: int = 0, col: int = 0,
                     n_rows: int = 1, n_cols: int = 1) -> None:
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe0ee",
                   edgecolor="none", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#333333", zorder=4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#333333", zorder=4)
    ax.set_extent([cfg.aoi_west, cfg.aoi_east,
                   cfg.aoi_south, cfg.aoi_north],
                  crs=ccrs.PlateCarree())

    cell_step = cell_step_deg(cfg)
    ax.gridlines(
        draw_labels=False,
        linewidth=0.4, color="#333333", alpha=0.55,
        xlocs=np.arange(cfg.aoi_west, cfg.aoi_east + 1e-9, cell_step),
        ylocs=np.arange(cfg.aoi_south, cfg.aoi_north + 1e-9, cell_step),
    )

    label_step = cfg.reference_gridline_deg
    gl = ax.gridlines(
        draw_labels=True, linewidth=0, alpha=0,
        xlocs=np.arange(np.floor(cfg.aoi_west),
                        np.ceil(cfg.aoi_east) + label_step, label_step),
        ylocs=np.arange(np.floor(cfg.aoi_south),
                        np.ceil(cfg.aoi_north) + label_step, label_step),
    )
    gl.top_labels = gl.right_labels = False
    if col > 0 and n_cols > 1:
        gl.left_labels = False
    if row < n_rows - 1 and n_rows > 1:
        gl.bottom_labels = False
    gl.xlabel_style = {"size": 8, "color": "#333333"}
    gl.ylabel_style = {"size": 8, "color": "#333333"}


def plot_panel_dual_layer(
    ax,
    native_grid: np.ndarray, native_transform: rasterio.Affine,
    coarse_grid: np.ndarray, coarse_transform: rasterio.Affine,
    cfg: Config, panel_title: str,
    bg_cmap, bg_norm,
    row: int = 0, col: int = 0,
    n_rows: int = 1, n_cols: int = 1,
) -> "plt.cm.ScalarMappable":
    extent = extent_from(coarse_transform, coarse_grid.shape)
    im = ax.imshow(
        coarse_grid, extent=extent, origin="upper",
        cmap=bg_cmap, norm=bg_norm,
        transform=ccrs.PlateCarree(),
        interpolation="none",
        alpha=0.6,
        zorder=1,
    )

    mean_lat = 0.5 * (cfg.aoi_north + cfg.aoi_south)
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(mean_lat))
    km_per_deg_lat = 111.0

    mask = native_grid > 0
    rs, cs = np.where(mask)
    covs = native_grid[mask]

    sizes_km = (
        cfg.marker_min_km
        + (cfg.marker_max_km - cfg.marker_min_km) * (covs / 100.0)
    )
    width_deg = sizes_km / km_per_deg_lon
    height_deg = sizes_km / km_per_deg_lat

    lon_centers = native_transform.c + (cs + 0.5) * native_transform.a
    lat_centers = native_transform.f + (rs + 0.5) * native_transform.e

    for lon, lat, w_d, h_d in zip(lon_centers, lat_centers, width_deg, height_deg):
        ell = Ellipse(
            (lon, lat), width=w_d, height=h_d,
            facecolor="#1a1a2e", edgecolor="none",
            alpha=0.45,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.add_patch(ell)

    add_basemap_dual(ax, cfg, row=row, col=col, n_rows=n_rows, n_cols=n_cols)
    ax.set_title(panel_title, fontsize=11, pad=4)
    return im


def comparison_suptitle_dual(cfg: Config) -> str:
    return f"Station coverage reliability, CHIRPS {cfg.version}, West Africa"


def plot_comparison_dual(
    panels_data: list[tuple[np.ndarray, rasterio.Affine,
                            np.ndarray, rasterio.Affine, int, str]],
    cfg: Config,
) -> None:
    bg_cmap = get_bg_cmap(cfg)
    bg_norm = mcolors.Normalize(vmin=0, vmax=100)

    n = len(panels_data)
    if n == 4:
        n_rows, n_cols = 2, 2
        figsize = (13, 12)
        top, bottom, hspace, wspace = 0.93, 0.08, 0.04, 0.06
    elif n == 2:
        n_rows, n_cols = 1, 2
        figsize = (13, 6.8)
        top, bottom, hspace, wspace = 0.86, 0.20, 0.0, 0.02
    else:
        n_rows, n_cols = 1, n
        figsize = (6.5 * n, 6.8)
        top, bottom, hspace, wspace = 0.86, 0.20, 0.0, 0.02

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        n_rows, n_cols,
        wspace=wspace, hspace=hspace,
        left=0.05, right=0.97,
        top=top, bottom=bottom,
    )
    axes = []
    for i in range(n):
        r, c = divmod(i, n_cols)
        axes.append(fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree()))

    last_im = None
    for i, (ax, panel) in enumerate(zip(axes, panels_data)):
        native_grid, native_t, coarse_grid, coarse_t, n_months, label = panel
        r, c = divmod(i, n_cols)
        last_im = plot_panel_dual_layer(
            ax, native_grid, native_t, coarse_grid, coarse_t,
            cfg, f"{label}  ({n_months} months)",
            bg_cmap, bg_norm,
            row=r, col=c, n_rows=n_rows, n_cols=n_cols,
        )

    # Realize layout so we can read the actual rendered panel size
    # and match legend circles to map circles in absolute points.
    fig.canvas.draw()
    panel_bbox_in = axes[0].get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    lon_range = cfg.aoi_east - cfg.aoi_west
    inches_per_deg = panel_bbox_in.width / lon_range
    pts_per_deg = inches_per_deg * 72.0
    mean_lat = 0.5 * (cfg.aoi_north + cfg.aoi_south)
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(mean_lat))
    km_per_deg_lat = 111.0
    pts_per_km = pts_per_deg / np.sqrt(km_per_deg_lon * km_per_deg_lat)

    cb_y = 0.04 if n_rows > 1 else 0.10
    cb_ax = fig.add_axes([0.06, cb_y, 0.40, 0.018])
    cb = fig.colorbar(
        last_im, cax=cb_ax, orientation="horizontal",
        ticks=[0, 25, 50, 75, 100],
    )
    cb.set_label(
        f"% months with ≥1 station per {cfg.bg_grid_km:.0f} km cell",
        fontsize=10,
    )

    size_legend_y = cb_y - 0.018
    size_ax = fig.add_axes([0.55, size_legend_y, 0.42, 0.025])
    size_ax.set_xlim(0, 1)
    size_ax.set_ylim(0, 1)
    size_ax.axis("off")

    heading = "Native pixel size = % months with ≥1 station :"
    size_ax.text(0.0, 0.5, heading, ha="left", va="center", fontsize=10)

    sample_pcts = [1.0, 100.0]
    sample_labels = ["1%", "100%"]
    sample_sizes_km = [
        cfg.marker_min_km + (cfg.marker_max_km - cfg.marker_min_km) * p / 100.0
        for p in sample_pcts
    ]
    diameters_pt = [sz * pts_per_km for sz in sample_sizes_km]
    s_values = [d ** 2 for d in diameters_pt]
    small_x, large_x = 0.67, 0.78
    size_ax.scatter([small_x, large_x], [0.5, 0.5],
                    s=s_values, c="#1a1a2e", alpha=0.45,
                    marker="o", edgecolors="none")
    size_ax.text(small_x + 0.020, 0.5, sample_labels[0],
                 ha="left", va="center", fontsize=10)
    size_ax.text(large_x + 0.024, 0.5, sample_labels[1],
                 ha="left", va="center", fontsize=10)

    title_y = 0.97 if n_rows > 1 else 0.95
    fig.suptitle(comparison_suptitle_dual(cfg), fontsize=18, y=title_y)
    cfg.out_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(cfg.out_path, dpi=200, bbox_inches="tight", pad_inches=0.10)
    print(f"Saved {cfg.out_path}")


def summarise(label: str, grid: np.ndarray, cfg: Config) -> None:
    nz = (grid > 0).sum()
    unit = "%" if cfg.metric == "pct_months" else ""
    print(
        f"  {label}: cells={grid.size}, with-stations={nz} "
        f"({100 * nz / grid.size:.1f}%), "
        f"max={grid.max():.2f}{unit}, "
        f"mean={grid.mean():.2f}{unit}"
    )


def main(cfg: Config = CONFIG) -> None:
    if cfg.comparison_periods is not None and cfg.dual_layer:
        print(f"CHIRPS {cfg.version}, dual-layer mode "
              f"(native {cfg.marker_min_km:.0f}–{cfg.marker_max_km:.0f} km squares "
              f"on {cfg.bg_grid_km:.0f} km coloured cells), "
              f"comparison of {len(cfg.comparison_periods)} periods")
        panels_data: list[tuple[np.ndarray, rasterio.Affine,
                                np.ndarray, rasterio.Affine, int, str]] = []
        for start, end in cfg.comparison_periods:
            sub_cfg = replace(cfg, study_start=start, study_end=end)
            native_grid, native_t, n_months = compute_native_pct_grid(sub_cfg)
            coarse_grid, coarse_t, _ = compute_coarse_pct_grid(sub_cfg)
            label = f"{start} to {end}"
            summarise(f"{label} native", native_grid, sub_cfg)
            summarise(f"{label} {cfg.bg_grid_km:.0f}km", coarse_grid, sub_cfg)
            panels_data.append(
                (native_grid, native_t, coarse_grid, coarse_t, n_months, label)
            )
        plot_comparison_dual(panels_data, cfg)
        return

    if cfg.comparison_periods is not None:
        print(f"CHIRPS {cfg.version}, metric={cfg.metric}, "
              f"comparison of {len(cfg.comparison_periods)} periods")
        panels: list[tuple[np.ndarray, rasterio.Affine, int, str]] = []
        for start, end in cfg.comparison_periods:
            sub_cfg = replace(cfg, study_start=start, study_end=end)
            grid, transform, n_months = compute_grid(sub_cfg)
            label = f"{start} to {end}"
            summarise(label, grid, sub_cfg)
            panels.append((grid, transform, n_months, label))
        plot_comparison(panels, cfg)
        return

    print(f"CHIRPS {cfg.version}, metric={cfg.metric}, "
          f"{cfg.study_start} to {cfg.study_end}")
    grid, transform, n_months = compute_grid(cfg)
    summarise(f"{cfg.study_start} to {cfg.study_end}", grid, cfg)
    plot_single(grid, transform, cfg, n_months)
    save_csv(grid, transform, cfg)


if __name__ == "__main__":
    main()
