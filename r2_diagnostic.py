"""
R² diagnostic — single merged Figure 2 (2x2 + colorbar), West Africa.

Panels:
    A. July R² timeseries, 1981-2025 (West African wet season).
    B. January R² timeseries, 1981-2025 (dry season).
       A and B compare West Africa against NW US, a well-gauged reference.
    C. Mean July R² over West Africa, 1981-1985 (early epoch).
    D. Mean July R² over West Africa, 2021-2025 (late epoch).

R² is the variance explained by the CHIRPS3 station-blending procedure
(Funk et al. 2026, Eq. 10). It ranges from ~0.25 in satellite-only cells
(no station influence) to ~0.85 in cells anchored by a nearby station.

Source: https://data.chc.ucsb.edu/products/CHIRPS/v3.0/diagnostics/monthly.Rsquared.estimate/tifs/
File pattern: monthly-R2.YYYY.MM.tif

Dependencies: numpy, rasterio, pandas, matplotlib, cartopy.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import rasterio
from rasterio.windows import from_bounds
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


URL_TEMPLATE = (
    "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/diagnostics/"
    "monthly.Rsquared.estimate/tifs/monthly-R2.{year}.{month:02d}.tif"
)


@dataclass(frozen=True)
class Region:
    name: str
    west: float
    south: float
    east: float
    north: float


REGIONS: tuple[Region, ...] = (
    Region("West Africa",  -6.0,   3.0,  17.0, 25.0),
    Region("NW US",      -125.0,  40.0, -110.0, 50.0),
)
WA = REGIONS[0]


@dataclass(frozen=True)
class Config:
    years: tuple[int, ...] = tuple(range(1981, 2026))
    months: tuple[int, ...] = (1, 7)
    spatial_month: int = 7
    early_window: tuple[int, int] = (1981, 1985)
    late_window: tuple[int, int] = (2021, 2025)
    floor_exact: float = 0.25
    floor_threshold: float = 0.30
    cache_dir: Path = Path("r2_cache")
    timeseries_csv: Path = Path("figures/r2_timeseries.csv")
    out_timeseries: Path = Path("figures/Fig4a_r2_timeseries.png")
    out_spatial: Path = Path("figures/Fig4b_r2_spatial.png")
    out_combined: Path = Path("figures/Fig2_r2_combined.png")


CONFIG = Config()


def download_if_missing(year: int, month: int, cache: Path) -> Path:
    fname = f"monthly-R2.{year}.{month:02d}.tif"
    local = cache / fname
    if local.exists():
        return local
    cache.mkdir(parents=True, exist_ok=True)
    url = URL_TEMPLATE.format(year=year, month=month)
    print(f"  downloading {fname}")
    urlretrieve(url, local)
    return local


def read_region_values(path: Path, region: Region) -> np.ndarray:
    with rasterio.open(path) as ds:
        win = from_bounds(region.west, region.south,
                          region.east, region.north, ds.transform)
        arr = ds.read(1, window=win).astype(np.float32)
        nodata = ds.nodata
        if nodata is not None:
            arr = arr[arr != nodata]
        # The R² files carry no nodata tag and store ocean as 0, a value Eq. 10
        # cannot produce (its minimum is ~0.207), so 0 is treated as no data.
        arr = arr[(arr > 0) & (arr <= 1)]
    return arr


def read_region_array(path: Path, region: Region) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(path) as ds:
        win = from_bounds(region.west, region.south,
                          region.east, region.north, ds.transform)
        arr = ds.read(1, window=win).astype(np.float32)
        nodata = ds.nodata
        transform = ds.window_transform(win)
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[(arr <= 0) | (arr > 1)] = np.nan  # 0 = ocean, see read_region_values
    return arr, transform


def collect_time_series(cfg: Config) -> pd.DataFrame:
    rows: list[dict] = []
    for year in cfg.years:
        for month in cfg.months:
            path = download_if_missing(year, month, cfg.cache_dir)
            for region in REGIONS:
                vals = read_region_values(path, region)
                if vals.size == 0:
                    continue
                rows.append(dict(
                    year=year, month=month, region=region.name,
                    median=float(np.median(vals)),
                    p25=float(np.percentile(vals, 25)),
                    p75=float(np.percentile(vals, 75)),
                    frac_sat_only=float((vals < cfg.floor_threshold).mean()),
                    n=int(vals.size),
                ))
    df = pd.DataFrame(rows)
    cfg.timeseries_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.timeseries_csv, index=False)
    print(f"Saved {cfg.timeseries_csv}")
    return df


def average_spatial(cfg: Config, region: Region, month: int,
                    year_range: tuple[int, int]) -> tuple[np.ndarray, rasterio.Affine]:
    arrays = []
    transform = None
    for year in range(year_range[0], year_range[1] + 1):
        path = download_if_missing(year, month, cfg.cache_dir)
        arr, tr = read_region_array(path, region)
        if transform is None:
            transform = tr
        arrays.append(arr)
    return np.nanmean(np.stack(arrays, axis=0), axis=0), transform


STYLE = {
    "West Africa": dict(color="#B23A48", marker="o", lw=2.4, label="West Africa"),
    "NW US":       dict(color="#2E5339", marker="s", lw=1.4, ls="--",
                        label="NW US (well-gauged reference)"),
}


def month_name(m: int) -> str:
    return ["", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"][m]


def plot_timeseries(ax, df: pd.DataFrame, cfg: Config,
                    month: int, panel_label: str, subtitle: str) -> None:
    sub_month = df[df["month"] == month]
    for region_name, s in STYLE.items():
        sub = sub_month[sub_month["region"] == region_name].sort_values("year")
        if sub.empty:
            continue
        ax.fill_between(sub["year"], sub["p25"], sub["p75"],
                        color=s["color"], alpha=0.15, linewidth=0)
        ax.plot(sub["year"], sub["median"], **s)
    ax.axhline(cfg.floor_exact, color="#777", linestyle=":", linewidth=1.0)
    ax.text(cfg.years[0] + 0.5, cfg.floor_exact - 0.012,
            "satellite-only value (0.25)",
            fontsize=12, color="#555", va="top")
    ax.set_xlim(cfg.years[0] - 1, cfg.years[-1] + 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel(f"Median {month_name(month)} R²", fontsize=14)
    ax.set_title(f"{panel_label}. {subtitle}", fontsize=15, pad=8)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0, 0.015, 1, 0.08),
        mode="expand",
        borderaxespad=0,
        fontsize=10.5,
        framealpha=0.95,
        handletextpad=0.4,
        columnspacing=0.8,
    )


def plot_spatial(ax, grid: np.ndarray, transform: rasterio.Affine,
                 region: Region, panel_label: str,
                 subtitle: str, title_pad: float = 6) -> "plt.cm.ScalarMappable":
    h, w = grid.shape
    left = transform.c
    top = transform.f
    right = left + transform.a * w
    bottom = top + transform.e * h
    extent = (left, right, bottom, top)
    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    im = ax.imshow(grid, extent=extent, origin="upper",
                   cmap=cmap, norm=norm,
                   transform=ccrs.PlateCarree(),
                   interpolation="none", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe0ee", edgecolor="none", zorder=2)
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


def render_combined_figure(cfg: Config, ts: pd.DataFrame) -> None:
    """Single Figure 2: A,B timeseries (top) + C,D spatial maps (bottom)."""
    print(f"Pulling July spatial data, "
          f"{cfg.early_window[0]}–{cfg.early_window[1]}...")
    early_grid, early_t = average_spatial(cfg, WA, cfg.spatial_month, cfg.early_window)
    print(f"Pulling July spatial data, "
          f"{cfg.late_window[0]}–{cfg.late_window[1]}...")
    late_grid, late_t = average_spatial(cfg, WA, cfg.spatial_month, cfg.late_window)

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[5.0, 6.4, 0.30], width_ratios=[1, 1],
        hspace=0.32, wspace=0.14,
        left=0.07, right=0.95, top=0.91, bottom=0.05,
    )

    # Top row: timeseries (A July wet season, B January dry season)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    plot_timeseries(ax_a, ts, cfg, month=7, panel_label="A",
                    subtitle="July — West African wet season")
    plot_timeseries(ax_b, ts, cfg, month=1, panel_label="B",
                    subtitle="January — dry season")

    # Bottom row: spatial maps over West Africa (C early epoch, D late epoch)
    ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_d = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    plot_spatial(ax_c, early_grid, early_t, WA, "C",
                 f"Mean July R², {cfg.early_window[0]}–{cfg.early_window[1]}",
                 title_pad=10)
    im_d = plot_spatial(ax_d, late_grid, late_t, WA, "D",
                        f"Mean July R², {cfg.late_window[0]}–{cfg.late_window[1]}",
                        title_pad=10)

    # Shared colorbar for the two maps
    cax = fig.add_subplot(gs[2, :])
    cb = fig.colorbar(im_d, cax=cax, orientation="horizontal",
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Mean July R²", fontsize=10)

    fig.suptitle(
        "CHIRPS v3.0 station-blending R² — West Africa "
        "(A, B timeseries vs NW US reference;  C, D July maps, early vs late)",
        fontsize=14, y=0.965,
    )
    cfg.out_combined.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out_combined, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cfg.out_combined}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Summary (median R²) ===")
    for region in df["region"].unique():
        for month in (1, 7):
            sub = df[(df["region"] == region) &
                     (df["month"] == month)].sort_values("year")
            if sub.empty:
                continue
            first, last = sub.iloc[0], sub.iloc[-1]
            print(
                f"{region:14s}  {month_name(month)[:3]}  "
                f"{int(first['year'])}: {first['median']:.2f}  "
                f"->  {int(last['year'])}: {last['median']:.2f}"
            )


def main(cfg: Config = CONFIG) -> None:
    print("Pulling time-series data (Jan + Jul)...")
    ts = collect_time_series(cfg)
    render_combined_figure(cfg, ts)
    print_summary(ts)


if __name__ == "__main__":
    main()
