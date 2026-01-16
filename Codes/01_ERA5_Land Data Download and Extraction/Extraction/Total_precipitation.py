#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5-Land hourly 'tp'  ->  Nepal-local (UTC+05:45) basin precipitation with AREA WEIGHTING
------------------------------------------------------------------------------------------------
What this script does
- Reads ERA5-Land hourly accumulated precipitation ('tp', meters).
- De-accumulates to hourly increments (mm).
- Splits each UTC hour into 4×15-min chunks (mass-conserving) for exact local-day accounting.
- Shifts to Nepal Standard Time (UTC+05:45), then resamples to local DAILY totals (mm) per grid.
- Computes basin-average daily precipitation depth (mm) using **area weights** derived from
  fractional overlap of each grid cell with the basin polygon (computed in an equal-area CRS).
- Writes per-grid daily CSV (needed for SWAT), basin-daily CSV, monthly totals (Year×Month),
  and monthly climatology. Also produces QA plots (daily series, year–month heatmap),
  plus a FRACTIONAL OVERLAP map (0–1) to visualize the masking/weights.

Key points
- Precipitation is **extensive**: basin-average depth must be **area-weighted** (km²), not just
  equal-weight or cosine(lat). We compute overlap area per cell in EPSG:6933 and weight by it.
- Fraction (0–1) is used for QA visualization; **area (km²)** is used for weighting.

Requirements
- netCDF4, numpy, pandas, geopandas, shapely, matplotlib

Author: (adapted for Diwakar Adhikari)
"""

import os
import glob
from datetime import datetime as dt

import netCDF4
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ------------------------------- #
# Configuration
# ------------------------------- #
BASIN_SHP = r"C:/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/Shp file/Tarakeshwor.shp"  # your basin
NC_DIR    = r"C:/Users/Diwakar Adhikari/Documents/ERA5_Land/precipitation"                            # ERA5-Land tp
OUT_DIR   = r"C:/Users/Diwakar Adhikari/Downloads/Boko Thesis"  # outputs

# Year/file pattern (adjust as needed)
YEARS  = [str(y) for y in range(1984, 2025)]
HALVES = ["H1p", "H2p"]  # files like 1985H1p.nc, 1985H2p.nc

# Local time: Nepal Standard Time
LOCAL_TZ_OFFSET_MIN = 5 * 60 + 45  # +05:45

# Outputs
OUT_DAILY_GRID_CSV       = os.path.join(OUT_DIR, "era5landa_daily_grid_NST_BULK.csv")
OUT_DAILY_BASIN_CSV      = os.path.join(OUT_DIR, "era5landa_daily_basin_NST_BULK.csv")
OUT_MONTHLY_WIDE_BASIN   = os.path.join(OUT_DIR, "era5landa_monthly_basin_NST_BULK.csv")
OUT_MONTHLY_CLIMO_BASIN  = os.path.join(OUT_DIR, "era5landa_monthly_climo_basin_NST_BULK.csv")
FIG_DAILY_SERIES         = os.path.join(OUT_DIR, "figa_basin_daily_series_NST_BULK.png")
FIG_YEAR_MONTH_HEATMAP   = os.path.join(OUT_DIR, "figa_basin_year_month_heatmap_NST_BULK.png")
FIG_CLIMO_MONTHLY_CYCLE  = os.path.join(OUT_DIR, "figa_basin_climatology_monthly_cycle_NST_BULK.png")
FIG_FRACTION_WEIGHTS_PNG = os.path.join(OUT_DIR, "fractional_overlap_weights_NST.png")  # QA map

# CRS choices
GEOGRAPHIC_CRS  = "EPSG:4326"  # lon/lat degrees
EQUAL_AREA_CRS  = "EPSG:6933"  # global equal-area for robust area math

# Plot annotations on the fraction map (disable for very large grids)
PLOT_ANNOTATE = True
MAX_CELLS_TO_ANNOTATE = 1200

os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200})

# ------------------------------- #
# Helpers: time & file ordering
# ------------------------------- #
def robust_num2date(time_var):
    """Parse time variable and return tz-aware UTC DatetimeIndex (hourly)."""
    vals = time_var[:]
    units = time_var.units
    cal = getattr(time_var, 'calendar', 'standard')
    cf_times = netCDF4.num2date(vals, units=units, calendar=cal, only_use_cftime_datetimes=True)
    pd_times = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day, getattr(t, "hour", 0)) for t in cf_times])
    return pd_times.tz_localize("UTC")

def sort_files_by_start_time(nc_dir, years, halves):
    """Return sorted list of NetCDF file paths by their first timestamp."""
    paths = []
    for y in years:
        for h in halves:
            paths.extend(glob.glob(os.path.join(nc_dir, f"{y}{h}.nc")))
    # optionally include next year halves if needed for boundaries (not strictly required here)
    file_list = []
    for p in paths:
        try:
            with netCDF4.Dataset(p, 'r') as ds:
                tvar = ds.variables.get('valid_time', ds.variables.get('time'))
                if tvar is None:
                    continue
                times = robust_num2date(tvar)
                if len(times) == 0:
                    continue
                file_list.append((times[0], p))
        except Exception:
            continue
    file_list.sort(key=lambda x: x[0])
    return [p for _, p in file_list]

# ------------------------------- #
# Helpers: grid geometry & weights
# ------------------------------- #
def infer_edges(coords: np.ndarray) -> np.ndarray:
    """Infer cell edge coordinates from center coordinates (ascending or descending)."""
    coords = np.asarray(coords, dtype=float)
    edges = np.empty(len(coords) + 1, dtype=float)
    edges[1:-1] = (coords[:-1] + coords[1:]) / 2.0
    first_step = coords[1] - coords[0]
    last_step  = coords[-1] - coords[-2]
    edges[0]   = coords[0] - first_step / 2.0
    edges[-1]  = coords[-1] + last_step  / 2.0
    return edges

def build_cell_polygons(lats, lons, lat_edges=None, lon_edges=None):
    """Build shapely Polygons for each lat/lon cell using edges. Returns list[Polygon], (nlat,nlon)."""
    nlat, nlon = len(lats), len(lons)
    if lat_edges is None: lat_edges = infer_edges(lats)
    if lon_edges is None: lon_edges = infer_edges(lons)
    polys = []
    for j in range(nlat):
        lat_min = min(lat_edges[j], lat_edges[j+1])
        lat_max = max(lat_edges[j], lat_edges[j+1])
        for i in range(nlon):
            lon_min = min(lon_edges[i], lon_edges[i+1])
            lon_max = max(lon_edges[i], lon_edges[i+1])
            polys.append(Polygon([(lon_min, lat_min),
                                  (lon_max, lat_min),
                                  (lon_max, lat_max),
                                  (lon_min, lat_max)]))
    return polys, (nlat, nlon)

def compute_overlap_area_and_fraction(basin_gdf_geo, lats, lons, lat_edges=None, lon_edges=None,
                                      equal_area_crs=EQUAL_AREA_CRS):
    """
    Compute per-cell overlap area (km²) and fraction (0..1):
      area_in = area(cell ∩ basin) in km² (equal-area CRS)
      frac    = area_in / area_cell (0..1)
    Returns:
      areas_in_km2 [nlat,nlon], frac_mat [nlat,nlon]
    """
    # Reproject basin to equal-area CRS
    basin_eq = basin_gdf_geo.to_crs(equal_area_crs)
    basin_union_eq = basin_eq.unary_union

    # Build cell polygons in lon/lat, then reproject to equal-area CRS
    cell_polys_ll, shape = build_cell_polygons(lats, lons, lat_edges, lon_edges)
    cells_ll = gpd.GeoDataFrame(geometry=cell_polys_ll, crs=GEOGRAPHIC_CRS)
    cells_eq = cells_ll.to_crs(equal_area_crs)

    areas_in = np.zeros(len(cells_eq), dtype=float)
    fracs    = np.zeros(len(cells_eq), dtype=float)

    for idx, poly_eq in enumerate(cells_eq.geometry.values):
        area_cell = poly_eq.area                     # m² (in equal-area CRS units)
        if area_cell <= 0:
            areas_in[idx] = 0.0
            fracs[idx]    = 0.0
            continue
        inter = poly_eq.intersection(basin_union_eq)
        area_in = inter.area if not inter.is_empty else 0.0  # m²
        areas_in[idx] = area_in / 1e6                        # -> km²
        fracs[idx]    = (area_in / area_cell) if area_cell > 0 else 0.0

    areas_in_km2 = areas_in.reshape(shape)
    frac_mat     = fracs.reshape(shape)
    return areas_in_km2, frac_mat

def build_grid_keys(lats, lons):
    """Return grid column keys matching our naming convention."""
    keys = []
    for j in range(len(lats)):
        for i in range(len(lons)):
            keys.append(f"lat{float(lats[j]):.3f}_lon{float(lons[i]):.3f}")
    return np.array(keys).reshape(len(lats), len(lons))

# ------------------------------- #
# Helpers: plotting
# ------------------------------- #
def plot_fraction_weights(frac_mat, lats, lons, basin_geo, lat_edges=None, lon_edges=None,
                          annotate=True, out_path=None, max_cells_to_annotate=1200):
    """
    Visualize fractional overlap (0..1) per grid cell in lon/lat, overlay basin boundary.
    """
    # Edges
    if lat_edges is None: lat_edges = infer_edges(lats)
    if lon_edges is None: lon_edges = infer_edges(lons)
    LonE, LatE = np.meshgrid(lon_edges, lat_edges)

    # Mask zeros
    data_plot = np.where(frac_mat > 0, frac_mat, np.nan)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(LonE, LatE, data_plot, cmap="viridis", vmin=0.0, vmax=1.0, shading="auto")

    # Basin outline
    try:
        basin_geo.boundary.plot(ax=ax, color="black", linewidth=1.0)
    except Exception:
        pass

    # Optional annotations
    if annotate:
        LonC, LatC = np.meshgrid(lons, lats)
        n_cells_to_annotate = int((frac_mat > 0).sum())
        if n_cells_to_annotate <= max_cells_to_annotate:
            for j in range(frac_mat.shape[0]):
                for i in range(frac_mat.shape[1]):
                    f = float(frac_mat[j, i])
                    if f > 0:
                        ax.text(LonC[j, i], LatC[j, i], f"{f:.2f}",
                                ha="center", va="center", fontsize=7, color="white",
                                path_effects=[pe.withStroke(linewidth=1, foreground="black")])
        else:
            ax.text(0.01, 0.99,
                    f"Annotations disabled (too many cells: {n_cells_to_annotate})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=9, color="darkred",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax.set_title("Fractional Overlap Weights (0–1) – Masked Grid Inside Basin")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Overlap Fraction (area inside basin / cell area)")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Saved fractional overlap plot to: {out_path}")
    plt.show()
    return fig, ax

# ------------------------------- #
# Core precipitation data pipeline
# ------------------------------- #
def load_all_hourly_accumulated(file_paths, grid_coords, lats, lons):
    """
    Load selected cells from all files and return DataFrame of hourly accumulated 'tp' (meters).
    Returns a DataFrame with UTC DatetimeIndex and columns = selected grid cells (order preserved).
    """
    lon_len = len(lons)
    sel_flat_idx = np.array([i * lon_len + j for (i, j) in grid_coords], dtype=np.int64)
    all_times, all_data = [], []

    print("Step 1: Loading hourly accumulated 'tp' (m) for selected grid cells...")
    for path in tqdm(file_paths, desc="Loading files"):
        with netCDF4.Dataset(path, 'r') as ds:
            tvar = ds.variables.get('valid_time', ds.variables.get('time'))
            if tvar is None:
                continue
            times = robust_num2date(tvar)  # tz-aware UTC
            tp_m = ds.variables['tp'][:]   # [time, lat, lon] in meters
            for i in range(len(times)):
                flat = tp_m[i, :, :].ravel()
                tp_sel = flat[sel_flat_idx]
                all_times.append(times[i])
                all_data.append(tp_sel)

    if not all_data:
        raise RuntimeError("No data loaded from provided NetCDF files.")

    all_data = np.vstack(all_data)  # shape: [n_hours, n_grids]
    df = pd.DataFrame(all_data, index=all_times)
    df = df.sort_index()
    print(f" ✓ Loaded {len(df)} hourly records for {all_data.shape[1]} selected grid cells.")
    return df

def deaccumulate_hourly(df_accum):
    """
    Convert accumulated 'tp' (m) to hourly increments (mm).
    ERA5-Land tp is accumulation with end-of-period labeling; treatment here:
    - For hours 2..23: increment = value(t) - value(t-1)
    - For hour 1: keep as-is (first hour of UTC day)
    - First row: set NaN unless hour==1
    - Clip tiny negatives to 0; drop remaining negatives/NaNs
    """
    print("Step 2: De-accumulating hourly data...")
    df_inc = df_accum.copy() * 1000.0  # m -> mm
    hours = df_inc.index.hour

    for i in range(1, len(df_inc)):
        if hours[i] == 1:
            # First hour of UTC day: leave as-is (represents the last hour's accumulation)
            pass
        else:
            df_inc.iloc[i] = df_inc.iloc[i] - df_accum.iloc[i - 1] * 1000.0

    # First row rule
    if hours[0] != 1:
        df_inc.iloc[0] = np.nan

    # Numerical hygiene
    df_inc[(df_inc < 0) & (df_inc.abs() < 1e-6)] = 0.0
    df_inc = df_inc.dropna()
    df_inc = df_inc[(df_inc >= 0).all(axis=1)]
    print(f" ✓ De-accumulated to {len(df_inc)} hourly increments (mm).")
    print(f" ✓ Total precip (UTC hourly): {df_inc.sum().sum():.2f} mm")
    return df_inc

def split_to_15min_chunks(df_hourly):
    """
    Split each hourly value into 4×15-minute chunks for exact local-day alignment.
    Sums remain conserved.
    """
    print("Step 3: Splitting each hour into 4×15-minute chunks...")
    chunks_list, times_list = [], []
    for idx in df_hourly.index:
        quarter = df_hourly.loc[idx] / 4.0
        for offset_min in [45, 30, 15, 0]:
            chunk_time = idx - pd.Timedelta(minutes=offset_min)
            times_list.append(chunk_time)
            chunks_list.append(quarter.values)
    chunks_df = pd.DataFrame(chunks_list, index=times_list, columns=None)
    chunks_df = chunks_df.sort_index()
    print(f" ✓ Created {len(chunks_df)} 15-minute chunks.")
    print(f" ✓ Total precip (15-min chunks): {chunks_df.sum().sum():.2f} mm")
    return chunks_df

def shift_to_local_time(df_chunks, offset_minutes):
    """Shift to local NST (+05:45) and drop tz info for clean daily resampling."""
    print(f"Step 4: Shifting to local time (UTC+{offset_minutes//60}:{offset_minutes%60:02d})...")
    df_local = df_chunks.copy()
    df_local.index = df_local.index + pd.Timedelta(minutes=offset_minutes)
    df_local.index = df_local.index.tz_localize(None)  # naive timestamps in local clock
    print(f" ✓ Shifted to NST: {df_local.index[0]} to {df_local.index[-1]}")
    print(f" ✓ Total precip (NST 15-min): {df_local.sum().sum():.2f} mm")
    return df_local

def resample_to_daily(df_15min):
    """Sum 15-minute local data to daily totals (mm)."""
    print("Step 5: Resampling to daily NST totals...")
    df_daily = df_15min.resample('D').sum()
    print(f" ✓ Resampled to {len(df_daily)} daily records.")
    print(f" ✓ Total precip (NST daily): {df_daily.sum().sum():.2f} mm")
    return df_daily

# ------------------------------- #
# Main
# ------------------------------- #
def main():
    print("=" * 78)
    print("ERA5-Land Precipitation (tp) -> Local (NST) Daily with AREA WEIGHTING")
    print("=" * 78)

    # Load basin geometry (in geographic CRS for plotting; equal-area used internally)
    basin_geo = gpd.read_file(BASIN_SHP)
    if basin_geo.crs is None:
        basin_geo = basin_geo.set_crs(GEOGRAPHIC_CRS)
    basin_geo = basin_geo.to_crs(GEOGRAPHIC_CRS)

    # Find the first file to obtain the grid
    first_file = None
    for y in YEARS:
        candidates = (glob.glob(os.path.join(NC_DIR, f"{y}H1p.nc")) or
                      glob.glob(os.path.join(NC_DIR, f"{y}H2p.nc")))
        if candidates:
            first_file = candidates[0]
            break
    if first_file is None:
        raise FileNotFoundError("No ERA5-Land NetCDF files found to read grid.")

    with netCDF4.Dataset(first_file, 'r') as ds0:
        lats = ds0.variables['latitude'][:]
        lons = ds0.variables['longitude'][:]
        # Use bounds if present; else infer edges
        lat_bnds = ds0.variables.get('lat_bnds') or ds0.variables.get('latitude_bnds')
        lon_bnds = ds0.variables.get('lon_bnds') or ds0.variables.get('longitude_bnds')

        if lat_bnds is not None and getattr(lat_bnds[:], "shape", (0,))[-1] == 2 and len(lat_bnds[:]) == len(lats):
            lat_edges = np.concatenate(([lat_bnds[0, 0]], lat_bnds[:, 1]))
        else:
            lat_edges = infer_edges(lats)

        if lon_bnds is not None and getattr(lon_bnds[:], "shape", (0,))[-1] == 2 and len(lon_bnds[:]) == len(lons):
            lon_edges = np.concatenate(([lon_bnds[0, 0]], lon_bnds[:, 1]))
        else:
            lon_edges = infer_edges(lons)

    # Compute overlap AREA (km²) and FRACTION (0–1)
    print("Computing overlap area (km²) and fraction (0–1) per grid cell...")
    areas_in_km2, frac_mat = compute_overlap_area_and_fraction(
        basin_gdf_geo=basin_geo,
        lats=lats, lons=lons,
        lat_edges=lat_edges, lon_edges=lon_edges,
        equal_area_crs=EQUAL_AREA_CRS
    )

    # Build grid keys and select cells with area_in > 0
    grid_keys_matrix = build_grid_keys(lats, lons)
    selected_mask = areas_in_km2 > 0.0
    selected_indices = np.argwhere(selected_mask)  # list of (j,i)
    selected_keys = grid_keys_matrix[selected_mask].ravel()

    # AREA weights (km²) as a Series keyed by column name
    weights_area = {grid_keys_matrix[j, i]: float(areas_in_km2[j, i])
                    for (j, i) in selected_indices}
    weights_area_series = pd.Series(weights_area)
    A_total_km2 = float(weights_area_series.sum())
    if A_total_km2 <= 0:
        raise RuntimeError("Total overlap area is zero—check shapefile/grid alignment.")

    print(f" ✓ Selected {len(selected_indices)} grid cells with positive overlap area.")
    print(f" ✓ Total basin area captured by grid (km²): {A_total_km2:,.2f}")

    # QA figure: fractional overlap map (0–1)
    try:
        plot_fraction_weights(frac_mat=frac_mat, lats=lats, lons=lons,
                              basin_geo=basin_geo,
                              lat_edges=lat_edges, lon_edges=lon_edges,
                              annotate=PLOT_ANNOTATE, out_path=FIG_FRACTION_WEIGHTS_PNG,
                              max_cells_to_annotate=MAX_CELLS_TO_ANNOTATE)
    except Exception as e:
        print(f"Fractional overlap plot failed: {e}")

    # Build sorted file list
    files_sorted = sort_files_by_start_time(NC_DIR, YEARS, HALVES)
    if not files_sorted:
        raise FileNotFoundError("No ERA5-Land NetCDF files found for processing.")
    print(f" ✓ Found {len(files_sorted)} NetCDF files to process\n")

    # Load only the selected cells (positive overlap) from all files
    df_accum = load_all_hourly_accumulated(files_sorted, selected_indices, lats, lons)

    # De-accumulate to hourly increments (mm)
    df_hourly_utc = deaccumulate_hourly(df_accum)

    # Split to 15-min chunks, shift to NST, resample to daily
    df_15min_utc = split_to_15min_chunks(df_hourly_utc)
    df_15min_nst = shift_to_local_time(df_15min_utc, LOCAL_TZ_OFFSET_MIN)
    df_daily_grid = resample_to_daily(df_15min_nst)

    # Mass conservation check (hourly vs daily totals)
    print("\n" + "=" * 78)
    print("QUALITY CHECK – Mass Conservation")
    total_hourly = df_hourly_utc.sum().sum()
    total_daily  = df_daily_grid.sum().sum()
    diff = abs(total_hourly - total_daily)
    print(f" Total (UTC hourly): {total_hourly:.2f} mm")
    print(f" Total (NST daily):  {total_daily:.2f} mm")
    print(f" Difference:         {diff:.6f} mm ({diff/total_hourly*100:.4f}%)")
    if diff < 0.01:
        print(" ✓ PASS – Mass is conserved!")
    else:
        print(" ⚠ WARNING – Significant mass difference detected!")
    print("=" * 78 + "\n")

    # Name columns with grid keys (selected subset)
    df_daily_grid.columns = selected_keys
    df_daily_grid.index.name = "Date_NST"

    # ------------------ Basin mean (AREA-WEIGHTED depth, mm) ------------------
    # Align area weights to columns; compute area-weighted mean depth per day
    w = weights_area_series.reindex(df_daily_grid.columns).fillna(0.0)  # km²
    denom = w.sum()
    if denom <= 0:
        raise RuntimeError("Area-weight denominator is zero after alignment.")
    basin_daily_mm = (df_daily_grid * w).sum(axis=1) / denom
    basin_daily_mm.name = "BasinMean_mm"
    print(" ✓ Basin daily mean precipitation (area-weighted) computed.\n")

    # ------------------ SAVE OUTPUTS ------------------
    print("Saving outputs...")
    # Grid daily (per-grid) – for SWAT
    df_daily_grid.to_csv(OUT_DAILY_GRID_CSV)

    # Basin daily
    basin_daily_mm.to_frame().to_csv(OUT_DAILY_BASIN_CSV, index_label="Date_NST")

    # Monthly totals (sum of local daily basin means)
    monthly_basin = basin_daily_mm.resample("MS").sum()
    mb = monthly_basin.to_frame(name="MonthlyTotal_mm")
    mb["Year"] = mb.index.year
    mb["Month"] = mb.index.month
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    wide = mb.pivot(index="Year", columns="Month", values="MonthlyTotal_mm").sort_index()
    wide.rename(columns=names, inplace=True)
    wide = wide.reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    wide.to_csv(OUT_MONTHLY_WIDE_BASIN, index_label="year")

    # Climatology (mean of monthly totals across years)
    climo = mb.groupby("Month")["MonthlyTotal_mm"].mean().rename(index=names)
    climo.to_csv(OUT_MONTHLY_CLIMO_BASIN, index_label="Month", header=["ClimatologyMonthlyTotal_mm"])

    print(f" ✓ Saved:\n   - Per-grid daily CSV  : {OUT_DAILY_GRID_CSV}\n"
          f"   - Basin daily CSV     : {OUT_DAILY_BASIN_CSV}\n"
          f"   - Monthly totals CSV  : {OUT_MONTHLY_WIDE_BASIN}\n"
          f"   - Monthly climo CSV   : {OUT_MONTHLY_CLIMO_BASIN}\n")

    # ------------------ PLOTS ------------------
    print("Creating plots...")
    plt.style.use("seaborn-v0_8")

    # (1) Daily basin series
    fig1, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(basin_daily_mm.index, basin_daily_mm.values, color="tab:blue", lw=0.8)
    ax1.set_title("Nepal-local daily precipitation – Basin mean (ERA5-Land, area-weighted)")
    ax1.set_xlabel("Date (NST)")
    ax1.set_ylabel("Precipitation (mm)")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(FIG_DAILY_SERIES)

    # (2) Year–Month heatmap of monthly totals
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im = ax2.imshow(wide.values, aspect="auto", cmap="Blues")
    ax2.set_title("Monthly Total Precipitation (mm) – Basin mean (ERA5-Land, NST)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Year")
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(list(wide.columns))
    ax2.set_yticks(range(len(wide.index)))
    ax2.set_yticklabels(wide.index)
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.ax.set_ylabel("mm", rotation=90)
    fig2.tight_layout()
    fig2.savefig(FIG_YEAR_MONTH_HEATMAP)

    # (3) Climatological monthly cycle
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    ax3.plot(climo.index, climo.values, marker="o", color="tab:green")
    ax3.set_title("Climatological Monthly Total – Basin mean (ERA5-Land, NST)")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Precipitation (mm)")
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(FIG_CLIMO_MONTHLY_CYCLE)

    plt.show()
    print(f" ✓ Saved figures in: {OUT_DIR}\n")
    print("=" * 78)
    print("PROCESSING COMPLETE!")
    print("=" * 78)

if __name__ == "__main__":
    main()