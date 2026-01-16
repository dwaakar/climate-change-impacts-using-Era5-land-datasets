
# -*- coding: utf-8 -*-
"""
ERA5-Land hourly SSRD (J/m^2) -> Local daily (UTC+06:00) MJ/m^2/day
with fractional-overlap basin averaging (area-based), no 15-min chunking.

What this script does
- Reads ERA5-Land hourly SSRD (surface solar radiation downwards), accumulated energy [J/m^2].
- De-accumulates to hourly energy [J/m^2] per ECMWF conversion table (01 UTC special case).
- Shifts timestamps by +06:00 (whole hour), then sums hourly energy per local calendar day.
- Converts daily energy to MJ/m^2/day (J -> MJ).
- Computes basin mean using fractional overlap (area-based) weights.
- Saves per-grid daily, basin daily, monthly mean of daily, and monthly climatology.
- Provides simple QA plots and mass-conservation check.

References:
- ECMWF Confluence: Conversion table for accumulated variables (ERA5-Land SSR hourly: diff to get hourly) 
  https://confluence.ecmwf.int/pages/viewpage.action?pageId=197702790
- ECMWF parameter DB (ssrd): accumulated energy in J/m^2; divide by seconds to get W/m^2
  https://codes.ecmwf.int/grib/param-db/169
"""

import os
import glob
import numpy as np
import pandas as pd
import netCDF4
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- Configuration ----------------
BASIN_SHP = r"C:/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/Shp file/MRB.shp"
NC_DIR_SOLAR = r"C:/Users/Diwakar Adhikari/Downloads/SWAT+/ERA5_Land/solar_radiation"
OUT_DIR = os.path.join(NC_DIR_SOLAR, "Boko Thesis - UTC+06 Solar")
os.makedirs(OUT_DIR, exist_ok=True)

YEARS  = [str(y) for y in range(1984, 2025)]
HALVES = ["H1_ssrd", "H2_ssrd"]  # e.g., 1985H1_ssrd.nc, 1985H2_ssrd.nc

# Local time shift (pseudo-NST using whole hour)
LOCAL_TZ_OFFSET_MIN = 6 * 60  # +06:00

# CRS choices
GEOGRAPHIC_CRS  = "EPSG:4326"
EQUAL_AREA_CRS  = "EPSG:6933"

# Outputs
OUT_DAILY_GRID_CSV   = os.path.join(OUT_DIR, "ssrd_daily_grid_UTCp6_MJm2.csv")
OUT_DAILY_BASIN_CSV  = os.path.join(OUT_DIR, "ssrd_daily_basin_UTCp6_MJm2.csv")
OUT_MONTHLY_WIDE_CSV = os.path.join(OUT_DIR, "ssrd_monthly_mean_daily_basin_UTCp6_MJm2.csv")
OUT_CLIMO_CSV        = os.path.join(OUT_DIR, "ssrd_climatology_monthly_mean_daily_basin_UTCp6_MJm2.csv")

FIG_DAILY_SERIES     = os.path.join(OUT_DIR, "fig_ssrd_daily_basin_UTCp6.png")
FIG_CLIMO_CYCLE      = os.path.join(OUT_DIR, "fig_ssrd_climo_cycle_UTCp6.png")

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200})

# ---------------- Helpers ----------------
def robust_num2date(time_var):
    """Parse time variable and return tz-aware UTC DatetimeIndex (hourly)."""
    vals = time_var[:]
    units = time_var.units
    cal   = getattr(time_var, 'calendar', 'standard')
    cf_times = netCDF4.num2date(vals, units=units, calendar=cal, only_use_cftime_datetimes=True)
    pd_times = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day, getattr(t, "hour", 0)) for t in cf_times])
    return pd_times.tz_localize("UTC")

def infer_edges(coords):
    """Infer cell edge coordinates from center coordinates (ascending or descending)."""
    coords = np.asarray(coords, dtype=float)
    edges = np.empty(len(coords)+1, dtype=float)
    edges[1:-1] = (coords[:-1] + coords[1:]) / 2.0
    first_step = coords[1] - coords[0]
    last_step  = coords[-1] - coords[-2]
    edges[0]   = coords[0] - first_step/2.0
    edges[-1]  = coords[-1] + last_step/2.0
    return edges

def build_cell_polygons(lats, lons, lat_edges=None, lon_edges=None):
    """Build shapely Polygons for each lat/lon cell using edges."""
    nlat, nlon = len(lats), len(lons)
    if lat_edges is None: lat_edges = infer_edges(lats)
    if lon_edges is None: lon_edges = infer_edges(lons)
    polys = []
    for j in range(nlat):
        lat_min = min(lat_edges[j],   lat_edges[j+1])
        lat_max = max(lat_edges[j],   lat_edges[j+1])
        for i in range(nlon):
            lon_min = min(lon_edges[i], lon_edges[i+1])
            lon_max = max(lon_edges[i], lon_edges[i+1])
            polys.append(Polygon([(lon_min, lat_min),
                                  (lon_max, lat_min),
                                  (lon_max, lat_max),
                                  (lon_min, lat_max)]))
    return polys, (nlat, nlon)

def compute_overlap_area_and_fraction(basin_gdf_geo, lats, lons, lat_edges=None, lon_edges=None, equal_area_crs=EQUAL_AREA_CRS):
    """Compute per-cell overlap area (km²) and fraction (0..1) in equal-area CRS."""
    basin_eq = basin_gdf_geo.to_crs(equal_area_crs)
    basin_union_eq = basin_eq.unary_union

    cell_polys_ll, shape = build_cell_polygons(lats, lons, lat_edges, lon_edges)
    cells_ll = gpd.GeoDataFrame(geometry=cell_polys_ll, crs=GEOGRAPHIC_CRS)
    cells_eq = cells_ll.to_crs(equal_area_crs)

    areas_in = np.zeros(len(cells_eq), dtype=float)
    for idx, poly_eq in enumerate(cells_eq.geometry.values):
        area_cell = poly_eq.area  # m²
        if area_cell <= 0:
            continue
        inter = poly_eq.intersection(basin_union_eq)
        area_in = inter.area if not inter.is_empty else 0.0  # m²
        areas_in[idx] = area_in / 1e6  # km²

    return areas_in.reshape(shape)

def build_grid_keys(lats, lons):
    """Return grid column keys 'lat<lat>_lon<lon>'."""
    keys = []
    for j in range(len(lats)):
        for i in range(len(lons)):
            keys.append(f"lat{float(lats[j]):.3f}_lon{float(lons[i]):.3f}")
    return np.array(keys).reshape(len(lats), len(lons))

def sort_files_by_start_time(nc_dir, years, halves):
    paths = []
    for y in years:
        for h in halves:
            paths.extend(glob.glob(os.path.join(nc_dir, f"{y}{h}.nc")))
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

# ---------------- Core pipeline ----------------
def main():
    print("="*78)
    print("ERA5-Land SSRD -> Local daily (UTC+06:00) MJ/m^2/day with area-weighted basin mean")
    print("="*78)

    # Basin geometry
    basin_geo = gpd.read_file(BASIN_SHP)
    if basin_geo.crs is None:
        basin_geo = basin_geo.set_crs(GEOGRAPHIC_CRS)
    basin_geo = basin_geo.to_crs(GEOGRAPHIC_CRS)

    # Grid from first file
    first_file = None
    for y in YEARS:
        candidates = glob.glob(os.path.join(NC_DIR_SOLAR, f"{y}{HALVES[0]}.nc")) or \
                     glob.glob(os.path.join(NC_DIR_SOLAR, f"{y}{HALVES[1]}.nc"))
        if candidates:
            first_file = candidates[0]
            break
    if first_file is None:
        raise FileNotFoundError("No ERA5-Land SSRD NetCDF files found to read grid.")

    with netCDF4.Dataset(first_file, 'r') as ds0:
        lats = ds0.variables['latitude'][:]
        lons = ds0.variables['longitude'][:]
        # Optional bounds
        lat_bnds = ds0.variables.get('lat_bnds') or ds0.variables.get('latitude_bnds')
        lon_bnds = ds0.variables.get('lon_bnds') or ds0.variables.get('longitude_bnds')
        if lat_bnds is not None and getattr(lat_bnds[:], "shape", (0,))[ -1 ] == 2 and len(lat_bnds[:]) == len(lats):
            lat_edges = np.concatenate(([lat_bnds[0,0]], lat_bnds[:,1]))
        else:
            lat_edges = infer_edges(lats)
        if lon_bnds is not None and getattr(lon_bnds[:], "shape", (0,))[ -1 ] == 2 and len(lon_bnds[:]) == len(lons):
            lon_edges = np.concatenate(([lon_bnds[0,0]], lon_bnds[:,1]))
        else:
            lon_edges = infer_edges(lons)

    # Overlap area (km²) per cell
    print("Computing per-cell overlap area (km²)...")
    areas_in_km2 = compute_overlap_area_and_fraction(
        basin_gdf_geo=basin_geo, lats=lats, lons=lons, lat_edges=lat_edges, lon_edges=lon_edges,
        equal_area_crs=EQUAL_AREA_CRS
    )
    grid_keys_mat = build_grid_keys(lats, lons)
    selected_mask   = areas_in_km2 > 0.0
    selected_indices = np.argwhere(selected_mask)
    selected_keys    = grid_keys_mat[selected_mask].ravel()
    weights_area     = { grid_keys_mat[j,i]: float(areas_in_km2[j,i]) for (j,i) in selected_indices }
    w_area_series    = pd.Series(weights_area)
    A_total_km2      = float(w_area_series.sum())
    print(f" ✓ Selected {len(selected_indices)} overlapping cells; total captured basin area: {A_total_km2:,.2f} km²")

    # Files
    files_sorted = sort_files_by_start_time(NC_DIR_SOLAR, YEARS, HALVES)
    if not files_sorted:
        raise FileNotFoundError("No ERA5-Land SSRD files found for processing.")
    print(f" ✓ Found {len(files_sorted)} NetCDF files")

    # Load accumulated SSRD for selected cells (J/m^2)
    print("Step 1: Loading hourly accumulated SSRD (J/m^2) for selected cells...")
    lon_len = len(lons)
    sel_flat_idx = np.array([j*lon_len + i for (j,i) in selected_indices], dtype=np.int64)
    all_times, all_data = [], []
    for path in tqdm(files_sorted, desc="Loading files"):
        with netCDF4.Dataset(path, 'r') as ds:
            tvar = ds.variables.get('valid_time', ds.variables.get('time'))
            if tvar is None: 
                continue
            times = robust_num2date(tvar)  # UTC tz-aware
            if 'ssrd' not in ds.variables:
                raise RuntimeError(f"'ssrd' not found in {path}")
            ssrd_J = ds.variables['ssrd'][:]  # [time, lat, lon], accumulated J/m^2
            for i in range(len(times)):
                flat = ssrd_J[i, :, :].ravel()
                sel  = flat[sel_flat_idx]
                all_times.append(times[i])
                all_data.append(sel)
    if not all_data:
        raise RuntimeError("No SSRD data loaded.")
    accum = np.vstack(all_data)  # [n_hours, n_cells]
    df_accum = pd.DataFrame(accum, index=all_times)  # UTC hourly
    df_accum = df_accum.sort_index()
    print(f" ✓ Loaded {len(df_accum)} hourly records for {accum.shape[1]} cells.")

    # Step 2: De-accumulate to hourly energy (J/m^2)
    print("Step 2: De-accumulating to hourly energy (J/m^2)...")
    df_inc = df_accum.copy()
    hours = df_inc.index.hour
    # First row rule
    if hours[0] != 1:
        df_inc.iloc[0] = np.nan
    for i in range(1, len(df_inc)):
        if hours[i] == 1:
            # First hour of UTC day: keep as-is (energy in 00-01 UTC)
            pass
        else:
            df_inc.iloc[i] = df_accum.iloc[i] - df_accum.iloc[i-1]
    # Numerical hygiene: zero-out tiny negatives, drop NaNs, keep non-negative rows
    df_inc[(df_inc < 0) & (df_inc.abs() < 1e-3)] = 0.0
    df_inc = df_inc.dropna()
    df_inc = df_inc[(df_inc >= 0).all(axis=1)]
    print(f" ✓ De-accumulated to {len(df_inc)} hourly increments.")

    # Step 3: Shift to local time (UTC+06:00) and resample to daily energy totals
    print("Step 3: Shifting timestamps by +06:00 and summing daily energy...")
    df_local = df_inc.copy()
    df_local.index = (df_local.index + pd.Timedelta(minutes=LOCAL_TZ_OFFSET_MIN)).tz_localize(None)
    df_daily_grid_J = df_local.resample('D').sum()     # J/m^2 per local day per grid
    df_daily_grid_MJ = df_daily_grid_J / 1e6          # MJ/m^2/day
    df_daily_grid_MJ.index.name = "Date_local_UTCp6"
    df_daily_grid_MJ.columns = selected_keys

    # QA: mass conservation (sum of hourly J vs sum of daily J after shift)
    total_hourly_J = df_inc.sum().sum()
    total_daily_J  = df_daily_grid_J.sum().sum()
    diff_J = abs(total_hourly_J - total_daily_J)
    print("\n=== QA: Mass Conservation ===")
    print(f" Total hourly (UTC): {total_hourly_J:,.2f} J/m^2 (summed over selected cells)")
    print(f" Total daily (UTC+06): {total_daily_J:,.2f} J/m^2 (summed over selected cells)")
    print(f" Difference: {diff_J:,.6f} J/m^2 ({diff_J/total_hourly_J*100:.6f}%)")
    print(" ============================\n")

    # Step 4: Basin mean (area-weighted energy, extensive)
    print("Step 4: Computing basin mean (area-weighted)...")
    w = pd.Series(weights_area).reindex(df_daily_grid_MJ.columns).fillna(0.0)  # km²
    denom = w.sum()
    if denom <= 0:
        raise RuntimeError("Area-weight denominator is zero after alignment.")
    basin_daily_MJ = (df_daily_grid_MJ * w).sum(axis=1) / denom
    basin_daily_MJ.name = "BasinMean_MJm2_day"

    # Step 5: Monthly mean of daily totals & climatology
    print("Step 5: Monthly mean of daily totals & climatology...")
    monthly_mean_daily = basin_daily_MJ.resample("MS").mean()
    monthly_df = monthly_mean_daily.to_frame(name="MonthlyMeanDailySolar_MJm2_day")
    monthly_df["Year"]  = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_wide = monthly_df.pivot(index="Year", columns="Month", values="MonthlyMeanDailySolar_MJm2_day").sort_index()
    monthly_wide.rename(columns=month_names, inplace=True)
    monthly_wide = monthly_wide.reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    climo_mean = monthly_df.groupby("Month")["MonthlyMeanDailySolar_MJm2_day"].mean().rename(index=month_names)

    # Save
    print("Saving outputs...")
    df_daily_grid_MJ.to_csv(OUT_DAILY_GRID_CSV, index_label="Date_local_UTCp6")
    basin_daily_MJ.to_frame().to_csv(OUT_DAILY_BASIN_CSV, index_label="Date_local_UTCp6")
    monthly_wide.to_csv(OUT_MONTHLY_WIDE_CSV, index_label="year")
    climo_mean.to_csv(OUT_CLIMO_CSV, index_label="Month", header=["ClimatologyMonthlyMeanDailySolar_MJm2_day"])
    print(f" ✓ Saved:\n - Per-grid daily: {OUT_DAILY_GRID_CSV}\n - Basin daily: {OUT_DAILY_BASIN_CSV}\n - Monthly mean of daily: {OUT_MONTHLY_WIDE_CSV}\n - Monthly climatology: {OUT_CLIMO_CSV}")

    # Plots
    print("Creating plots...")
    plt.style.use("seaborn-v0_8")
    # Daily basin series
    fig1, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(basin_daily_MJ.index, basin_daily_MJ.values, color="tab:orange", lw=0.9)
    ax1.set_title("Daily Solar Radiation – Basin Mean (MJ/m²/day) [UTC+06]")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("MJ/m²/day")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(FIG_DAILY_SERIES)
    # Climatological monthly cycle
    fig2, ax2 = plt.subplots(figsize=(9,4.5))
    ax2.plot(climo_mean.index, climo_mean.values, marker="o", color="tab:purple")
    ax2.set_title("Climatological Monthly Mean Daily Solar (MJ/m²/day) – Basin Mean [UTC+06]")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("MJ/m²/day")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_CLIMO_CYCLE)
    plt.show()

    print("="*78)
    print("PROCESSING COMPLETE (SSRD -> daily MJ/m²/day, UTC+06:00)")
    print("="*78)

if __name__ == "__main__":
    main()
