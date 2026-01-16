
# -*- coding: utf-8 -*-
"""
windspeed_u10_v10.py
----------------------------------------------------------------
Convert ERA5-Land hourly 10 m wind components from separate NetCDF streams:
  - u10 (eastward component, m/s)
  - v10 (northward component, m/s)
into local-time (NST = UTC+05:45) **daily mean wind speed (m/s)**.

Keeps the same pipeline as your RH script:
 • polygon–polygon FRACTIONAL OVERLAP (equal-area CRS) for basin weighting
 • robust time alignment between u10 and v10 halves per year
 • local-time grouping before daily averaging
 • basin-weighted daily mean series + monthly averages + monthly climatology
 • diagnostics and plots

Author: Diwakar-friendly, modular version
Date: 2025-12-19 (NST)
"""
import os
import glob
import netCDF4
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# 0) User Configuration
# -------------------------------
# Basin shapefile (WGS84 polygons)
BASIN_SHP = r"C:/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/Shp file/MRB.shp"

# Separate NetCDF directories for u10 and v10 (hourly ERA5-Land files)
NC_DIR_U10 = r"C:/Users/Diwakar Adhikari/Downloads/SWAT+/ERA5_Land/Wind_Speed/wind_u10"
NC_DIR_V10 = r"C:/Users/Diwakar Adhikari/Downloads/SWAT+/ERA5_Land/Wind_Speed/wind_v10"

# Output directory
OUT_DIR = r"C:/Users/Diwakar Adhikari/Downloads/SWAT+/ERA5_Land/Wind_Speed"
os.makedirs(OUT_DIR, exist_ok=True)

# Years & halves (adjust as needed)
YEARS = [str(y) for y in range(1985, 2025)]
HALVES_u10 = ["H1_u10", "H2_u10"]
HALVES_v10 = ["H1_v10", "H2_v10"]

# Local time offset: Nepal Standard Time (no DST)
LOCAL_TZ_OFFSET_MIN = 5*60 + 45  # +05:45
DATE_INDEX_NAME = "Date_NST"

# Basin averaging options
USE_FRACTIONAL_OVERLAP = True  # area-based fractional overlap of grid cells with basin
MULTIPLY_BY_COSLAT = False     # Optional: multiply weights by cos(lat)
EQUAL_AREA_CRS = "EPSG:6933"   # NSIDC Equal-Area (meters) for reliable areas

# Diagnostics
PRINT_GRID_COUNT = True
PRINT_UTM_AREA = True  # optional area check in EPSG:32645 (UTM Zone 45N)

# QC thresholds
MIN_HOURS_PER_DAY = 18       # flag days with fewer hours
REQUIRE_FULL_LOCAL_DAY = False  # set True to drop days with <24 hours

# Output files
OUT_GRID_DAILY_CSV   = os.path.join(OUT_DIR, "era5land_daily_wind_grid_NST_mps.csv")
OUT_BASIN_DAILY_CSV  = os.path.join(OUT_DIR, "era5land_daily_wind_basin_NST_mps.csv")
OUT_MONTHLY_WIDE_CSV = os.path.join(OUT_DIR, "era5land_monthly_mean_daily_wind_basin_NST_mps.csv")
OUT_CLIMO_CSV        = os.path.join(OUT_DIR, "era5land_monthly_mean_daily_wind_climatology_basin_NST_mps.csv")

# Figures
FIG_DAILY_SERIES   = os.path.join(OUT_DIR, "fig_basin_daily_wind_series_NST.png")
FIG_MONTHLY_CLIMO  = os.path.join(OUT_DIR, "fig_basin_monthly_wind_climatology_NST.png")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200})

# -------------------------------
# 1) Helpers
# -------------------------------
def robust_num2date(time_var):
    """Parse CF time variable -> tz-naive pandas DatetimeIndex at the exact hour (UTC concept)."""
    vals = time_var[:]
    units = time_var.units
    cal = getattr(time_var, 'calendar', 'standard')
    cftimes = netCDF4.num2date(vals, units=units, calendar=cal, only_use_cftime_datetimes=True)
    pdtimes = pd.to_datetime([pd.Timestamp(t.year, t.month, t.day, getattr(t, "hour", 0)) for t in cftimes])
    return pdtimes  # naive; interpret as UTC later

def load_basin_geometry(shp_path):
    """Load basin shapefile and return GeoDataFrame in WGS84 & unary union geometry."""
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    return gdf, gdf.unary_union

def edges_from_centers(arr: np.ndarray) -> np.ndarray:
    """Compute cell edge coordinates from center coordinates for a 1D monotonic array."""
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        raise ValueError("Need at least two center points to estimate edges.")
    d = np.diff(arr)
    edges_mid = arr[:-1] + d/2.0
    first_edge = arr[0] - d[0]/2.0
    last_edge = arr[-1] + d[-1]/2.0
    return np.concatenate([[first_edge], edges_mid, [last_edge]])

def grid_name_from_indices(lats, lons, ilat, ilon) -> str:
    """Stable grid name consistent across the script."""
    return f"lat{float(lats[ilat]):.3f}_lon{float(lons[ilon]):.3f}"

def compute_fractional_overlap_weights(lats, lons, basin_union_wgs84, basin_union_ea):
    """
    Build cell polygons from center coordinates, project to equal-area CRS,
    intersect with basin polygon, and compute fractional overlap weights.
    Returns:
      grid_coords: list of (ilat, ilon) with non-zero overlap
      grid_keys: list of grid names aligned to grid_coords
      weights: np.array normalized to sum=1 (area-based, optional cos(lat) multiplier)
      total_overlap_area_km2: float
      basin_area_km2: float
    """
    lat_edges = edges_from_centers(lats)
    lon_edges = edges_from_centers(lons)

    selected_cells, grid_keys, weights_raw = [], [], []
    total_overlap_area_m2 = 0.0
    basin_area_km2 = 0.0
    try:
        basin_area_km2 = float(basin_union_ea.area / 1e6)
    except Exception:
        pass

    for ilat in range(len(lats)):
        lat_s = float(lat_edges[ilat])
        lat_n = float(lat_edges[ilat+1])
        for ilon in range(len(lons)):
            lon_w = float(lon_edges[ilon])
            lon_e = float(lon_edges[ilon+1])
            # WGS84 cell polygon
            poly_wgs84 = Polygon([(lon_w, lat_s), (lon_e, lat_s), (lon_e, lat_n), (lon_w, lat_n)])
            if not poly_wgs84.intersects(basin_union_wgs84):
                continue
            # Equal-area projection
            gdf_cell = gpd.GeoDataFrame({"geometry": [poly_wgs84]}, crs="EPSG:4326").to_crs(EQUAL_AREA_CRS)
            poly_ea = gdf_cell.iloc[0].geometry
            if poly_ea.is_empty or not poly_ea.is_valid:
                continue
            # Intersection
            if not poly_ea.intersects(basin_union_ea):
                continue
            inter = poly_ea.intersection(basin_union_ea)
            if inter.is_empty:
                continue
            overlap_area = float(inter.area)   # m^2
            full_area = float(poly_ea.area)    # m^2
            if full_area <= 0:
                continue
            frac = overlap_area / full_area
            if frac <= 0:
                continue

            selected_cells.append((ilat, ilon))
            gname = grid_name_from_indices(lats, lons, ilat, ilon)
            weight = frac
            if MULTIPLY_BY_COSLAT:
                weight *= np.cos(np.deg2rad(float(lats[ilat])))
            grid_keys.append(gname)
            weights_raw.append(weight)
            total_overlap_area_m2 += overlap_area

    if len(selected_cells) == 0:
        raise RuntimeError("Fractional overlap produced zero selected cells. Check basin extent and grid.")

    weights_raw = np.array(weights_raw, dtype=float)
    weights = weights_raw / weights_raw.sum()
    total_overlap_area_km2 = total_overlap_area_m2 / 1e6
    return selected_cells, grid_keys, weights, total_overlap_area_km2, basin_area_km2

def compute_centroid_mask_and_weights(lats, lons, basin_union_wgs84):
    """Fallback: centroid-in-basin mask with equal weights."""
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    pts = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    inside = [basin_union_wgs84.covers(pt) for pt in pts]  # includes boundary
    idxs = np.where(inside)[0]
    grid_coords = [(i // len(lons), i % len(lons)) for i in idxs]
    if len(grid_coords) == 0:
        raise RuntimeError("Centroid mask returned zero grid cells. Check basin shapefile and coordinates.")
    grid_keys = [grid_name_from_indices(lats, lons, i, j) for (i, j) in grid_coords]
    weights = np.ones(len(grid_coords), dtype=float) / len(grid_coords)
    return grid_coords, grid_keys, weights

# -------------------------------
# 2) Main
# -------------------------------
def main():
    print("="*72)
    print("ERA5-Land u10 & v10 (separate files) -> Daily Wind Speed (m/s) [NST] – GRID + BASIN (fractional overlap)")
    print("="*72)

    # Basin geometry
    basin_gdf_wgs84, basin_union_wgs84 = load_basin_geometry(BASIN_SHP)
    basin_gdf_ea = basin_gdf_wgs84.to_crs(EQUAL_AREA_CRS)
    basin_union_ea = basin_gdf_ea.unary_union

    # Initialize grid/weights using FIRST AVAILABLE u10 file
    first_file = None
    for y in YEARS:
        for half in HALVES_u10:
            cand = glob.glob(os.path.join(NC_DIR_U10, f"{y}{half}.nc"))
            if cand:
                first_file = cand[0]
                break
        if first_file:
            break
    if first_file is None:
        raise FileNotFoundError("No u10 NetCDF files found to initialize grid.")
    with netCDF4.Dataset(first_file, 'r') as ds0:
        lats = ds0.variables['latitude'][:]
        lons = ds0.variables['longitude'][:]
        total_cells = len(lats) * len(lons)

        if USE_FRACTIONAL_OVERLAP:
            grid_coords, grid_keys, weights, total_overlap_area_km2, basin_area_km2 = \
                compute_fractional_overlap_weights(lats, lons, basin_union_wgs84, basin_union_ea)
            if PRINT_GRID_COUNT:
                print(f"[Overlap] Selected {len(grid_coords)} grid cells out of {total_cells} total.")
                print(f"[Area-weighted] Effective overlapped area: {total_overlap_area_km2:,.2f} km²")
                print(f"[Polygon] Basin area ({EQUAL_AREA_CRS}): {basin_area_km2:,.2f} km²")
            if PRINT_UTM_AREA:
                try:
                    basin_utm = basin_gdf_wgs84.to_crs("EPSG:32645")  # UTM Zone 45N
                    basin_utm_km2 = float(basin_utm.area.sum() / 1e6)
                    print(f"[Polygon] Basin area (EPSG:32645): {basin_utm_km2:,.2f} km²")
                except Exception as e:
                    print(f"UTM area computation skipped (projection error): {e}")
        else:
            grid_coords, grid_keys, weights = compute_centroid_mask_and_weights(lats, lons, basin_union_wgs84)
            if PRINT_GRID_COUNT:
                print(f"[Centroid mask] Selected {len(grid_coords)} grid cells out of {total_cells} total.")

    # Accumulators (local-day): sum of hourly speed & count of hours per grid
    wind_sum_daily = {}   # {"YYYY/MM/DD": np.array(n_grid,)}
    wind_count_daily = {} # {"YYYY/MM/DD": np.array(n_grid,)}

    # Flatten selection indices once
    lon_len = len(lons)
    sel_flat = np.array([i*lon_len + j for (i, j) in grid_coords], dtype=np.int64)

    print("\nStep: Scanning NetCDF files & accumulating daily wind speed (m/s) in local time (NST)...")
    for year in tqdm(YEARS, desc="Years"):
        for hu, hv in zip(HALVES_u10, HALVES_v10):
            fu = glob.glob(os.path.join(NC_DIR_U10, f"{year}{hu}.nc"))
            fv = glob.glob(os.path.join(NC_DIR_V10, f"{year}{hv}.nc"))
            if not fu or not fv:
                continue
            path_u = fu[0]
            path_v = fv[0]
            with netCDF4.Dataset(path_u, 'r') as dsU, netCDF4.Dataset(path_v, 'r') as dsV:
                # Time variables
                tvar_U = dsU.variables.get('valid_time', dsU.variables.get('time'))
                tvar_V = dsV.variables.get('valid_time', dsV.variables.get('time'))
                if tvar_U is None or tvar_V is None:
                    continue
                times_U = robust_num2date(tvar_U)
                times_V = robust_num2date(tvar_V)

                # Read variables (m/s)
                u_name = 'u10' if 'u10' in dsU.variables else list(dsU.variables.keys())[0]
                v_name = 'v10' if 'v10' in dsV.variables else list(dsV.variables.keys())[0]
                u10 = dsU.variables[u_name][:]  # [time, lat, lon], m/s
                v10 = dsV.variables[v_name][:]  # [time, lat, lon], m/s

                # Align by UTC timestamps (intersection)
                idx_U = pd.Index(times_U)
                idx_V = pd.Index(times_V)
                common = idx_U.intersection(idx_V)
                if common.empty:
                    continue

                # Map times to indices
                loc_U = idx_U.get_indexer(common)
                loc_V = idx_V.get_indexer(common)

                # Flatten to [time, lat*lon], then select basin grid cells
                u_flat = u10[loc_U].reshape(len(common), -1)
                v_flat = v10[loc_V].reshape(len(common), -1)
                u_sel  = u_flat[:, sel_flat]
                v_sel  = v_flat[:, sel_flat]

                # === THE ONLY CALCULATION THAT CHANGED ===
                # Hourly wind speed magnitude (m/s) in UTC
                speed_utc = np.sqrt(u_sel*u_sel + v_sel*v_sel)  # [n_time, n_grid]

                # Shift timestamps to local time (NST)
                idx_local = (common.tz_localize("UTC") + pd.Timedelta(minutes=LOCAL_TZ_OFFSET_MIN)).tz_localize(None)

                # Accumulate into local-day sums & counts
                day_keys = pd.to_datetime(idx_local.date)  # local day
                df_hour = pd.DataFrame(speed_utc, index=idx_local)  # columns 0..n_grid-1
                grouped = df_hour.groupby(day_keys)
                for dkey, g in grouped:
                    day_str = dkey.strftime("%Y/%m/%d")
                    gsum = np.array(g.sum(axis=0), dtype=float)
                    gcount = np.array((~g.isna()).sum(axis=0), dtype=float)
                    if day_str not in wind_sum_daily:
                        wind_sum_daily[day_str] = gsum
                        wind_count_daily[day_str] = gcount
                    else:
                        wind_sum_daily[day_str] += gsum
                        wind_count_daily[day_str] += gcount

    # Build daily grid wind speed (NST): mean = sum / count
    print("\nStep: Finalizing daily NST grid wind speed ...")
    all_days = sorted(wind_sum_daily.keys())
    daily_mean_list, daily_count_list = [], []
    for d in all_days:
        s = wind_sum_daily[d]
        c = wind_count_daily[d]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(c > 0, s / c, np.nan)  # daily mean m/s
        daily_mean_list.append(m)
        daily_count_list.append(c)

    daily_mean_arr = np.vstack(daily_mean_list)  # [days, grids]
    daily_count_arr = np.vstack(daily_count_list)

    # Optionally require full days (24 hours) across all selected grids
    if REQUIRE_FULL_LOCAL_DAY:
        full_mask = (daily_count_arr >= 24).all(axis=1)
        kept = np.count_nonzero(full_mask)
        dropped = len(all_days) - kept
        if dropped > 0:
            print(f"✓ Dropping {dropped} local days with incomplete hourly coverage; keeping {kept}.")
        daily_mean_arr = daily_mean_arr[full_mask]
        all_days = [d for (d, keep) in zip(all_days, full_mask) if keep]

    # DataFrame for grid daily wind speed
    df_daily_grid = pd.DataFrame(
        daily_mean_arr,
        index=pd.to_datetime(all_days, format="%Y/%m/%d"),
        columns=grid_keys
    )
    df_daily_grid.index.name = DATE_INDEX_NAME
    print(f"[Final] Total number of basin grids present in daily dataset: {df_daily_grid.shape[1]}")

    # Basin mean (weighted by fractional overlap, normalized)
    print("\nStep: Computing basin mean wind speed ...")
    w = pd.Series(weights, index=df_daily_grid.columns, dtype=float)
    num = (df_daily_grid * w).sum(axis=1, skipna=True)
    den = (~df_daily_grid.isna()).mul(w, axis=1).sum(axis=1)
    basin_daily = num / den
    basin_daily.name = "BasinMean_wind_mps"

    # QC flag: enough hours in majority of grids?
    qc_flags = []
    for d in all_days:
        vals = wind_count_daily[d]
        if len(vals) == 0:
            qc_flags.append(False)
        else:
            ok = (np.mean(np.array(vals) >= MIN_HOURS_PER_DAY) >= 0.5)
            qc_flags.append(ok)
    qc_series = pd.Series(qc_flags, index=df_daily_grid.index, name="QC_enough_hours")

    # -------------------------------
    # Monthly aggregates & climatology (from local daily means)
    # -------------------------------
    print("\nStep: Computing monthly aggregates & climatology ...")
    df_daily_grid["QC_enough_hours"] = qc_series
    basin_daily_df = basin_daily.to_frame()
    basin_daily_df["QC_enough_hours"] = qc_series

    monthly_mean = basin_daily.resample("MS").mean()  # monthly mean of daily wind (m/s)
    monthly_df = monthly_mean.to_frame(name="MonthlyMeanDailyWind_mps")
    monthly_df["Year"] = monthly_df.index.year
    monthly_df["Month"] = monthly_df.index.month

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly_wide = monthly_df.pivot(index="Year", columns="Month", values="MonthlyMeanDailyWind_mps").sort_index()
    monthly_wide.rename(columns=month_names, inplace=True)
    monthly_wide = monthly_wide.reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

    climo = monthly_df.groupby("Month")["MonthlyMeanDailyWind_mps"].mean().rename(index=month_names)
    climo.name = "ClimatologyMonthlyMeanDailyWind_mps"

    # -------------------------------
    # Save outputs
    # -------------------------------
    print("\nStep: Saving CSV outputs ...")
    df_daily_grid.drop(columns=["QC_enough_hours"]).to_csv(OUT_GRID_DAILY_CSV)                 # grid-daily wind (m/s)
    basin_daily_df.to_csv(OUT_BASIN_DAILY_CSV, index_label=DATE_INDEX_NAME)                    # basin-daily wind (m/s) + QC
    monthly_wide.to_csv(OUT_MONTHLY_WIDE_CSV, index_label="year")
    climo.to_csv(OUT_CLIMO_CSV, index_label="Month", header=["ClimatologyMonthlyMeanDailyWind_mps"])

    print(f"✓ Grid-daily wind (NST): {OUT_GRID_DAILY_CSV}")
    print(f"✓ Basin-daily wind (NST): {OUT_BASIN_DAILY_CSV}")
    print(f"✓ Monthly mean of daily wind (NST, Year×Month): {OUT_MONTHLY_WIDE_CSV}")
    print(f"✓ Monthly climatology of daily wind (NST): {OUT_CLIMO_CSV}")

    # -------------------------------
    # Plots: (a) Daily series; (b) Monthly climatology
    # -------------------------------
    print("\nStep: Creating plots ...")
    plt.style.use("seaborn-v0_8")

    # (a) Daily basin wind time series
    fig1, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(basin_daily.index, basin_daily.values, color="tab:blue", lw=0.8)
    ax1.set_title("Daily Wind Speed (m/s) – Basin Mean [ERA5-Land, NST]")
    ax1.set_xlabel("Date (NST)")
    ax1.set_ylabel("Wind speed (m/s)")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(FIG_DAILY_SERIES)

    # (b) Monthly climatological cycle
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.plot(climo.index, climo.values, marker="o", color="tab:green")
    ax2.set_title("Climatological Monthly Mean of Daily Wind (m/s) – Basin Mean [NST]")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Wind speed (m/s)")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIG_MONTHLY_CLIMO)

    plt.show()
    print(f"✓ Saved plots to: {OUT_DIR}")
    print("\n" + "="*72)
    print("PROCESSING COMPLETE!")
    print("="*72)

if __name__ == "__main__":
    main()
