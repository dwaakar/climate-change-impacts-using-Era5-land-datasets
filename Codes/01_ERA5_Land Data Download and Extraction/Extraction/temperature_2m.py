import netCDF4
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from datetime import datetime as dt
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASIN_SHP = r"C:/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/Shp file/Tarakeshwor.shp"
NC_DIR_TEMP = r"C:/Users/Diwakar Adhikari/Documents/ERA5_Land/temperature"
OUT_DIR = os.path.join(NC_DIR_TEMP, "Boko Thesis")
os.makedirs(OUT_DIR, exist_ok=True)

# Outputs (keep your naming with NST labels)
OUT_DAILY_XLSX = rf"{OUT_DIR}/daily_temperature_stats_1984_2025_local.xlsx"  # 3 sheets: Avg/Max/Min
OUT_MONTHLY_CSV = rf"{OUT_DIR}/monthly_avg_temperature_by_year_1984_2025_local.csv"
OUT_CLIMO_CSV = rf"{OUT_DIR}/monthly_avg_temperature_climatology_1984_2025_local.csv"

YEARS = [str(y) for y in range(1984, 2025)]
HALVES = ["H1t", "H2t"]  # -> files like 1985H1t.nc, 1985H2t.nc

# Weighting & overlap options
USE_FRACTIONAL_OVERLAP = True         # <<< True = area-based fractional overlap of grid cells with basin
MULTIPLY_BY_COSLAT = False            # If True, final weight = overlap_fraction * cos(lat)
USE_LAT_WEIGHTING_WHEN_NO_FRAC = False  # Only used if USE_FRACTIONAL_OVERLAP=False; fallback mode
EQUAL_AREA_CRS = "EPSG:6933"          # NSIDC Equal-Area for reliable area computations
PRINT_GRID_COUNT = True               # Print how many grid cells overlap
PRINT_UTM_AREA = True                 # For extra area check (optional)

# Local time configuration (Nepal Standard Time is UTC+05:45)
LOCAL_TZ_OFFSET_MIN = 5*60 + 45  # +05:45
DATE_INDEX_NAME = "Date_NST"     # rename daily index to make it explicit

# ------------------------------------------------------------
# 1) Load Basin
# ------------------------------------------------------------
basin_wgs84 = gpd.read_file(BASIN_SHP).to_crs("EPSG:4326")
basin_union_wgs84 = basin_wgs84.unary_union
basin_ea = basin_wgs84.to_crs(EQUAL_AREA_CRS)
basin_union_ea = basin_ea.unary_union

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
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

def compute_fractional_overlap_weights(lats, lons):
    """
    Build cell polygons from center coordinates, project to equal-area CRS,
    intersect with basin polygon, and compute fractional overlap weights.

    Returns:
        selected_cells: list of (ilat, ilon)
        weights_per_grid: dict {grid_name: weight}
        total_overlap_area_km2: float
        basin_area_km2: float
    """
    lat_edges = edges_from_centers(lats)
    lon_edges = edges_from_centers(lons)

    selected_cells = []
    weights_per_grid = {}

    total_overlap_area_m2 = 0.0

    # Precompute basin area in equal-area CRS
    basin_area_km2 = float(basin_ea.area.sum() / 1e6)

    # Loop over grid to form polygons → project → intersect
    for ilat in range(len(lats)):
        lat_s = float(lat_edges[ilat])
        lat_n = float(lat_edges[ilat+1])
        for ilon in range(len(lons)):
            lon_w = float(lon_edges[ilon])
            lon_e = float(lon_edges[ilon+1])

            # Cell polygon in WGS84
            poly_wgs84 = Polygon([(lon_w, lat_s), (lon_e, lat_s), (lon_e, lat_n), (lon_w, lat_n)])

            # Quick reject: if cell doesn't intersect basin bounding box, skip
            if not poly_wgs84.intersects(basin_union_wgs84):
                continue

            # Project cell polygon to equal-area
            gdf_cell = gpd.GeoDataFrame({"geometry": [poly_wgs84]}, crs="EPSG:4326").to_crs(EQUAL_AREA_CRS)
            poly_ea = gdf_cell.iloc[0].geometry
            if not poly_ea.is_valid or poly_ea.is_empty:
                continue

            # Intersect with basin in equal-area CRS
            if not poly_ea.intersects(basin_union_ea):
                continue

            inter = poly_ea.intersection(basin_union_ea)
            if inter.is_empty:
                continue

            overlap_area = float(inter.area)            # m^2
            full_area = float(poly_ea.area)            # m^2
            if full_area <= 0:
                continue
            frac = overlap_area / full_area

            if frac > 0:
                selected_cells.append((ilat, ilon))
                gname = grid_name_from_indices(lats, lons, ilat, ilon)
                weight = frac
                if MULTIPLY_BY_COSLAT:
                    weight *= np.cos(np.deg2rad(float(lats[ilat])))
                weights_per_grid[gname] = weight
                total_overlap_area_m2 += overlap_area

    total_overlap_area_km2 = total_overlap_area_m2 / 1e6
    return selected_cells, weights_per_grid, total_overlap_area_km2, basin_area_km2

def compute_centroid_mask_and_weights(lats, lons):
    """
    Fallback: centroid-in-basin mask with optional cos(lat) weights.
    """
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = [Point(lon, lat) for lon, lat in zip(lon_grid.ravel(), lat_grid.ravel())]
    inside = [basin_union_wgs84.covers(pt) for pt in points]  # includes boundary
    indices = np.where(inside)[0]
    grid_coords = [(i // len(lons), i % len(lons)) for i in indices]

    weights_per_grid = {}
    for (ilat, ilon) in grid_coords:
        gname = grid_name_from_indices(lats, lons, ilat, ilon)
        if USE_LAT_WEIGHTING_WHEN_NO_FRAC:
            weights_per_grid[gname] = float(np.cos(np.deg2rad(float(lats[ilat]))))
        else:
            weights_per_grid[gname] = 1.0

    return grid_coords, weights_per_grid

def basin_mean_weighted(df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Row-wise weighted mean that ignores NaNs.
    - Aligns weights to available columns each day.
    - Denominator is the sum of weights for non-NaN entries on that day.
    """
    if df.shape[1] == 0:
        return pd.Series(index=df.index, dtype=float)

    w = pd.Series(weights, dtype=float)
    # align to df columns; missing columns -> 0 weight
    w = w.reindex(df.columns).fillna(0.0)

    # Numerator: value * weight; Denominator: weight where value is not NaN
    num = df.mul(w, axis=1).sum(axis=1, skipna=True)
    den = (~df.isna()).mul(w, axis=1).sum(axis=1)
    return num / den

# ------------------------------------------------------------
# 2) Storage for daily stats (local-time days)
# ------------------------------------------------------------
acc_sum = {}   # {"YYYY/MM/DD": {grid_name: sum_temp_c}}
acc_count = {} # {"YYYY/MM/DD": {grid_name: count_hours}}
acc_max = {}   # {"YYYY/MM/DD": {grid_name: max_temp_c}}
acc_min = {}   # {"YYYY/MM/DD": {grid_name: min_temp_c}}

# We'll fill/update this once per grid system (cached by simple key)
overlap_cache = {}
area_printed_for_key = set()

# ------------------------------------------------------------
# 3) Process ERA5/ERA5-Land Temperature Files
# ------------------------------------------------------------
for year in tqdm(YEARS, desc="Processing Years"):
    for half in HALVES:
        nc_files = glob.glob(f"{NC_DIR_TEMP}/{year}{half}.nc")
        print(f"\nYear {year}, Half {half}: Found {len(nc_files)} file(s)")
        for file in nc_files:
            print(f"Processing {file}...")
            ds = netCDF4.Dataset(file, 'r')

            # Variables (robust 'valid_time'/'time')
            lats = ds.variables['latitude'][:]
            lons = ds.variables['longitude'][:]

            # Build a cache key for this grid (assuming regular grid/order consistent)
            cache_key = (len(lats), float(lats[0]), float(lats[-1]), len(lons), float(lons[0]), float(lons[-1]))

            time_var = ds.variables.get('valid_time', ds.variables.get('time'))
            if time_var is None:
                ds.close()
                raise RuntimeError("No 'valid_time' or 'time' variable found in dataset.")

            # Convert to cftime datetimes
            times_cf = netCDF4.num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, 'calendar', 'standard'),
                only_use_cftime_datetimes=True
            )

            # Temperature variable (Kelvin)
            if 't2m' in ds.variables:
                t2m = ds.variables['t2m'][:]  # [time, lat, lon]
            else:
                for cand in ['T2', '2t', 't']:
                    if cand in ds.variables:
                        t2m = ds.variables[cand][:]
                        print(f"Using fallback temperature variable '{cand}'")
                        break
                else:
                    ds.close()
                    raise RuntimeError("No temperature variable 't2m' (or fallback) found in dataset.")

            # ---- Prepare overlap-based selection & weights ----
            if USE_FRACTIONAL_OVERLAP:
                if cache_key in overlap_cache:
                    grid_coords, weights_per_grid, total_overlap_area_km2, basin_area_km2 = overlap_cache[cache_key]
                else:
                    grid_coords, weights_per_grid, total_overlap_area_km2, basin_area_km2 = compute_fractional_overlap_weights(lats, lons)
                    overlap_cache[cache_key] = (grid_coords, weights_per_grid, total_overlap_area_km2, basin_area_km2)

                if cache_key not in area_printed_for_key:
                    if PRINT_GRID_COUNT:
                        print(f"[Overlap] Selected {len(grid_coords)} grid cells out of {len(lats)*len(lons)} total.")
                    print(f"[Area-weighted] Effective overlapped area: {total_overlap_area_km2:,.2f} km²")
                    print(f"[Polygon] Basin area ({EQUAL_AREA_CRS}): {basin_area_km2:,.2f} km²")
                    if PRINT_UTM_AREA:
                        try:
                            basin_utm = basin_wgs84.to_crs("EPSG:32645")  # UTM Zone 45N (optional)
                            basin_utm_km2 = float(basin_utm.area.sum() / 1e6)
                            print(f"[Polygon] Basin area (EPSG:32645): {basin_utm_km2:,.2f} km²")
                        except Exception as e:
                            print(f"UTM area computation skipped (projection error): {e}")
                    area_printed_for_key.add(cache_key)

            else:
                grid_coords, weights_per_grid = compute_centroid_mask_and_weights(lats, lons)
                if PRINT_GRID_COUNT:
                    print(f"[Centroid mask] Selected {len(grid_coords)} grid cells out of {len(lats)*len(lons)} total.")

            # ---- HOURLY -> DAILY **LOCAL TIME (NST)** ----
            for i in range(len(times_cf)):
                tcf = times_cf[i]
                # Naive pandas UTC timestamp at the hour mark
                ts_utc = pd.Timestamp(int(tcf.year), int(tcf.month), int(tcf.day), int(getattr(tcf, "hour", 0)))
                # >>> LOCAL TIME SHIFT <<<
                ts_local = (ts_utc.tz_localize("UTC") + pd.Timedelta(minutes=LOCAL_TZ_OFFSET_MIN)).tz_localize(None)
                day_key = ts_local.strftime("%Y/%m/%d")  # Local day key (NST)

                # Initialize nested dicts
                if day_key not in acc_sum:
                    acc_sum[day_key] = {}
                    acc_count[day_key] = {}
                    acc_max[day_key] = {}
                    acc_min[day_key] = {}

                # Update stats for selected grid cells
                for (ilat, ilon) in grid_coords:
                    gname = grid_name_from_indices(lats, lons, ilat, ilon)
                    temp_c = float(t2m[i, ilat, ilon]) - 273.15  # Kelvin -> °C

                    # sum & count (for average)
                    acc_sum[day_key][gname] = acc_sum[day_key].get(gname, 0.0) + temp_c
                    acc_count[day_key][gname] = acc_count[day_key].get(gname, 0) + 1

                    # max
                    if gname not in acc_max[day_key]:
                        acc_max[day_key][gname] = temp_c
                    else:
                        acc_max[day_key][gname] = max(acc_max[day_key][gname], temp_c)

                    # min
                    if gname not in acc_min[day_key]:
                        acc_min[day_key][gname] = temp_c
                    else:
                        acc_min[day_key][gname] = min(acc_min[day_key][gname], temp_c)

            ds.close()

print("Finished reading files and building **local-time** (NST) daily temperature aggregates.")

# ------------------------------------------------------------
# 4) Build Daily DataFrames (Avg / Max / Min)
# ------------------------------------------------------------
daily_avg = {}
for day, grid_sums in acc_sum.items():
    counts = acc_count[day]
    # average = sum / count per grid (works even if a day has <24 hours available)
    daily_avg[day] = {g: (grid_sums[g] / counts[g]) for g in grid_sums.keys()}

daily_max = acc_max
daily_min = acc_min

# Create DataFrames
df_avg = pd.DataFrame.from_dict(daily_avg, orient='index').sort_index()
df_max = pd.DataFrame.from_dict(daily_max, orient='index').sort_index()
df_min = pd.DataFrame.from_dict(daily_min, orient='index').sort_index()
df_avg.index.name = df_max.index.name = df_min.index.name = DATE_INDEX_NAME  # >>> LOCAL TIME LABEL

# Collect the final weights (from the last/any cache entry; they should be identical for all files of same grid)
# If multiple keys exist (unlikely if all files share the same grid), merge them.
final_weights = {}
for (_k, (_coords, wts, _oa, _ba)) in overlap_cache.items():
    final_weights.update(wts)

# Weighted basin mean (row-wise robust)
df_avg["BasinMean_C"] = basin_mean_weighted(df_avg, final_weights)
df_max["BasinMean_C"] = basin_mean_weighted(df_max, final_weights)
df_min["BasinMean_C"] = basin_mean_weighted(df_min, final_weights)

# ------------------------------------------------------------
# 5) Save Daily to Excel (3 sheets, Local Time)
# ------------------------------------------------------------
with pd.ExcelWriter(OUT_DAILY_XLSX, engine="openpyxl") as writer:
    df_avg.to_excel(writer, sheet_name="Daily_Avg_C_NST")
    df_max.to_excel(writer, sheet_name="Daily_Max_C_NST")
    df_min.to_excel(writer, sheet_name="Daily_Min_C_NST")
print(f"Saved local-time daily Avg/Max/Min temperature to {OUT_DAILY_XLSX}")

# ------------------------------------------------------------
# 6) Monthly Average (Year × Month) & Climatology (from local-time days)
# ------------------------------------------------------------
df_avg_ts = df_avg.copy()
df_avg_ts.index = pd.to_datetime(df_avg_ts.index, format="%Y/%m/%d")  # already local-day dates

# Use basin-average **local** daily mean for monthly average
basin_daily_avg = df_avg_ts["BasinMean_C"]

# Monthly average temperature (mean of daily averages)
monthly_avg = basin_daily_avg.resample("MS").mean()  # 'MS' = Month Start (local calendar)
monthly_df = monthly_avg.to_frame(name="MonthlyAvg_C")
monthly_df["Year"] = monthly_df.index.year
monthly_df["Month"] = monthly_df.index.month

# Pivot to Year × Month (Jan..Dec)
monthly_wide = monthly_df.pivot(index="Year", columns="Month", values="MonthlyAvg_C").sort_index()
month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
monthly_wide.rename(columns=month_names, inplace=True)
monthly_wide = monthly_wide.reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

# Save monthly by year (Avg only)
monthly_wide.to_csv(OUT_MONTHLY_CSV, index_label="year")
print(f"Saved local-time monthly average temperature (Year × Month) to {OUT_MONTHLY_CSV}")

# Monthly climatology (mean of monthly averages across years)
climo_mean = monthly_df.groupby("Month")["MonthlyAvg_C"].mean().rename(index=month_names)
climo_mean.to_csv(OUT_CLIMO_CSV, index_label="Month", header=["ClimatologyMonthlyAvg_C"])
print(f"Saved local-time monthly average temperature climatology to {OUT_CLIMO_CSV}")

# ------------------------------------------------------------
# 7) Visualizations
# ------------------------------------------------------------
plt.style.use("seaborn-v0_8")

# --- Daily trends: Avg / Max / Min (Basin Mean, NST) ---
df_max_ts = pd.to_datetime(df_max.index, format="%Y/%m/%d")
df_min_ts = pd.to_datetime(df_min.index, format="%Y/%m/%d")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_avg_ts.index, df_avg_ts["BasinMean_C"], label="Daily Avg (°C)", color="tab:blue", alpha=0.8)
ax.plot(df_max_ts, df_max["BasinMean_C"], label="Daily Max (°C)", color="tab:red", alpha=0.6)
ax.plot(df_min_ts, df_min["BasinMean_C"], label="Daily Min (°C)", color="tab:green", alpha=0.6)
ax.set_title("Daily Temperature – Basin Mean (°C) [NST]")
ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# --- Year–Month heatmap of monthly average temperature (NST) ---
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(monthly_wide.values, aspect="auto", cmap="coolwarm")
ax.set_title("Monthly Average Temperature (°C) – Basin Mean [NST]")
ax.set_xlabel("Month")
ax.set_ylabel("Year")
ax.set_xticks(range(12))
ax.set_xticklabels(monthly_wide.columns)
ax.set_yticks(range(len(monthly_wide.index)))
ax.set_yticklabels(monthly_wide.index)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.set_ylabel("°C", rotation=90)
plt.tight_layout()
plt.show()

# --- Climatological seasonal cycle (mean monthly avg temperature, NST) ---
plt.figure(figsize=(10, 5))
plt.plot(climo_mean.index, climo_mean.values, marker="o", color="tab:purple")
plt.title("Climatological Monthly Average Temperature (°C) – Basin Mean [NST]")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ Completed LOCAL-TIME (NST) daily & monthly temperature extraction with FRACTIONAL OVERLAP weighting and visualizations.")