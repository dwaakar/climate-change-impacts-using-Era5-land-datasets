# -*- coding: utf-8 -*-
"""
GCM Temperature (tas_day) → Area-true fractional overlap weighting
------------------------------------------------------------------

What this script does:
- Loads CMIP6 daily tas files for 'historical' and scenarios (ssp245, ssp585).
- Builds an area-true fractional mask for each model grid:
    fraction(i,j) = area(cell_ij ∩ basin) / area(cell_ij)
  computed in a **Lambert Azimuthal Equal-Area (LAEA)** projection centered on the basin.
- Uses these fractions as weights to compute basin-average **temperature (°C)** time series,
  which is appropriate because temperature is an **intensive** variable (values not scaled by area).
- Produces:
    * Annual mean tas (1985–2100) per model (CSV).
    * Monthly climatology (means over 1985–2014 and 2071–2100) per model (CSV).
- (Optional) Creates a QA plot of fractional overlap (0–1) per grid cell with basin outline.

Notes:
- No files are converted on disk; reprojections occur **in-memory** for the area math.
- If a model grid repeats across historical and scenario files, the mask/weights
  computed from the historical file are reused for the scenario file.

Requirements:
- netCDF4, numpy, pandas, geopandas, shapely, pyproj, matplotlib (optional for QA plot).
"""

import netCDF4
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import os
from tqdm import tqdm
from collections import defaultdict

from shapely.geometry import box
from shapely.ops import transform
import pyproj
import warnings
warnings.filterwarnings('ignore')

# ------------------------------- #
# Configuration
# ------------------------------- #
BASE_DIR = "C:/Users/Diwakar Adhikari/Downloads/GCMs/temperature"
SHAPEFILE_PATH = "C:/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/Shp file/MRB.shp"  # UPDATE THIS PATH

SCENARIOS = ["ssp245", "ssp585"]
SCENARIO_TAGS = ['_ssp245', '_ssp585', '_historical']

VAR_NAME = 'tas'  # near-surface air temperature (K)

# Periods (years inclusive)
PERIOD_1 = (1985, 2014)  # Historical baseline period
PERIOD_2 = (2071, 2100)  # Future projection period

# Optional QA plot
MAKE_QA_PLOT = True
QA_PLOT_PATH = os.path.join(BASE_DIR, "fractional_overlap_weights_tas.png")

# ------------------------------- #
# Load shapefile & prepare union
# ------------------------------- #
print("Loading shapefile...")
study_area = gpd.read_file(SHAPEFILE_PATH)
if study_area.crs is None:
    study_area = study_area.set_crs(epsg=4326)
else:
    study_area = study_area.to_crs(epsg=4326)  # Ensure WGS84
print(f"Study area loaded: {study_area.shape[0]} feature(s)")
study_area_union = study_area.unary_union

# ------------------------------- #
# Helpers: grid bounds & weighting
# ------------------------------- #
def infer_bounds(coords: np.ndarray) -> np.ndarray:
    """
    Infer cell edges (bounds) from center coordinates.
    Handles ascending or descending arrays.
    Returns edges array of length len(coords)+1.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n == 1:
        # Fallback width if single point (rare in daily CMIP grids)
        step = 0.5
        return np.array([coords[0] - step/2, coords[0] + step/2], dtype=float)
    edges = np.empty(n + 1, dtype=float)
    # inner edges
    edges[1:-1] = (coords[:-1] + coords[1:]) / 2.0
    # end edges by extrapolation of the first/last step
    first_step = coords[1] - coords[0]
    last_step  = coords[-1] - coords[-2]
    edges[0]   = coords[0] - first_step / 2.0
    edges[-1]  = coords[-1] + last_step  / 2.0
    return edges

def create_area_true_fraction_mask(lats, lons, shapefile_union):
    """
    Create an area-true fractional overlap mask for grid cells that overlap with the study area.

    Returns:
        mask     : 2D boolean array (True if a cell intersects study area)
        fractions: 2D float array of overlap fractions (0..1), computed in an equal-area projection
        lat_bounds, lon_bounds: edges arrays used (len = n+1)
        laea_proj: pyproj Transformer used (for possible reuse/QA)
    """
    # Calculate grid cell boundaries (edges)
    lat_bounds = infer_bounds(lats)
    lon_bounds = infer_bounds(lons)

    # Equal-area projection centered on the basin (LAEA)
    # This preserves area locally and avoids zone issues
    wgs84 = pyproj.CRS('EPSG:4326')
    centroid = shapefile_union.centroid
    laea = pyproj.CRS(
        f'+proj=laea +lat_0={centroid.y} +lon_0={centroid.x} +datum=WGS84 +units=m'
    )
    to_laea = pyproj.Transformer.from_crs(wgs84, laea, always_xy=True).transform

    # Project the basin union to LAEA for intersections
    study_area_laea = transform(to_laea, shapefile_union)

    # Initialize arrays
    nlat, nlon = len(lats), len(lons)
    mask = np.zeros((nlat, nlon), dtype=bool)
    fractions = np.zeros((nlat, nlon), dtype=float)

    # Loop cells
    for i in range(nlat):
        # sort lat edges for robustness
        lat_min = min(lat_bounds[i],   lat_bounds[i+1])
        lat_max = max(lat_bounds[i],   lat_bounds[i+1])
        for j in range(nlon):
            lon_min = min(lon_bounds[j], lon_bounds[j+1])
            lon_max = max(lon_bounds[j], lon_bounds[j+1])

            # Create lon/lat rectangle (cell)
            cell_box_ll = box(lon_min, lat_min, lon_max, lat_max)

            if not cell_box_ll.intersects(shapefile_union):
                # no overlap → leave mask False and fraction 0
                continue

            # Project cell to LAEA
            cell_box_laea = transform(to_laea, cell_box_ll)

            # Intersection area and fraction
            inter = cell_box_laea.intersection(study_area_laea)
            cell_area = cell_box_laea.area
            inter_area = inter.area if not inter.is_empty else 0.0

            mask[i, j] = inter_area > 0.0
            fractions[i, j] = (inter_area / cell_area) if cell_area > 0.0 else 0.0

    return mask, fractions, lat_bounds, lon_bounds, to_laea

def extract_daily_temperature(file_path, mask=None, fractions=None,
                              shapefile_union=None, var_name='tas'):
    """
    Extract daily temperature and compute basin-average (°C) using **fractional weights**.

    If mask/fractions are None, they are computed from the file's lat/lon grid and shapefile union.
    Returns:
        df_daily   : DataFrame with index=Date (UTC or calendar) and column 'Temperature' (°C)
        mask       : 2D boolean mask (for reuse)
        fractions  : 2D float (0..1) fractional weights (for reuse)
        lat        : 1D array of lat centers
        lon        : 1D array of lon centers
        lat_bounds : 1D edges
        lon_bounds : 1D edges
    """
    try:
        ds = netCDF4.Dataset(file_path, 'r')
        # Time var
        time_var = ds.variables.get('time')
        if time_var is None or var_name not in ds.variables:
            ds.close()
            return None, None, None, None, None, None, None

        # Read time via cftime → pandas
        time = netCDF4.num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, 'calendar', 'standard'),
            only_use_cftime_datetimes=True
        )
        try:
            # Mixed parsing handles various formats
            time = pd.to_datetime([str(t) for t in time], format='mixed', errors='coerce')
        except Exception as e:
            print(f"Time conversion failed for {file_path}: {e}")
            ds.close()
            return None, None, None, None, None, None, None

        valid_mask = ~pd.isnull(time)
        if not np.any(valid_mask):
            print(f"No valid dates found in {file_path}")
            ds.close()
            return None, None, None, None, None, None, None

        # Coordinates
        lat = ds.variables.get('lat') or ds.variables.get('latitude')
        lon = ds.variables.get('lon') or ds.variables.get('longitude')
        if lat is None or lon is None:
            print(f"Missing lat/lon in {file_path}")
            ds.close()
            return None, None, None, None, None, None, None

        lat = np.asarray(lat[:], dtype=float)
        lon = np.asarray(lon[:], dtype=float)

        # Build mask/fractions if not provided
        if mask is None or fractions is None:
            if shapefile_union is None:
                raise RuntimeError("shapefile_union must be provided when mask/fractions are None.")
            mask, fractions, lat_bounds, lon_bounds, _ = create_area_true_fraction_mask(lat, lon, shapefile_union)
        else:
            lat_bounds = infer_bounds(lat)
            lon_bounds = infer_bounds(lon)

        # Read tas variable [time, lat, lon] and convert K → °C
        data_var = ds.variables[var_name][:]  # assume [time, lat, lon]
        tas_celsius = data_var[valid_mask] - 273.15

        # Weighted mean per time step using fractions only over overlapping cells
        # (Temperature is intensive → fraction weights appropriate)
        if not np.any(mask):
            print(f"Warning: No grid cells overlap with study area for {file_path}")
            ds.close()
            return None, None, None, None, None, None, None

        frac_vec = fractions[mask]
        denom = np.sum(frac_vec)
        if denom <= 0:
            temp_weighted_mean = np.full(tas_celsius.shape[0], np.nan)
        else:
            temp_weighted_mean = np.array([
                np.sum(tas_celsius[t][mask] * frac_vec) / denom
                for t in range(tas_celsius.shape[0])
            ])

        df_daily = pd.DataFrame({'Date': time[valid_mask], 'Temperature': temp_weighted_mean})
        df_daily.set_index('Date', inplace=True)

        ds.close()
        return df_daily, mask, fractions, lat, lon, lat_bounds, lon_bounds

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None, None, None, None, None

# ------------------------------- #
# Model matching (historical vs scenarios)
# ------------------------------- #
def extract_model_prefix(filename: str) -> str | None:
    """Extract the model prefix from filename using scenario tags."""
    for tag in SCENARIO_TAGS:
        if tag in filename:
            return filename.split(tag)[0]
    return None

def clean_model_name(model_prefix: str) -> str:
    """Clean model name by removing common tas_day prefixes and .nc suffix."""
    cleaned = model_prefix.replace('tas_day_', '')
    cleaned = cleaned.replace('.nc', '')
    return cleaned

# Collect available files grouped by model prefix
all_files = defaultdict(dict)

# Historical
hist_dir = os.path.join(BASE_DIR, "historical")
if os.path.exists(hist_dir):
    nc_files = glob.glob(f"{hist_dir}/*.nc")
    for fpath in nc_files:
        fname = os.path.basename(fpath)
        prefix = extract_model_prefix(fname)
        if prefix:
            all_files[prefix]["historical"] = fpath

# Scenarios
for scenario in SCENARIOS:
    scenario_dir = os.path.join(BASE_DIR, scenario)
    if os.path.exists(scenario_dir):
        nc_files = glob.glob(f"{scenario_dir}/*.nc")
        for fpath in nc_files:
            fname = os.path.basename(fpath)
            prefix = extract_model_prefix(fname)
            if prefix:
                all_files[prefix][scenario] = fpath

# Match models by scenario
print("\n=== Model Matching ===")
matched_models = {sc: {} for sc in SCENARIOS}
for scenario in SCENARIOS:
    for prefix, files in all_files.items():
        if "historical" in files and scenario in files:
            matched_models[scenario][prefix] = {
                "historical": files["historical"],
                "scenario": files[scenario]
            }
    print(f"\n{scenario.upper()}: {len(matched_models[scenario])} models matched")
    for model in matched_models[scenario]:
        print(f" - {clean_model_name(model)}")

# ------------------------------- #
# Optional QA plotting
# ------------------------------- #
def plot_fraction_heatmap(fractions, lats, lons, basin_gdf, lat_bounds=None, lon_bounds=None, out_path=None):
    """Heatmap of fractional overlap (0–1) with basin boundary."""
    import matplotlib.pyplot as plt
    import numpy as np

    if lat_bounds is None:
        lat_bounds = infer_bounds(lats)
    if lon_bounds is None:
        lon_bounds = infer_bounds(lons)

    LonE, LatE = np.meshgrid(lon_bounds, lat_bounds)
    data_plot = np.where(fractions > 0, fractions, np.nan)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(LonE, LatE, data_plot, cmap="viridis", vmin=0.0, vmax=1.0, shading="auto")
    try:
        basin_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0)
    except Exception:
        pass

    ax.set_title("Fractional Overlap Weights (0–1) – Basin Mask on GCM Grid")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Overlap Fraction")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"Saved fractional overlap plot to: {out_path}")
    plt.show()
    return fig, ax

# ------------------------------- #
# Process scenario sets
# ------------------------------- #
for scenario in SCENARIOS:
    print(f"\n{'='*60}")
    print(f"Processing {scenario.upper()}")
    print(f"{'='*60}")

    annual_data = {}
    monthly_data_period1 = {}
    monthly_data_period2 = {}

    for model_prefix, files in tqdm(matched_models[scenario].items(), desc=f"Processing {scenario}"):
        hist_file = files["historical"]
        scen_file = files["scenario"]

        # Extract historical with area-true fractional weighting
        df_hist, mask, fractions, lat, lon, lat_bounds, lon_bounds = extract_daily_temperature(
            hist_file, shapefile_union=study_area_union, var_name=VAR_NAME
        )

        if df_hist is None:
            print(f"Skipping (no data) for {clean_model_name(model_prefix)}")
            continue

        # Reuse mask/fractions for scenario file (assumes same grid for model)
        df_scen, _, _, _, _, _, _ = extract_daily_temperature(
            scen_file, mask=mask, fractions=fractions, shapefile_union=study_area_union, var_name=VAR_NAME
        )
        if df_scen is None:
            print(f"Skipping scenario for {clean_model_name(model_prefix)}")
            continue

        # Combine
        df_combined = pd.concat([df_hist, df_scen]).sort_index()
        # Limit to desired range
        df_combined = df_combined[(df_combined.index.year >= 1985) & (df_combined.index.year <= 2100)]

        # Clean name
        clean_name = clean_model_name(model_prefix)

        # Annual mean tas
        annual_avg = df_combined.resample("Y").mean()
        annual_avg.index = annual_avg.index.year
        annual_data[clean_name] = annual_avg["Temperature"]

        # PERIOD 1: 1985–2014 monthly climatology (average of monthly means across years)
        df_period1 = df_combined[(df_combined.index.year >= PERIOD_1[0]) & (df_combined.index.year <= PERIOD_1[1])]
        if not df_period1.empty:
            monthly_avg_p1 = df_period1.groupby(df_period1.index.month)["Temperature"].mean()
            monthly_avg_p1.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            monthly_data_period1[clean_name] = monthly_avg_p1

        # PERIOD 2: 2071–2100 monthly climatology
        df_period2 = df_combined[(df_combined.index.year >= PERIOD_2[0]) & (df_combined.index.year <= PERIOD_2[1])]
        if not df_period2.empty:
            monthly_avg_p2 = df_period2.groupby(df_period2.index.month)["Temperature"].mean()
            monthly_avg_p2.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            monthly_data_period2[clean_name] = monthly_avg_p2

        # Optional QA plot once per scenario (first model)
        if MAKE_QA_PLOT:
            try:
                plot_fraction_heatmap(fractions, lat, lon, study_area, lat_bounds, lon_bounds, out_path=QA_PLOT_PATH)
                MAKE_QA_PLOT = False  # only make once
            except Exception as e:
                print(f"QA plot failed: {e}")

    # Build and save outputs
    df_annual = pd.DataFrame.from_dict(annual_data, orient='index').sort_index()
    df_annual.index.name = "Model"

    df_monthly_p1 = pd.DataFrame.from_dict(monthly_data_period1, orient='index').sort_index()
    df_monthly_p1.index.name = "Model"

    df_monthly_p2 = pd.DataFrame.from_dict(monthly_data_period2, orient='index').sort_index()
    df_monthly_p2.index.name = "Model"

    # Save to CSVs
    out_annual = os.path.join(BASE_DIR, f"tas_{scenario}_annual.csv")
    out_m_p1   = os.path.join(BASE_DIR, f"tas_{scenario}_monthly_1985_2014.csv")
    out_m_p2   = os.path.join(BASE_DIR, f"tas_{scenario}_monthly_2071_2100.csv")

    df_annual.to_csv(out_annual)
    df_monthly_p1.to_csv(out_m_p1)
    df_monthly_p2.to_csv(out_m_p2)

    print(f"\n✅ Saved {scenario} data:")
    print(f" - Annual (1985–2100): {os.path.basename(out_annual)}")
    print(f" - Monthly Climatology (1985–2014): {os.path.basename(out_m_p1)}")
    print(f" - Monthly Climatology (2071–2100): {os.path.basename(out_m_p2)}")

print("\n✅ All temperature data extraction completed!")
print(f"\nPeriod 1: {PERIOD_1[0]}–{PERIOD_1[1]} (Historical baseline)")
print(f"Period 2: {PERIOD_2[0]}–{PERIOD_2[1]} (Future projection)")
print("Monthly climatology represents area-weighted average temperature (°C) for each calendar month.")
print("Data weighted by grid cell overlap fraction, computed in an equal-area projection (LAEA).")