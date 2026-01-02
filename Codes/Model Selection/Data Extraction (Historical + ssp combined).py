# -*- coding: utf-8 -*-
"""
CMIP6 Precipitation (pr_day) → Basin-average with AREA WEIGHTING + QA plots
---------------------------------------------------------------------------
What this script does
- Loads CMIP6 daily precipitation (pr) files for 'historical' and scenarios (ssp245, ssp585).
- Builds grid-cell polygons, intersects them with the basin in an equal-area projection
  (Lambert Azimuthal Equal-Area, LAEA), and computes:
    * overlap fraction (0..1)   → for QA visualization
    * overlap area (m², km²)    → used as weights for precipitation (extensive variable)
- Converts pr from kg m⁻² s⁻¹ to mm/day (×86400).
- Computes the basin-average daily precipitation depth (mm/day) using **area weights**.
- Combines historical and scenario time series (1985–2100), writes:
    * Annual totals per model (CSV)
    * Monthly climatology (1985–2014 and 2071–2100) per model (CSV)
- Plots (similar to ERA5 precipitation script):
    * Fractional overlap QA heatmap (0–1) with basin outline
    * Daily basin series (mm/day)
    * Year–Month heatmap of monthly totals (mm)
    * Climatological monthly total (mm)

Requirements: netCDF4, numpy, pandas, geopandas, shapely, pyproj, matplotlib
No files are modified on disk; reprojection is done in memory.
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import netCDF4
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Polygon
from shapely.ops import transform
import pyproj
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ------------------------------- #
# Configuration
# ------------------------------- #
BASE_DIR = "C:/Users/Diwakar Adhikari/Downloads/GCMs/precipitation"
SHAPEFILE_PATH = "C:/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/Shp file/MRB.shp"

SCENARIOS = ["ssp245", "ssp585"]
SCENARIO_TAGS = ['_ssp245', '_ssp585', '_historical']
VAR_NAME = 'pr'  # kg m^-2 s^-1

# Periods
PERIOD_1 = (1985, 2014)  # Historical baseline period
PERIOD_2 = (2071, 2100)  # Future projection period

# Output directory (figures saved here as well)
OUT_DIR = BASE_DIR
MAKE_QA_PLOT = True           # plot fraction heatmap once per run
PLOT_FIRST_N_MODELS = 1       # to avoid too many figures, plot for first model per scenario
QA_PLOT_PATH = os.path.join(OUT_DIR, "fractional_overlap_weights_pr.png")

# CRS
GEOGRAPHIC_CRS = "EPSG:4326"  # lon/lat degrees
# We will use LAEA (equal-area) centered on the basin for area math
# (equivalent to using a stable global equal-area like EPSG:6933, but locally more accurate)

# ------------------------------- #
# Load shapefile & prepare union
# ------------------------------- #
print("Loading shapefile...")
basin_gdf = gpd.read_file(SHAPEFILE_PATH)
if basin_gdf.crs is None:
    basin_gdf = basin_gdf.set_crs(GEOGRAPHIC_CRS)
else:
    basin_gdf = basin_gdf.to_crs(GEOGRAPHIC_CRS)
basin_union_ll = basin_gdf.unary_union
print(f"Basin loaded with {len(basin_gdf)} feature(s).")

# ------------------------------- #
# Helpers: grid bounds & projection
# ------------------------------- #
def infer_bounds(coords: np.ndarray) -> np.ndarray:
    """Infer cell edge coordinates from centers (handles ascending/descending)."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n == 1:
        step = 0.5
        return np.array([coords[0] - step/2, coords[0] + step/2], dtype=float)
    edges = np.empty(n + 1, dtype=float)
    edges[1:-1] = (coords[:-1] + coords[1:]) / 2.0
    first_step = coords[1] - coords[0]
    last_step  = coords[-1] - coords[-2]
    edges[0]   = coords[0] - first_step / 2.0
    edges[-1]  = coords[-1] + last_step  / 2.0
    return edges

def build_cell_polygons(lats, lons, lat_bounds=None, lon_bounds=None):
    """Return list of per-cell lon/lat polygons and the grid shape."""
    nlat, nlon = len(lats), len(lons)
    if lat_bounds is None: lat_bounds = infer_bounds(lats)
    if lon_bounds is None: lon_bounds = infer_bounds(lons)
    polys = []
    for i in range(nlat):
        lat_min = min(lat_bounds[i],   lat_bounds[i+1])
        lat_max = max(lat_bounds[i],   lat_bounds[i+1])
        for j in range(nlon):
            lon_min = min(lon_bounds[j], lon_bounds[j+1])
            lon_max = max(lon_bounds[j], lon_bounds[j+1])
            polys.append(box(lon_min, lat_min, lon_max, lat_max))
    return polys, (nlat, nlon), lat_bounds, lon_bounds

def make_laea_transform(shapefile_union_ll):
    """Return a pyproj transform function to LAEA centered on basin centroid."""
    wgs84 = pyproj.CRS(GEOGRAPHIC_CRS)
    c = shapefile_union_ll.centroid
    laea = pyproj.CRS(f"+proj=laea +lat_0={c.y} +lon_0={c.x} +datum=WGS84 +units=m")
    to_laea = pyproj.Transformer.from_crs(wgs84, laea, always_xy=True).transform
    return to_laea

# ------------------------------- #
# Fraction & area computation
# ------------------------------- #
def compute_overlap_fraction_and_area(lats, lons, basin_union_ll):
    """
    Compute for every cell:
      - overlap fraction (0..1) = area(cell∩basin) / area(cell) in LAEA
      - overlap area (m²) = area(cell∩basin) in LAEA
    Returns: mask (bool), fractions (0..1), areas_m2, lat_bounds, lon_bounds
    """
    cell_polys_ll, shape, lat_bounds, lon_bounds = build_cell_polygons(lats, lons)
    to_laea = make_laea_transform(basin_union_ll)
    basin_laea = transform(to_laea, basin_union_ll)

    # Project cells to LAEA
    cells_ll_gdf = gpd.GeoDataFrame(geometry=cell_polys_ll, crs=GEOGRAPHIC_CRS)
    cells_laea = cells_ll_gdf.to_crs(basin_gdf.crs)  # ensure same CRS first (EPSG:4326)
    # Apply transform explicitly for robustness
    cells_laea.geometry = cells_laea.geometry.apply(lambda g: transform(to_laea, g))

    mask = np.zeros(len(cells_laea), dtype=bool)
    fractions = np.zeros(len(cells_laea), dtype=float)
    areas_m2 = np.zeros(len(cells_laea), dtype=float)

    for idx, poly in enumerate(cells_laea.geometry.values):
        a_cell = poly.area
        if a_cell <= 0:
            continue
        inter = poly.intersection(basin_laea)
        if inter.is_empty:
            continue
        a_inter = inter.area
        mask[idx] = a_inter > 0
        areas_m2[idx] = a_inter
        fractions[idx] = a_inter / a_cell

    # Reshape to grid
    mask = mask.reshape(shape)
    fractions = fractions.reshape(shape)
    areas_m2 = areas_m2.reshape(shape)
    return mask, fractions, areas_m2, lat_bounds, lon_bounds

# ------------------------------- #
# Data extraction per file
# ------------------------------- #
def extract_daily_precip_basin(file_path, mask=None, areas_m2=None,
                               basin_union_ll=None, var_name='pr'):
    """
    Read a single CMIP file and compute basin-average daily precipitation (mm/day)
    using **area weights** (m²). If mask/areas_m2 are None, compute them from this file.
    Returns: (df_daily_mm, mask, areas_m2, lat, lon, lat_bounds, lon_bounds)
    """
    try:
        ds = netCDF4.Dataset(file_path, 'r')

        # Time
        tvar = ds.variables.get('time')
        if tvar is None or var_name not in ds.variables:
            ds.close()
            return None, None, None, None, None, None, None

        times_cf = netCDF4.num2date(
            tvar[:],
            units=tvar.units,
            calendar=getattr(tvar, 'calendar', 'standard'),
            only_use_cftime_datetimes=True
        )
        times = pd.to_datetime([str(t) for t in times_cf], format='mixed', errors='coerce')
        valid_time = ~pd.isnull(times)
        if not np.any(valid_time):
            ds.close()
            return None, None, None, None, None, None, None

        # Coordinates (lat/lon variable names differ across models)
        lat_var = ds.variables.get('lat', ds.variables.get('latitude'))
        lon_var = ds.variables.get('lon', ds.variables.get('longitude'))
        if lat_var is None or lon_var is None:
            print(f"Missing lat/lon in {file_path}")
            ds.close()
            return None, None, None, None, None, None, None
        lat = np.asarray(lat_var[:], dtype=float)
        lon = np.asarray(lon_var[:], dtype=float)

        # Mask/areas if needed
        if mask is None or areas_m2 is None:
            if basin_union_ll is None:
                raise RuntimeError("Need basin_union_ll when computing mask/areas.")
            mask, fractions, areas_m2, lat_bounds, lon_bounds = compute_overlap_fraction_and_area(lat, lon, basin_union_ll)
        else:
            lat_bounds = infer_bounds(lat)
            lon_bounds = infer_bounds(lon)

        # Read precipitation and convert to mm/day
        pr = ds.variables[var_name][:]  # [time, lat, lon], kg m^-2 s^-1
        pr_mm_day = pr[valid_time] * 86400.0

        # AREA-weighted mean across space
        if not np.any(mask):
            ds.close()
            return None, None, None, None, None, None, None

        area_vec = areas_m2[mask]  # m²
        denom = np.sum(area_vec)
        if denom <= 0:
            basin_daily = np.full(pr_mm_day.shape[0], np.nan)
        else:
            basin_daily = np.array([
                np.sum(pr_mm_day[t][mask] * area_vec) / denom
                for t in range(pr_mm_day.shape[0])
            ])

        df_daily = pd.DataFrame({'Date': times[valid_time], 'Precip': basin_daily})
        df_daily.set_index('Date', inplace=True)

        ds.close()
        return df_daily, mask, areas_m2, lat, lon, lat_bounds, lon_bounds

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None, None, None, None, None

# ------------------------------- #
# Model matching (historical vs scenarios)
# ------------------------------- #
def extract_model_prefix(filename: str) -> str | None:
    for tag in SCENARIO_TAGS:
        if tag in filename:
            return filename.split(tag)[0]
    return None

def clean_model_name(model_prefix: str) -> str:
    cleaned = model_prefix.replace('pr_day_', '')
    cleaned = cleaned.replace('.nc', '')
    return cleaned

all_files = {}
# historical
hist_dir = os.path.join(BASE_DIR, "historical")
if os.path.exists(hist_dir):
    for f in glob.glob(f"{hist_dir}/*.nc"):
        prefix = extract_model_prefix(os.path.basename(f))
        if prefix:
            all_files.setdefault(prefix, {})["historical"] = f
# scenarios
for sc in SCENARIOS:
    sc_dir = os.path.join(BASE_DIR, sc)
    if os.path.exists(sc_dir):
        for f in glob.glob(f"{sc_dir}/*.nc"):
            prefix = extract_model_prefix(os.path.basename(f))
            if prefix:
                all_files.setdefault(prefix, {})[sc] = f

print("\n=== Model Matching ===")
matched_models = {sc: {} for sc in SCENARIOS}
for sc in SCENARIOS:
    for prefix, files in all_files.items():
        if "historical" in files and sc in files:
            matched_models[sc][prefix] = {"historical": files["historical"], "scenario": files[sc]}
    print(f"\n{sc.upper()}: {len(matched_models[sc])} models matched")
    for model in matched_models[sc]:
        print(f" - {clean_model_name(model)}")

# ------------------------------- #
# Plots (QA and products)
# ------------------------------- #
def plot_fraction_heatmap(fractions, lats, lons, basin_gdf, lat_bounds=None, lon_bounds=None, out_path=None):
    """Heatmap of overlap fraction (0–1) with basin outline."""
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
    ax.set_title("Fractional Overlap (0–1) – GCM Grid vs Basin")
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

def plot_basin_products(df_daily, scenario, model_name, out_dir):
    """
    Plots similar to ERA5 precipitation script:
      (1) Daily basin series (mm/day)
      (2) Year–Month heatmap of monthly totals (mm)
      (3) Climatological monthly total (mm)
    """
    plt.style.use("seaborn-v0_8")

    # Ensure regular datetime index
    s = df_daily["Precip"].copy()
    s = s.sort_index()

    # (1) Daily series
    fig1, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(s.index, s.values, color="tab:blue", lw=0.7)
    ax1.set_title(f"Daily Precipitation – Basin Mean (mm/day)\n{scenario.upper()} – {model_name}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("mm/day")
    ax1.grid(alpha=0.3)
    fig1.tight_layout()
    path1 = os.path.join(out_dir, f"{model_name}_{scenario}_daily_series.png")
    fig1.savefig(path1)

    # (2) Year–Month heatmap of monthly totals
    monthly_totals = s.resample("MS").sum()
    wide = (monthly_totals.to_frame("MonthlyTotal_mm")
            .assign(Year=lambda d: d.index.year, Month=lambda d: d.index.month)
            .pivot(index="Year", columns="Month", values="MonthlyTotal_mm")
            .sort_index())
    # Order months
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    wide.rename(columns=names, inplace=True)
    wide = wide.reindex(columns=list(names.values()))

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    im = ax2.imshow(wide.values, aspect="auto", cmap="Blues")
    ax2.set_title(f"Monthly Total Precipitation (mm)\n{scenario.upper()} – {model_name}")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Year")
    ax2.set_xticks(range(12)); ax2.set_xticklabels(list(wide.columns))
    ax2.set_yticks(range(len(wide.index))); ax2.set_yticklabels(wide.index)
    cbar = fig2.colorbar(im, ax=ax2); cbar.ax.set_ylabel("mm", rotation=90)
    fig2.tight_layout()
    path2 = os.path.join(out_dir, f"{model_name}_{scenario}_year_month_heatmap.png")
    fig2.savefig(path2)

    # (3) Climatological monthly total
    climo = (monthly_totals
             .groupby(monthly_totals.index.month)
             .mean()
             .rename(index=names))
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    ax3.plot(climo.index, climo.values, marker="o", color="tab:green")
    ax3.set_title(f"Climatology – Monthly Total (mm)\n{scenario.upper()} – {model_name}")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("mm")
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    path3 = os.path.join(out_dir, f"{model_name}_{scenario}_climo_monthly_cycle.png")
    fig3.savefig(path3)

    plt.show()
    print(f"Saved figures:\n  {path1}\n  {path2}\n  {path3}")

# ------------------------------- #
# Process per scenario
# ------------------------------- #
for scenario in SCENARIOS:
    print(f"\n{'='*60}\nProcessing {scenario.upper()}\n{'='*60}")

    annual_data = {}
    monthly_data_p1 = {}
    monthly_data_p2 = {}

    plotted_models = 0
    for model_prefix, files in tqdm(matched_models[scenario].items(), desc=f"{scenario} models"):
        hist_file = files["historical"]
        scen_file = files["scenario"]

        # Historical: compute mask/areas once per model grid
        df_hist, mask, areas_m2, lat, lon, lat_bounds, lon_bounds = extract_daily_precip_basin(
            hist_file, basin_union_ll=basin_union_ll, var_name=VAR_NAME
        )
        if df_hist is None:
            print(f"Skipping {clean_model_name(model_prefix)} (historical failed)")
            continue

        # Scenario: reuse mask/areas (assumes same grid)
        df_scen, _, _, _, _, _, _ = extract_daily_precip_basin(
            scen_file, mask=mask, areas_m2=areas_m2, basin_union_ll=basin_union_ll, var_name=VAR_NAME
        )
        if df_scen is None:
            print(f"Skipping scenario for {clean_model_name(model_prefix)}")
            continue

        # Combine and clip to 1985–2100
        df_combined = pd.concat([df_hist, df_scen]).sort_index()
        df_combined = df_combined[(df_combined.index.year >= 1985) & (df_combined.index.year <= 2100)]

        name = clean_model_name(model_prefix)

        # ----- Products for CSVs -----
        # Annual totals
        annual_total = df_combined.resample("Y").sum()
        annual_total.index = annual_total.index.year
        annual_data[name] = annual_total["Precip"]

        # Monthly climatologies for the two periods
        df_p1 = df_combined[(df_combined.index.year >= PERIOD_1[0]) & (df_combined.index.year <= PERIOD_1[1])]
        if not df_p1.empty:
            monthly_totals_p1 = df_p1.resample("M").sum()
            climo_p1 = monthly_totals_p1.groupby(monthly_totals_p1.index.month)["Precip"].mean()
            climo_p1.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly_data_p1[name] = climo_p1

        df_p2 = df_combined[(df_combined.index.year >= PERIOD_2[0]) & (df_combined.index.year <= PERIOD_2[1])]
        if not df_p2.empty:
            monthly_totals_p2 = df_p2.resample("M").sum()
            climo_p2 = monthly_totals_p2.groupby(monthly_totals_p2.index.month)["Precip"].mean()
            climo_p2.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            monthly_data_p2[name] = climo_p2

        # ----- QA plot (fraction heatmap) once -----
        if MAKE_QA_PLOT:
            try:
                # We need fractions for the QA plot. Recompute quickly using the same helper:
                _, fractions, _, _, _ = compute_overlap_fraction_and_area(lat, lon, basin_union_ll)
                plot_fraction_heatmap(fractions, lat, lon, basin_gdf, lat_bounds, lon_bounds, out_path=QA_PLOT_PATH)
            except Exception as e:
                print(f"QA plot failed: {e}")
            MAKE_QA_PLOT = False

        # ----- Per-model plots (limit to first N models per scenario) -----
        if plotted_models < PLOT_FIRST_N_MODELS:
            try:
                plot_basin_products(df_combined, scenario, name, OUT_DIR)
            except Exception as e:
                print(f"Plotting failed for {name}: {e}")
            plotted_models += 1

    # ----- Save CSVs for this scenario -----
    df_annual = pd.DataFrame.from_dict(annual_data, orient='index').sort_index()
    df_annual.index.name = "Model"

    df_monthly_p1 = pd.DataFrame.from_dict(monthly_data_p1, orient='index').sort_index()
    df_monthly_p1.index.name = "Model"

    df_monthly_p2 = pd.DataFrame.from_dict(monthly_data_p2, orient='index').sort_index()
    df_monthly_p2.index.name = "Model"

    out_annual = os.path.join(OUT_DIR, f"pr_{scenario}_annual.csv")
    out_m_p1   = os.path.join(OUT_DIR, f"pr_{scenario}_monthly_1985_2014.csv")
    out_m_p2   = os.path.join(OUT_DIR, f"pr_{scenario}_monthly_2071_2100.csv")

    df_annual.to_csv(out_annual)
    df_monthly_p1.to_csv(out_m_p1)
    df_monthly_p2.to_csv(out_m_p2)

    print(f"\n✅ Saved {scenario} outputs:")
    print(f" - Annual totals (1985–2100): {os.path.basename(out_annual)}")
    print(f" - Monthly climatology (1985–2014): {os.path.basename(out_m_p1)}")
    print(f" - Monthly climatology (2071–2100): {os.path.basename(out_m_p2)}")

print("\n✅ CMIP6 precipitation area-weighted processing complete.")
print("Basin mean depth is computed with absolute overlap area (m²) as weights;")
print("fractional overlap heatmap (0–1) is provided for QA; figures saved to BASE_DIR.")