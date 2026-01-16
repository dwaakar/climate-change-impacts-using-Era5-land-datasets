#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script — Temperature Regridding (bilinear) → Basin-only QA/QC → DEM-based Elevation Correction (CFTime-safe)
==============================================================================================

Purpose
-------
Step 1: Bilinear regridding of daily Tmax/Tmin (intensive variables) from GCM native grid to a user target grid
        (built from basin extent).
Step 2: DEM-based lapse-rate elevation correction using a global lapse rate with optional capping.

Notes
-----
* This script is adapted from a precipitation-focused workflow. It preserves directory handling, figures, CSVs,
  basin masking, CF-safe handling, and QA/QC structure, but replaces:
  - Regridding method → **bilinear** (temperature is intensive; no conservation needed).
  - Variable loading → supports `tasmax` or `tasmin`.
  - Unit conversion → Kelvin→Celsius if needed.
  - Orographic adjustment → replaced by DEM-based **lapse-rate correction** only (uses elevation; ignores slope/aspect/TPI).

Outputs per input file
----------------------
* NetCDF (basin-only): regridded temperature and elevation-corrected temperature (°C), lon/lat, basin fraction mask.
* QA/QC report (text): ranges, NaN shares, flat-day scan, regridding method and target grid info.
* Figures: basin overlap heatmap; monthly climatology of mean temperature; mean temperature vs elevation scatter.
* CSVs: daily basin-mean temperature (°C); monthly means (°C); yearly means (°C); full-grid daily CSV (basin cells only).

CF-time safety
--------------
The script avoids operations that would break non-standard calendars.

"""

import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import CRS
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds

try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except Exception:
    XESMF_AVAILABLE = False
    print("[ERROR] xESMF not available. Install via conda-forge: conda install -c conda-forge xesmf")

# ------------------------------ CONFIG ---------------------------------
# Update these paths before running
INPUT_DIR   = r'/mnt/c/users/Diwakar Adhikari/Downloads/Model Selection/GCM/tmin'  # folder with tasmax/tasmin NetCDFs
BASIN_SHP = r"/mnt/c/users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/Shp file/MRB.shp"
DEM_PATH = r"/mnt/c/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/DEM/MRB_JULY.tif"
OUTPUT_DIR  = r'/mnt/c/users/Diwakar Adhikari/Downloads/Model Selection/GCM/tmin/tmin_regrid/regrid_v1'
TARGET_RES_DEG   = 0.1        # target grid resolution in degrees
BBOX_BUFFER_DEG  = 0.0        # optional buffer around basin bbox (deg)

# DEM units handling: 'auto'|'m'|'ft'
DEM_UNITS = 'auto'

# Temperature variables to look for
T_VAR_CANDIDATES =["tasmin", "tmin"] #["tasmax", "tmax"] 

# Lapse-rate correction (°C/km); negative typical environmental lapse rate
LAPSE_RATE_C_PER_KM = -6.5
# Optional cap on absolute correction magnitude (°C) to avoid overcorrection
LAPSE_CORR_CAP_ABS_C = 3.0

# --------------------------- HELPERS (I/O & GRIDS) ----------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def infer_bounds(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    edges = np.empty(len(coords) + 1, dtype=float)
    edges[1:-1] = (coords[:-1] + coords[1:]) / 2.0
    edges[0]     = coords[0]  - (coords[1] - coords[0]) / 2.0
    edges[-1]    = coords[-1] + (coords[-1] - coords[-2]) / 2.0
    return edges

def build_grid_from_basin(basin_union_ll, res_deg: float, buffer_deg: float = 0.0):
    minx, miny, maxx, maxy = basin_union_ll.bounds
    minx -= buffer_deg; maxx += buffer_deg
    miny -= buffer_deg; maxy += buffer_deg
    lon_min = np.floor(minx / res_deg) * res_deg
    lon_max = np.ceil(maxx / res_deg) * res_deg
    lat_min = np.floor(miny / res_deg) * res_deg
    lat_max = np.ceil(maxy / res_deg) * res_deg
    lons = np.arange(lon_min, lon_max + 1e-9, res_deg)
    lats = np.arange(lat_min, lat_max + 1e-9, res_deg)
    lon_b = infer_bounds(lons)
    lat_b = infer_bounds(lats)
    return lons, lats, lon_b, lat_b

def cell_polygons(lons, lats, lon_b, lat_b):
    polys = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            polys.append(box(lon_b[j], lat_b[i], lon_b[j+1], lat_b[i+1]))
    return polys, (len(lats), len(lons))

def fractional_overlap(polys, basin_union) -> np.ndarray:
    fracs = np.zeros(len(polys), dtype=float)
    for k, poly in enumerate(polys):
        inter = poly.intersection(basin_union)
        if not inter.is_empty:
            fracs[k] = inter.area / poly.area
    return fracs

def build_xesmf_grid(lons, lats, lon_b, lat_b) -> xr.Dataset:
    Lon = np.tile(lons, (len(lats), 1))
    Lat = np.tile(lats.reshape(-1,1), (1, len(lons)))
    ds = xr.Dataset({'lon': (('y','x'), Lon), 'lat': (('y','x'), Lat)})
    ds['lon_b'] = (('y_b', 'x_b'), np.tile(lon_b, (len(lats)+1, 1)))
    ds['lat_b'] = (('y_b', 'x_b'), np.tile(lat_b.reshape(-1,1), (1, len(lons)+1)))
    return ds

def make_regridder(src: xr.Dataset, tgt: xr.Dataset) -> xe.Regridder:
    # Temperature is intensive: use bilinear
    return xe.Regridder(src, tgt, method='bilinear', periodic=False, reuse_weights=False)

# --------------------------- DEM HANDLING -------------------------------

def _maybe_convert_dem_units(elev: np.ndarray, src: rasterio.DatasetReader) -> np.ndarray:
    global DEM_UNITS
    elev2 = elev.copy()
    # explicit config first
    if DEM_UNITS == 'ft':
        elev2 = elev2 / 3.280839895
    elif DEM_UNITS == 'm':
        pass
    else:
        # auto-detect via metadata
        try:
            unit_meta = (src.tags().get('UNITTYPE') or src.tags().get('VERT_UNIT') or src.tags().get('VERTICAL_UNITS') or '').lower()
        except Exception:
            unit_meta = ''
        if any(k in unit_meta for k in ['foot','feet','ft']):
            elev2 = elev2 / 3.280839895
        else:
            mx = np.nanmax(elev2)
            if np.isfinite(mx) and mx > 12000:  # likely feet
                elev2 = elev2 / 3.280839895
    return elev2

def dem_to_given_grid(dem_path: str, lons: np.ndarray, lats: np.ndarray) -> Dict[str, np.ndarray]:
    """Project DEM to a given lon/lat grid (rectilinear), returning elevation (m) and also slope/aspect/TPI (unused)."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata

        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        width, height = len(lons), len(lats)
        dst_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width=width, height=height)
        dst_crs = CRS.from_epsg(4326)

        dst = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=dem,
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src_nodata,
            dst_nodata=np.nan
        )
        elev = np.where(np.isfinite(dst), dst, np.nan)
        elev = _maybe_convert_dem_units(elev, src)
        elev = np.where((elev < -500) | (elev > 9000), np.nan, elev)  # clamp unreasonable

        # compute gradients (not used later for temperature, but kept for parity)
        dy = np.deg2rad(np.gradient(lats)) * 6371000.0
        dx = np.deg2rad(np.gradient(lons)) * 6371000.0 * np.cos(np.deg2rad(lats)).reshape(-1,1)
        gy, gx = np.gradient(elev)
        with np.errstate(invalid='ignore', divide='ignore'):
            gy = gy / dy.reshape(-1,1)
            gx = gx / dx
        slope = np.sqrt(gx**2 + gy**2)
        aspect = (np.degrees(np.arctan2(gx, -gy)) + 360.0) % 360.0
        from scipy.ndimage import uniform_filter
        elev_mean = uniform_filter(np.nan_to_num(elev), size=5)
        tpi = elev - elev_mean
        return {"elev": elev, "slope": slope, "aspect": aspect, "tpi": tpi}

# --------------------------- BASIN & WEIGHTING --------------------------

def basin_fraction_matrix(lons, lats, lon_b, lat_b, basin_union_ll):
    polys, shape = cell_polygons(lons, lats, lon_b, lat_b)
    fracs = fractional_overlap(polys, basin_union_ll)
    return fracs.reshape(shape)

def area_weighted_basin_mean(da: xr.DataArray, frac: np.ndarray) -> xr.DataArray:
    w = xr.DataArray(frac, dims=("y","x"))
    w_norm = w / w.sum()
    ts = (da * w_norm).sum(dim=("y","x"))
    ts.name = 'basin_mean'; ts.attrs['units'] = da.attrs.get('units','')
    return ts

# --------------------------- TEMPERATURE HELPERS ------------------------

def load_temperature_variable(ds: xr.Dataset) -> Tuple[str, xr.DataArray]:
    for v in T_VAR_CANDIDATES:
        if v in ds.data_vars:
            return v, ds[v]
    # Try by CF standard_name
    for v, da in ds.data_vars.items():
        stdn = str(da.attrs.get('standard_name','')).lower()
        if stdn in ('air_temperature','air_temperature_maximum','air_temperature_minimum'):
            return v, da
    raise KeyError("No temperature variable (tasmax/tasmin) found in dataset.")

def to_celsius(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get('units','')).lower()
    out = da
    if units in ('k','kelvin') or units == '' or 'kelvin' in units:
        out = da - 273.15
        out.attrs['units'] = 'degC'
    elif 'c' in units:
        out.attrs['units'] = 'degC'
    else:
        # fallback: assume Kelvin
        out = da - 273.15
        out.attrs['units'] = 'degC'
    return out

def coord_ranges_ok(lon: np.ndarray, lat: np.ndarray) -> bool:
    lon_ok = np.all(np.isfinite(lon)) and ((np.nanmin(lon) >= -360) and (np.nanmax(lon) <= 360))
    lat_ok = np.all(np.isfinite(lat)) and ((np.nanmin(lat) >= -90) and (np.nanmax(lat) <= 90))
    return lon_ok and lat_ok

def bounds_monotonic_ok(bounds: np.ndarray) -> bool:
    b = np.asarray(bounds, dtype=float); diffs = np.diff(b)
    return np.all(np.isfinite(diffs)) and (np.all(diffs > 0) or np.all(diffs < 0))

# --------------------------- LAPSE-RATE CORRECTION ----------------------

def apply_lapse_rate_correction(t_regridded_c: xr.DataArray,
                                elev_target: np.ndarray,
                                elev_src_regridded_to_target: np.ndarray,
                                lapse_c_per_km: float = LAPSE_RATE_C_PER_KM,
                                cap_abs_c: Optional[float] = LAPSE_CORR_CAP_ABS_C) -> xr.DataArray:
    """DEM-based correction using Δz between target DEM and source-grid DEM regridded to target grid.
    T_corr = T_regrid + gamma * (Δz/1000), with optional |ΔT| cap.
    """
    dz_m = elev_target - elev_src_regridded_to_target
    delta_c = lapse_c_per_km * (dz_m / 1000.0)
    if cap_abs_c is not None and np.isfinite(cap_abs_c):
        delta_c = np.clip(delta_c, -abs(cap_abs_c), +abs(cap_abs_c))
    out = t_regridded_c.copy()
    # broadcast delta across time
    for t in range(out.sizes['time']):
        base = out.isel(time=t).values
        out.values[t,:,:] = base + delta_c
    out.attrs['long_name'] = (t_regridded_c.attrs.get('long_name','regridded temperature') +
                               ' with DEM-based elevation correction')
    out.attrs['elevation_correction'] = f"lapse_rate={lapse_c_per_km} C/km, cap_abs={cap_abs_c} C"
    return out

# --------------------------- PLOTS & DIAGNOSTICS ------------------------

def plot_fraction_heatmap(frac: np.ndarray, lons: np.ndarray, lats: np.ndarray, out_path: str):
    Lon, Lat = np.meshgrid(lons, lats)
    plt.figure(figsize=(10,6))
    im = plt.pcolormesh(Lon, Lat, frac, cmap='viridis', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im, label='Overlap fraction')
    plt.title('Basin fractional overlap (target grid)')
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_monthly_climatology_mean(ts_basin_c: xr.DataArray, out_path: str):
    # monthly mean temperatures (degC) then mean across years by month
    monthly_means = ts_basin_c.resample(time='MS').mean()
    clim = monthly_means.groupby('time.month').mean('time')
    months_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_nums = clim['month'].values.tolist()
    vals = [float(clim.sel(month=m).values) if m in month_nums else np.nan for m in range(1,13)]
    plt.figure(figsize=(8,4))
    plt.bar(months_labels, vals, color='tab:red')
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel('°C'); plt.title('Basin monthly climatology (mean of monthly means)')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_elev_gradient(t_mean_2d_c: xr.DataArray, elev: np.ndarray, out_path: str):
    plt.figure(figsize=(8,5))
    plt.scatter(elev.flatten(), t_mean_2d_c.values.flatten(), s=6, alpha=0.6)
    plt.xlabel('Elevation (m)'); plt.ylabel('°C'); plt.title('Grid-cell mean temperature vs elevation')
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def write_fullgrid_csv_for_swat(da: xr.DataArray, lon2d: np.ndarray, lat2d: np.ndarray, basin_frac: np.ndarray, out_csv: str):
    mask = basin_frac > 0
    y_idx, x_idx = np.where(mask)
    labels = [f"lat{lat2d[i,j]:.5f}_lon{lon2d[i,j]:.5f}" for i,j in zip(y_idx, x_idx)]
    date_strings = da['time'].dt.strftime('%Y-%m-%d').values
    records = []
    for t_idx, date in enumerate(date_strings):
        slice2d = da.isel(time=t_idx).values
        vals = [slice2d[i,j] for i,j in zip(y_idx, x_idx)]
        records.append([date] + vals)
    df = pd.DataFrame(records, columns=['date'] + labels)
    df.to_csv(out_csv, index=False)

def write_basin_diagnostics(ts_basin: xr.DataArray, out_dir: str, stem: str):
    # Daily basin mean (°C)
    df_daily = pd.DataFrame({'date': ts_basin['time'].dt.strftime('%Y-%m-%d').values,
                             'basin_mean_degC': ts_basin.values})
    df_daily.to_csv(os.path.join(out_dir, f"{stem}_daily_basin_mean_degC.csv"), index=False)

    # Monthly means (°C)
    monthly_means = ts_basin.resample(time='MS').mean()
    df_monthly = pd.DataFrame({'date': monthly_means['time'].dt.strftime('%Y-%m').values,
                               'degC': monthly_means.values})
    df_monthly.to_csv(os.path.join(out_dir, f"{stem}_monthly_basin_means_degC.csv"), index=False)

    # Yearly means (°C)
    yearly_means = ts_basin.resample(time='YS').mean()
    df_yearly = pd.DataFrame({'year': yearly_means['time'].dt.year.values,
                              'degC': yearly_means.values})
    df_yearly.to_csv(os.path.join(out_dir, f"{stem}_yearly_basin_means_degC.csv"), index=False)

# --------------------------- QA/QC --------------------------------------

def temperature_qc_report(da_src_c: xr.DataArray,
                          da_tgt_regrid_c: xr.DataArray,
                          da_tgt_elevcorr_c: xr.DataArray,
                          ds_tgt_grid: xr.Dataset,
                          lapse_info: str) -> str:
    report = []
    # Units and basic stats
    def _rng(da):
        vmin = float(np.nanmin(da.values)) if np.isfinite(da.values).any() else np.nan
        vmax = float(np.nanmax(da.values)) if np.isfinite(da.values).any() else np.nan
        return vmin, vmax

    src_min, src_max = _rng(da_src_c)
    r_min, r_max     = _rng(da_tgt_regrid_c)
    e_min, e_max     = _rng(da_tgt_elevcorr_c)

    report.append(f"Source (native) temperature [degC] range: {src_min:.3f} to {src_max:.3f}")
    report.append(f"Regridded temperature [degC] range: {r_min:.3f} to {r_max:.3f}")
    report.append(f"Elevation-corrected temperature [degC] range: {e_min:.3f} to {e_max:.3f}")

    # NaN share
    def _nan_share(da):
        arr = da.values
        return 100.0 * (np.isnan(arr).sum() / arr.size)
    report.append(f"NaN share (regridded): {_nan_share(da_tgt_regrid_c):.3f}%")
    report.append(f"NaN share (elev-corrected): {_nan_share(da_tgt_elevcorr_c):.3f}%")

    # Flat-day scan (min≈max across grid)
    mins = da_tgt_regrid_c.min(dim=('y','x')).values
    maxs = da_tgt_regrid_c.max(dim=('y','x')).values
    flat = np.where(np.abs(maxs - mins) <= 1e-12)[0]
    if flat.size > 0:
        dates_str = da_tgt_regrid_c['time'].dt.strftime('%Y-%m-%d').values
        first10 = ", ".join([str(dates_str[k]) for k in flat[:10]])
        report.append(f"[QC] {flat.size} flat day(s) detected where min≈max across grid. First 10 dates: {first10}")

    # Grid info and method
    ny, nx = ds_tgt_grid['lat'].shape
    report.append(f"Target grid: {ny} lat x {nx} lon (bilinear regridding)")
    report.append(f"Elevation correction details: {lapse_info}")
    return "\n".join(report)

# --------------------------- MAIN --------------------------------------

def main():
    if not XESMF_AVAILABLE:
        sys.exit(1)

    ensure_dir(OUTPUT_DIR)
    fig_dir = os.path.join(OUTPUT_DIR, 'figures'); ensure_dir(fig_dir)

    # Basin geometry in WGS84
    basin_gdf = gpd.read_file(BASIN_SHP)
    if basin_gdf.crs is None:
        basin_gdf = basin_gdf.set_crs(4326)
    else:
        basin_gdf = basin_gdf.to_crs(4326)
    basin_union_ll = unary_union(basin_gdf.geometry)

    # Build target grid
    lons, lats, lon_b, lat_b = build_grid_from_basin(basin_union_ll, TARGET_RES_DEG, buffer_deg=BBOX_BUFFER_DEG)
    ds_tgt = build_xesmf_grid(lons, lats, lon_b, lat_b)
    if not (coord_ranges_ok(ds_tgt['lon'].values, ds_tgt['lat'].values) and
            bounds_monotonic_ok(lon_b) and bounds_monotonic_ok(lat_b)):
        print("[ERROR] Target grid coordinates/bounds invalid."); sys.exit(1)

    # Basin fraction (target grid)
    frac_mat = basin_fraction_matrix(lons, lats, lon_b, lat_b, basin_union_ll)
    plot_fraction_heatmap(frac_mat, lons, lats, os.path.join(fig_dir, 'fraction_overlap_target.png'))

    # DEM on target grid (for final elevation)
    terrain_tgt = dem_to_given_grid(DEM_PATH, lons, lats)
    print(f"[INFO] DEM (target grid) stats: min={np.nanmin(terrain_tgt['elev']):.2f}, max={np.nanmax(terrain_tgt['elev']):.2f} m")

    nc_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.nc')))
    if len(nc_files) == 0:
        print(f"[ERROR] No NetCDF files found in {INPUT_DIR}"); sys.exit(1)

    for nc in nc_files:
        print(f"\n=== Processing: {os.path.basename(nc)} ===")
        ds_src = xr.open_dataset(nc)

        # Load variable and convert to Celsius
        var_name, tvar = load_temperature_variable(ds_src)
        t_c_src = to_celsius(tvar)

        # Source grid (rectilinear lat/lon expected)
        if {'lat','lon'}.issubset(set(t_c_src.dims)):
            lat_src = ds_src['lat'].values; lon_src = ds_src['lon'].values
        elif {'latitude','longitude'}.issubset(set(ds_src.coords)):
            lat_src = ds_src['latitude'].values; lon_src = ds_src['longitude'].values
        else:
            raise ValueError("Unsupported coordinate names; expecting lat/lon or latitude/longitude.")

        lon_b_src = infer_bounds(lon_src); lat_b_src = infer_bounds(lat_src)
        Lon_src, Lat_src = np.meshgrid(lon_src, lat_src)
        ds_src_grid = xr.Dataset({'lon': (('y','x'), Lon_src), 'lat': (('y','x'), Lat_src),
                                  'lon_b': (('y_b','x_b'), np.tile(lon_b_src, (len(lat_src)+1,1))),
                                  'lat_b': (('y_b','x_b'), np.tile(lat_b_src.reshape(-1,1), (1,len(lon_src)+1))) })

        # Align dims to (time,y,x)
        t_da = t_c_src.transpose(...)
        rename_map = {}
        for d in t_da.dims:
            dl = d.lower()
            if dl in ('lat','latitude'): rename_map[d] = 'y'
            if dl in ('lon','longitude'): rename_map[d] = 'x'
        if rename_map:
            t_da = t_da.rename(rename_map)

        # Regridder (bilinear)
        regridder = make_regridder(ds_src_grid, ds_tgt)
        t_regridded = regridder(t_da)
        t_regridded.name = f"{var_name}_regrid"; t_regridded.attrs['units'] = 'degC'
        t_regridded.attrs['long_name'] = f"Regridded {var_name} (degC)"

        # DEM on source grid (what model "saw" at coarse scale)
        terrain_src = dem_to_given_grid(DEM_PATH, lon_src, lat_src)
        elev_src_on_srcgrid = xr.DataArray(terrain_src['elev'], dims=("y","x"))
        elev_src_regridded_to_tgt = regridder(elev_src_on_srcgrid)

        # Lapse-rate elevation correction on target grid
        t_elevcorr = apply_lapse_rate_correction(t_regridded,
                                                elev_target=terrain_tgt['elev'],
                                                elev_src_regridded_to_target=elev_src_regridded_to_tgt.values,
                                                lapse_c_per_km=LAPSE_RATE_C_PER_KM,
                                                cap_abs_c=LAPSE_CORR_CAP_ABS_C)
        t_elevcorr.name = f"{var_name}_elev"; t_elevcorr.attrs['units'] = 'degC'

        # Basin-only mask for outputs
        mask = xr.DataArray((frac_mat > 0), dims=('y','x'))
        t_regrid_basin   = t_regridded.where(mask)
        t_elevcorr_basin = t_elevcorr.where(mask)

        # Basin mean (°C)
        ts_basin_xr = area_weighted_basin_mean(t_regrid_basin, frac_mat)

        # Figures
        stem = os.path.basename(nc).replace('.nc','')
        plot_monthly_climatology_mean(ts_basin_xr, os.path.join(fig_dir, f"{stem}_clim_monthly_means.png"))
        t_mean_2d = t_regrid_basin.mean(dim='time')
        plot_elev_gradient(t_mean_2d, terrain_tgt['elev'], os.path.join(fig_dir, f"{stem}_elev_gradient.png"))

        # Output dataset (EPSG:4326)
        ds_out = xr.Dataset({f"{var_name}_regrid": t_regrid_basin,
                             f"{var_name}_elev": t_elevcorr_basin,
                             'basin_frac': (('y','x'), frac_mat),
                             'lon': (('y','x'), ds_tgt['lon'].values),
                             'lat': (('y','x'), ds_tgt['lat'].values)})
        ds_out.attrs['title'] = f"Basin-only regridded {var_name} (degC) with DEM-based elevation correction"
        ds_out.attrs['source_file'] = os.path.basename(nc)
        ds_out.attrs['Conventions'] = 'CF-1.8'; ds_out.attrs['crs'] = 'EPSG:4326'
        ds_out.attrs['geospatial_lon_units'] = 'degrees_east'; ds_out.attrs['geospatial_lat_units'] = 'degrees_north'
        ds_out.attrs['regridding_method'] = 'bilinear'
        ds_out.attrs['elevation_correction'] = f"lapse_rate={LAPSE_RATE_C_PER_KM} C/km, cap_abs={LAPSE_CORR_CAP_ABS_C} C"

        out_nc = os.path.join(OUTPUT_DIR, f"{stem}_basin_temp_v1.nc")
        comp = dict(zlib=True, complevel=4)
        encoding = {k: comp for k in ds_out.data_vars}
        ds_out.to_netcdf(out_nc, encoding=encoding)
        print(f"[OK] Saved basin NetCDF: {out_nc}")

        # Diagnostics CSVs
        write_basin_diagnostics(ts_basin_xr, OUTPUT_DIR, stem)
        print("[OK] Saved basin diagnostic CSVs (daily/monthly/yearly means).")

        # Full-grid CSV for SWAT (basin cells only)
        out_csv_swat = os.path.join(OUTPUT_DIR, f"{stem}_daily_grid_for_SWAT.csv")
        write_fullgrid_csv_for_swat(t_regridded, ds_tgt['lon'].values, ds_tgt['lat'].values, frac_mat, out_csv_swat)
        print(f"[OK] Saved SWAT full-grid CSV (basin cells only): {out_csv_swat}")

        # QA/QC text
        qa_path = os.path.join(OUTPUT_DIR, f"{stem}_QA_temp_v1.txt")
        lapse_info = f"lapse_rate={LAPSE_RATE_C_PER_KM} C/km; cap_abs={LAPSE_CORR_CAP_ABS_C} C"
        qa_text = temperature_qc_report(t_da, t_regridded, t_elevcorr, ds_tgt, lapse_info)
        with open(qa_path, 'w') as f:
            f.write("QA/QC Report (Temperature, CFTime-safe)\n")
            f.write("======================================\n\n")
            f.write(f"Input file: {nc}\n")
            f.write(f"Variable processed: {var_name}\n")
            f.write(f"Target grid: {len(lats)} lat x {len(lons)} lon (res {TARGET_RES_DEG}°)\n\n")
            f.write("Regridding & Elevation Correction:\n")
            f.write(qa_text + "\n")
        print(f"[OK] QA saved: {qa_path}")

        print("\n✅ Temperature regrid + elevation correction complete for this file.")

if __name__ == '__main__':
    main()