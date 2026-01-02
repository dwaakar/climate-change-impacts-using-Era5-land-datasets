#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 — Regridding → Basin-only QA/QC → Orographic Adjustment (v3.1, CFTime-safe)
====================================================================================
Fixes vs v3:
- DEM handling: honor src nodata, set dst nodata to NaN, optional DEM units flag, auto-detect feet,
  clamp unrealistic elevations (< -500 m or > 9000 m) to NaN to avoid artifacts.
- Climatology: compute MONTHLY CLIMATOLOGY as the mean of monthly totals across years (mm/month),
  not the mean of daily rates.
- Keeps basin-only NetCDF (EPSG:4326), 2nd-order conservative with 1st-order fallback, SWAT CSV, and QA.
"""
import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')
from typing import Dict
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

# ----------------------------- CONFIG -----------------------------
INPUT_DIR = r'/mnt/c/users/Diwakar Adhikari/Downloads/Model Selection/GCM/precipitation'
BASIN_SHP = r"/mnt/c/users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/Shp file/MRB.shp"
DEM_PATH = r"/mnt/c/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/DEM/MRB_JULY.tif"
OUTPUT_DIR = r'/mnt/c/users/Diwakar Adhikari/Downloads/Model Selection/GCM/precipitation/regrid_v4'
TARGET_RES_DEG = 0.1
BBOX_BUFFER_DEG = 0.0
PR_VAR_CANDIDATES = ["pr", "precip", "precipitation", "tp"]
# If your DEM is known to be in feet, set to 'ft'; if meters, 'm'; if unknown, leave 'auto'.
DEM_UNITS = 'auto'  # 'auto' | 'm' | 'ft'
SEASONS = {
    "monsoon": [6, 7, 8, 9],
    "winter" : [12, 1, 2],
    "transition": [3, 4, 5, 10, 11]
}
OROG_WEIGHTS = {
    "monsoon": {"elev": 0.10, "aspect": 0.20, "slope": 0.10, "tpi": 0.05, "flow_dir_deg": 180},
    "winter" : {"elev": 0.08, "aspect": 0.15, "slope": 0.08, "tpi": 0.05, "flow_dir_deg": 270},
    "transition": {"elev": 0.06, "aspect": 0.10, "slope": 0.06, "tpi": 0.04, "flow_dir_deg": 225}
}
# ------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def infer_bounds(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    edges = np.empty(len(coords) + 1, dtype=float)
    edges[1:-1] = (coords[:-1] + coords[1:]) / 2.0
    edges[0] = coords[0] - (coords[1] - coords[0]) / 2.0
    edges[-1] = coords[-1] + (coords[-1] - coords[-2]) / 2.0
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

def load_pr_variable(ds: xr.Dataset) -> xr.DataArray:
    for v in PR_VAR_CANDIDATES:
        if v in ds.data_vars:
            return ds[v]
    for v, da in ds.data_vars.items():
        if str(da.attrs.get('standard_name', '')).lower() in ['precipitation_flux','precipitation']:
            return da
    raise KeyError("No precipitation variable found.")

def to_mm_per_day(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get('units', '')).lower()
    if 'kg' in units and 'm-2' in units and 's-1' in units:
        out = da * 86400.0
    elif units in ('m', 'metre', 'meter', 'meters'):
        out = da * 1000.0
    elif ('mm/day' in units) or ('mm d-1' in units) or ('mm per day' in units):
        out = da
    else:
        out = da * 86400.0
    out.attrs['units'] = 'mm/day'
    return out

def build_xesmf_grid(lons, lats, lon_b, lat_b) -> xr.Dataset:
    Lon = np.tile(lons, (len(lats), 1))
    Lat = np.tile(lats.reshape(-1,1), (1, len(lons)))
    ds = xr.Dataset({'lon': (('y','x'), Lon), 'lat': (('y','x'), Lat)})
    ds['lon_b'] = (('y_b','x_b'), np.tile(lon_b, (len(lats)+1, 1)))
    ds['lat_b'] = (('y_b','x_b'), np.tile(lat_b.reshape(-1,1), (1, len(lons)+1)))
    return ds

def make_regridder(src: xr.Dataset, tgt: xr.Dataset) -> xe.Regridder:
    try:
        return xe.Regridder(src, tgt, method='conservative_2nd', periodic=False, reuse_weights=False)
    except Exception as e2:
        print(f"[WARN] conservative_2nd failed: {e2}. Falling back to conservative (1st-order).")
        return xe.Regridder(src, tgt, method='conservative', periodic=False, reuse_weights=False)

# ---------------- DEM handling (units, nodata, reprojection) ----------------

def _maybe_convert_dem_units(elev: np.ndarray, src: rasterio.DatasetReader) -> np.ndarray:
    global DEM_UNITS
    elev2 = elev.copy()
    # Try explicit config first
    if DEM_UNITS == 'ft':
        elev2 = elev2 / 3.280839895
    elif DEM_UNITS == 'm':
        pass
    else:
        # auto-detect via common metadata keys
        unit_meta = None
        try:
            unit_meta = (src.tags().get('UNITTYPE') or src.tags().get('VERT_UNIT') or src.tags().get('VERTICAL_UNITS') or '').lower()
        except Exception:
            unit_meta = ''
        if 'foot' in unit_meta or 'feet' in unit_meta or 'ft' in unit_meta:
            elev2 = elev2 / 3.280839895
        else:
            # heuristic: if max > 12000, likely feet; convert
            mx = np.nanmax(elev2)
            if np.isfinite(mx) and mx > 12000:
                elev2 = elev2 / 3.280839895
    return elev2

def dem_to_target_grid(dem_path: str, lons: np.ndarray, lats: np.ndarray) -> Dict[str, np.ndarray]:
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
        # Units
        elev = _maybe_convert_dem_units(elev, src)
        # Clamp unrealistic values
        elev = np.where((elev < -500) | (elev > 9000), np.nan, elev)
    # gradients for slope/aspect
    dy = np.deg2rad(np.gradient(lats)) * 6371000.0
    dx = np.deg2rad(np.gradient(lons)) * 6371000.0 * np.cos(np.deg2rad(lats)).reshape(-1,1)
    gy, gx = np.gradient(elev)
    with np.errstate(invalid='ignore', divide='ignore'):
        gy = gy / dy.reshape(-1,1)
        gx = gx / dx
        slope = np.sqrt(gx**2 + gy**2)
        aspect = (np.rad2deg(np.arctan2(gx, -gy)) + 360.0) % 360.0
    from scipy.ndimage import uniform_filter
    elev_mean = uniform_filter(np.nan_to_num(elev), size=5)
    tpi = elev - elev_mean
    return {"elev": elev, "slope": slope, "aspect": aspect, "tpi": tpi}

# ---------------- Basin weighting & QA ----------------

def normalize(arr: np.ndarray) -> np.ndarray:
    a = arr.copy(); m = np.nanmean(a); s = np.nanstd(a)
    if s == 0 or np.isnan(s):
        return np.zeros_like(a)
    return (a - m) / s

def exposure_factor(aspect_deg: np.ndarray, flow_dir_deg: float) -> np.ndarray:
    ang = np.deg2rad(aspect_deg - flow_dir_deg)
    return np.cos(ang)

def apply_orographic_adjustment(pr_mmday: xr.DataArray, terrain: Dict[str, np.ndarray], months: np.ndarray, basin_frac: np.ndarray) -> xr.DataArray:
    elev_n = normalize(terrain['elev']); slope_n = normalize(terrain['slope']); tpi_n = normalize(terrain['tpi'])
    aspect_deg = terrain['aspect']
    pr_out = pr_mmday.copy()
    mask = basin_frac > 0
    for t in range(pr_mmday.shape[0]):
        m = int(months[t])
        if m in SEASONS['monsoon']:
            w = OROG_WEIGHTS['monsoon']
        elif m in SEASONS['winter']:
            w = OROG_WEIGHTS['winter']
        else:
            w = OROG_WEIGHTS['transition']
        expo = exposure_factor(aspect_deg, w['flow_dir_deg'])
        factor = 1.0 + (w['elev']*elev_n) + (w['slope']*slope_n) + (w['tpi']*tpi_n) + (w['aspect']*expo)
        base = pr_mmday[t,:,:].values
        adj = base.copy(); adj[mask] = base[mask] * factor[mask]
        pr_out[t,:,:] = adj
    pr_out.name = 'pr_orog'
    pr_out.attrs['long_name'] = 'Orographically adjusted precipitation (mm/day)'
    pr_out.attrs['units'] = 'mm/day'
    return pr_out

def basin_fraction_matrix(lons, lats, lon_b, lat_b, basin_union_ll):
    polys, shape = cell_polygons(lons, lats, lon_b, lat_b)
    fracs = fractional_overlap(polys, basin_union_ll)
    return fracs.reshape(shape)

def area_weighted_basin_mean(pr_mmday: xr.DataArray, frac: np.ndarray) -> xr.DataArray:
    w = xr.DataArray(frac, dims=("y","x"))
    w_norm = w / w.sum()
    ts = (pr_mmday * w_norm).sum(dim=("y","x"))
    ts.name = 'basin_mean_mmday'; ts.attrs['units'] = 'mm/day'
    return ts

# QA helpers

def cell_areas_from_bounds(lon_b: np.ndarray, lat_b: np.ndarray) -> np.ndarray:
    lonb = np.deg2rad(lon_b); latb = np.deg2rad(lat_b)
    dlon = np.diff(lonb)[None, :]; dphi = np.diff(latb)[:, None]
    phi_mid = 0.5*(latb[:-1,None] + latb[1:,None])
    R = 6371000.0
    A = (np.cos(phi_mid) * dphi * dlon) * (R**2)
    return A

def mass_conservation_report(pr_src_mmday: xr.DataArray,
                             pr_tgt_mmday: xr.DataArray,
                             lon_b_src: np.ndarray, lat_b_src: np.ndarray,
                             lon_b_tgt: np.ndarray, lat_b_tgt: np.ndarray,
                             basin_frac_tgt: np.ndarray,
                             ds_src_grid: xr.Dataset, ds_tgt_grid: xr.Dataset) -> str:
    A_src = xr.DataArray(cell_areas_from_bounds(lon_b_src, lat_b_src), dims=("y","x"))
    A_tgt = xr.DataArray(cell_areas_from_bounds(lon_b_tgt, lat_b_tgt), dims=("y","x"))
    m_src = (pr_src_mmday * A_src).sum(("y","x","time")).item()
    m_tgt = (pr_tgt_mmday * A_tgt).sum(("y","x","time")).item()
    diff_domain = 100.0 * (m_tgt - m_src) / (m_src if m_src != 0 else np.nan)
    try:
        regridder_basin = xe.Regridder(ds_tgt_grid, ds_src_grid, method='conservative', periodic=False, reuse_weights=False)
        basin_frac_src = regridder_basin(xr.DataArray(basin_frac_tgt, dims=("y","x")))
        basin_frac_src = xr.where(basin_frac_src < 0, 0, basin_frac_src)
        W_src = (xr.DataArray(cell_areas_from_bounds(lon_b_src, lat_b_src), dims=("y","x")) * basin_frac_src)
        W_tgt = (xr.DataArray(cell_areas_from_bounds(lon_b_tgt, lat_b_tgt), dims=("y","x")) * xr.DataArray(basin_frac_tgt, dims=("y","x")))
        m_src_basin = (pr_src_mmday * W_src).sum(("y","x","time")).item()
        m_tgt_basin = (pr_tgt_mmday * W_tgt).sum(("y","x","time")).item()
        diff_basin = 100.0 * (m_tgt_basin - m_src_basin) / (m_src_basin if m_src_basin != 0 else np.nan)
    except Exception:
        diff_basin = np.nan
    report = []
    report.append(f"Domain mass difference (tgt-src): {diff_domain:.6f}%")
    report.append(f"Basin mass difference (tgt-src): {diff_basin if np.isfinite(diff_basin) else float('nan'):.6f}%")
    return "\n".join(report)

# ---------------- Plots & CSV writers (CFTime-safe) ----------------

def plot_fraction_heatmap(frac: np.ndarray, lons: np.ndarray, lats: np.ndarray, out_path: str):
    Lon, Lat = np.meshgrid(lons, lats)
    plt.figure(figsize=(10,6))
    im = plt.pcolormesh(Lon, Lat, frac, cmap='viridis', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im, label='Overlap fraction'); plt.title('Basin fractional overlap (target grid)')
    plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_monthly_climatology_totals(ts_basin_mmday: xr.DataArray, out_path: str):
    # Convert daily basin-mean to monthly totals (mm/month), then mean across years for each month
    monthly_totals = ts_basin_mmday.resample(time='MS').sum()
    clim = monthly_totals.groupby('time.month').mean('time')
    months_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    month_nums = clim['month'].values.tolist()
    vals = [float(clim.sel(month=m).values) if m in month_nums else np.nan for m in range(1,13)]
    plt.figure(figsize=(8,4))
    plt.bar(months_labels, vals, color='tab:blue')
    plt.grid(axis='y', alpha=0.3)
    plt.ylabel('mm/month'); plt.title('Basin monthly climatology (mean of monthly totals)')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_elev_gradient(pr_mean: xr.DataArray, elev: np.ndarray, out_path: str):
    plt.figure(figsize=(8,5))
    plt.scatter(elev.flatten(), pr_mean.values.flatten(), s=6, alpha=0.6)
    plt.xlabel('Elevation (m)'); plt.ylabel('mm/day'); plt.title('Grid-cell mean precip vs elevation')
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def coord_ranges_ok(lon: np.ndarray, lat: np.ndarray) -> bool:
    lon_ok = np.all(np.isfinite(lon)) and ((np.nanmin(lon) >= -360) and (np.nanmax(lon) <= 360))
    lat_ok = np.all(np.isfinite(lat)) and ((np.nanmin(lat) >= -90) and (np.nanmax(lat) <= 90))
    return lon_ok and lat_ok

def bounds_monotonic_ok(bounds: np.ndarray) -> bool:
    b = np.asarray(bounds, dtype=float); diffs = np.diff(b)
    return np.all(np.isfinite(diffs)) and (np.all(diffs > 0) or np.all(diffs < 0))

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
    # Daily basin mean (mm/day), dates as strings
    df_daily = pd.DataFrame({'date': ts_basin['time'].dt.strftime('%Y-%m-%d').values, 'basin_mean_mmday': ts_basin.values})
    df_daily.to_csv(os.path.join(out_dir, f"{stem}_daily_basin_mean_mmday.csv"), index=False)
    # Monthly totals (mm/month)
    monthly_totals = ts_basin.resample(time='MS').sum()
    df_monthly = pd.DataFrame({'date': monthly_totals['time'].dt.strftime('%Y-%m').values, 'mm': monthly_totals.values})
    df_monthly.to_csv(os.path.join(out_dir, f"{stem}_monthly_basin_totals_mm.csv"), index=False)
    # Yearly totals (mm/year)
    yearly_totals = ts_basin.resample(time='YS').sum()
    df_yearly = pd.DataFrame({'year': yearly_totals['time'].dt.year.values, 'mm': yearly_totals.values})
    df_yearly.to_csv(os.path.join(out_dir, f"{stem}_yearly_basin_totals_mm.csv"), index=False)

# ---------------- Main ----------------

def main():
    if not XESMF_AVAILABLE:
        sys.exit(1)
    ensure_dir(OUTPUT_DIR); fig_dir = os.path.join(OUTPUT_DIR, 'figures'); ensure_dir(fig_dir)
    basin_gdf = gpd.read_file(BASIN_SHP)
    if basin_gdf.crs is None: basin_gdf = basin_gdf.set_crs(4326)
    else: basin_gdf = basin_gdf.to_crs(4326)
    basin_union_ll = unary_union(basin_gdf.geometry)
    lons, lats, lon_b, lat_b = build_grid_from_basin(basin_union_ll, TARGET_RES_DEG, buffer_deg=BBOX_BUFFER_DEG)
    ds_tgt = build_xesmf_grid(lons, lats, lon_b, lat_b)
    if not (coord_ranges_ok(ds_tgt['lon'].values, ds_tgt['lat'].values) and bounds_monotonic_ok(lon_b) and bounds_monotonic_ok(lat_b)):
        print("[ERROR] Target grid coordinates/bounds invalid."); sys.exit(1)
    frac_mat = basin_fraction_matrix(lons, lats, lon_b, lat_b, basin_union_ll)
    plot_fraction_heatmap(frac_mat, lons, lats, os.path.join(fig_dir, 'fraction_overlap_target.png'))
    terrain = dem_to_target_grid(DEM_PATH, lons, lats)
    print(f"[INFO] DEM stats after unit check & clamp: min={np.nanmin(terrain['elev']):.2f}, max={np.nanmax(terrain['elev']):.2f} m")

    nc_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.nc')))
    if len(nc_files) == 0:
        print(f"[ERROR] No NetCDF files found in {INPUT_DIR}"); sys.exit(1)
    for nc in nc_files:
        print(f"\n=== Processing: {os.path.basename(nc)} ===")
        ds_src = xr.open_dataset(nc)
        pr = load_pr_variable(ds_src)
        pr_mmday_src = to_mm_per_day(pr)
        if {'lat','lon'}.issubset(set(pr_mmday_src.dims)):
            lat_src = ds_src['lat'].values; lon_src = ds_src['lon'].values
        elif {'latitude','longitude'}.issubset(set(ds_src.coords)):
            lat_src = ds_src['latitude'].values; lon_src = ds_src['longitude'].values
        else:
            raise ValueError("Unsupported coordinate names; expecting lat/lon or latitude/longitude.")
        lon_b_src = infer_bounds(lon_src); lat_b_src = infer_bounds(lat_src)
        Lon_src, Lat_src = np.meshgrid(lon_src, lat_src)
        ds_src_grid = xr.Dataset({'lon': (('y','x'), Lon_src), 'lat': (('y','x'), Lat_src),
                                  'lon_b': (('y_b','x_b'), np.tile(lon_b_src, (len(lat_src)+1,1))),
                                  'lat_b': (('y_b','x_b'), np.tile(lat_b_src.reshape(-1,1), (1,len(lon_src)+1)))})
        pr_da = pr_mmday_src.transpose(...)
        rename_map = {}
        for d in pr_da.dims:
            dl = d.lower()
            if dl in ('lat','latitude'): rename_map[d] = 'y'
            if dl in ('lon','longitude'): rename_map[d] = 'x'
        if rename_map: pr_da = pr_da.rename(rename_map)
        regridder = make_regridder(ds_src_grid, ds_tgt)
        pr_regridded = regridder(pr_da)
        pr_regridded.name = 'pr_regrid'; pr_regridded.attrs['units'] = 'mm/day'; pr_regridded.attrs['long_name'] = 'Regridded precipitation (mm/day)'
        # Basin mean time series (mm/day)
        ts_basin_xr = area_weighted_basin_mean(pr_regridded, frac_mat)
        # Plot climatology of monthly totals (mm/month)
        plot_monthly_climatology_totals(ts_basin_xr, os.path.join(fig_dir, f"{os.path.basename(nc).replace('.nc','')}_clim_monthly_totals.png"))
        # Elevation-precip diagnostic
        pr_mean = pr_regridded.mean(dim='time')
        plot_elev_gradient(pr_mean, terrain['elev'], os.path.join(fig_dir, f"{os.path.basename(nc).replace('.nc','')}_elev_gradient.png"))
        # Orographic adjustment
        months = pr_regridded['time'].dt.month.values
        pr_orog = apply_orographic_adjustment(pr_regridded, terrain, months, frac_mat)
        # QA
        qa_text = mass_conservation_report(pr_da, pr_regridded, lon_b_src, lat_b_src, lon_b, lat_b, frac_mat, ds_src_grid, ds_tgt)
        # Basin-only mask for outputs
        mask = xr.DataArray((frac_mat > 0), dims=('y','x'))
        pr_regridded_basin = pr_regridded.where(mask)
        pr_orog_basin = pr_orog.where(mask)
        # Output dataset (EPSG:4326)
        ds_out = xr.Dataset({'pr_regrid': pr_regridded_basin, 'pr_orog': pr_orog_basin,
                             'basin_frac': (('y','x'), frac_mat), 'lon': (('y','x'), ds_tgt['lon'].values), 'lat': (('y','x'), ds_tgt['lat'].values)})
        ds_out.attrs['title'] = 'Basin-only regridded precipitation (mm/day) with orographic adjustment'
        ds_out.attrs['institution'] = 'Custom processing script v3.1 (CFTime-safe)'
        ds_out.attrs['source_file'] = os.path.basename(nc)
        ds_out.attrs['Conventions'] = 'CF-1.8'; ds_out.attrs['crs'] = 'EPSG:4326'
        ds_out.attrs['geospatial_lon_units'] = 'degrees_east'; ds_out.attrs['geospatial_lat_units'] = 'degrees_north'
        ds_out.attrs['regridding_method'] = 'conservative_2nd_fallback_conservative'
        ds_out.attrs['qa_mass_conservation'] = qa_text
        out_nc = os.path.join(OUTPUT_DIR, f"{os.path.basename(nc).replace('.nc','')}_basin_v3.nc")
        comp = dict(zlib=True, complevel=4); encoding = {v: comp for v in ['pr_regrid','pr_orog','basin_frac','lon','lat']}
        ds_out.to_netcdf(out_nc, encoding=encoding)
        print(f"[OK] Saved basin NetCDF: {out_nc}")
        # Diagnostics CSVs
        stem = os.path.basename(nc).replace('.nc','')
        write_basin_diagnostics(ts_basin_xr, OUTPUT_DIR, stem)
        print("[OK] Saved basin diagnostic CSVs (daily/monthly/yearly totals).")
        # Full-grid CSV for SWAT (basin cells only)
        out_csv_swat = os.path.join(OUTPUT_DIR, f"{stem}_daily_grid_for_SWAT.csv")
        write_fullgrid_csv_for_swat(pr_regridded, ds_tgt['lon'].values, ds_tgt['lat'].values, frac_mat, out_csv_swat)
        print(f"[OK] Saved SWAT full-grid CSV (basin cells only): {out_csv_swat}")
        # QA text
        qa_path = os.path.join(OUTPUT_DIR, f"{stem}_QA_v3.txt")
        with open(qa_path, 'w') as f:
            f.write("QA/QC Report (v3.1, CFTime-safe)\n")
            f.write("==================================\n\n")
            f.write(f"Input file: {nc}\n")
            f.write(f"Target grid: {len(lats)} lat x {len(lons)} lon (res {TARGET_RES_DEG}°)\n")
            f.write("\nMass conservation checks:\n"); f.write(qa_text + "\n")
            mins = pr_regridded.min(dim=('y','x')).values; maxs = pr_regridded.max(dim=('y','x')).values
            flat = np.where(np.abs(maxs - mins) <= 1e-12)[0]
            if flat.size > 0:
                f.write(f"\n[QC] {flat.size} flat day(s) detected where min≈max across grid. First 10 dates:\n")
                dates_str = pr_regridded['time'].dt.strftime('%Y-%m-%d').values
                for k in flat[:10]: f.write("  " + str(dates_str[k]) + "\n")
        print(f"[OK] QA saved: {qa_path}")
        print("\n✅ Script v3.1 complete for this file.")

if __name__ == '__main__':
    main()
