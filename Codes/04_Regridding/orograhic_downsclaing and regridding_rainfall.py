#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orographic Downscaling Pipeline for GCM Precipitation → ERA5-Land Grid (Cross-Platform)
--------------------------------------------------------------------------------------
- Auto-detects Windows vs WSL vs native Linux and adapts paths accordingly.
- Accepts Windows-style canonical paths and converts to /mnt/<drive>/... on WSL.
- Adds ESMPy/ESMF import compatibility (prefer `esmpy`, fallback to legacy `ESMF`).
- Keeps your original workflow intact (bounds, conservative regridding, DEM aggregation,
  orographic adjustment by lapse-rate or climatology ratio, optional QDM, diagnostics).

Dependencies (conda-forge recommended):
  conda install -c conda-forge xarray dask xesmf rioxarray rasterio scipy netcdf4 h5netcdf cftime esmpy

Notes on ESMPy/ESMF:
- Modern ESMPy uses `import esmpy` (module name: `esmpy`), while older pyESMF used `import ESMF`.
- xESMF>=0.7 prefers ESMPy and falls back to legacy ESMF; this script provides a small helper
  that tries `import esmpy as ESMF`, then `import ESMF`, and continues even if both are missing.

"""
import os
import re
import glob
import warnings
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from scipy import stats

# Optional libs (required for main features)
import xesmf as xe  # conservative regridding
import rioxarray    # DEM IO / reprojection

# ---------------------------------------------------------------------
# Platform helpers: Windows / WSL / Linux path adaptation
# ---------------------------------------------------------------------
from pathlib import Path

def is_wsl() -> bool:
    """Detect WSL by reading /proc/osrelease (works on WSL1/2)."""
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except Exception:
        return False

def win_to_wsl(p: str) -> str:
    """Convert 'C:\\Users\\...' -> '/mnt/c/Users/...' for WSL. Return p unchanged if not a drive path."""
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if not m:
        return p
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"

def adapt_path(p: Optional[str]) -> Optional[str]:
    """Return platform-appropriate path. If None/empty, return as-is."""
    if not p:
        return p
    # Expand env vars and ~
    p = os.path.expanduser(os.path.expandvars(p))
    # If native Windows, use as-is
    if os.name == "nt":
        return p
    # If WSL and a Windows-style drive path, convert
    if is_wsl():
        return win_to_wsl(p)
    # Native Linux: if given a Windows-style, convert; else return as-is
    return win_to_wsl(p)

# ---------------------------------------------------------------------
# ESMPy/ESMF import compatibility (informational)
# ---------------------------------------------------------------------

def init_esmpy():
    try:
        import esmpy as ESMF  # preferred modern package name
        print(f"[ESMPy] version={ESMF.__version__}")
        return ESMF
    except Exception as e1:
        try:
            import esmpy as ESMF  # legacy module name
            print(f"[pyESMF] version={ESMF.__version__}")
            return ESMF
        except Exception as e2:
            print("[INFO] ESMPy/ESMF not available. xESMF may still work if not importing ESMF directly.")
            return None

# ---------------------------------------------------------------------
# Helper functions (unchanged logic, lightly annotated)
# ---------------------------------------------------------------------

def build_bounds_from_centers(centers: np.ndarray) -> np.ndarray:
    """Construct 1D bounds [lower, upper] from 1D centers using midpoints with end extrapolation."""
    centers = np.asarray(centers)
    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return np.vstack([edges[:-1], edges[1:]]).T

def monthly_climatology(da: xr.DataArray) -> xr.DataArray:
    return da.groupby('time.month').mean('time')

# DEM aggregation & filling

def priority_flood_fill(elev: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
    """Minimal priority-flood depression filling (4-neighbourhood) in-memory."""
    from heapq import heappush, heappop
    arr = elev.copy()
    ny, nx = arr.shape
    if nodata is not None:
        mask = np.isclose(arr, nodata)
    else:
        mask = np.isnan(arr)
    visited = np.zeros_like(arr, dtype=bool)
    pq = []
    # Push boundary cells
    for i in range(ny):
        for j in (0, nx-1):
            if not mask[i, j]:
                heappush(pq, (arr[i, j], i, j))
                visited[i, j] = True
    for j in range(nx):
        for i in (0, ny-1):
            if not mask[i, j] and not visited[i, j]:
                heappush(pq, (arr[i, j], i, j))
                visited[i, j] = True
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
    while pq:
        z, y, x = heappop(pq)
        for dy, dx in nbrs:
            yy, xx = y+dy, x+dx
            if 0 <= yy < ny and 0 <= xx < nx and not visited[yy, xx] and not mask[yy, xx]:
                visited[yy, xx] = True
                if arr[yy, xx] < z:
                    arr[yy, xx] = z
                heappush(pq, (arr[yy, xx], yy, xx))
    return arr

def aggregate_dem_to_target(dem_path: str, tgt: xr.Dataset) -> xr.DataArray:
    """Aggregate high-res DEM (GeoTIFF) to target grid via xESMF 'conservative_normed'. Fill depressions in-memory."""
    dem_path = adapt_path(dem_path)
    dem = rioxarray.open_rasterio(dem_path, masked=True)
    if 'band' in dem.dims:
        dem = dem.isel(band=0)
    try:
        dem = dem.rio.reproject("EPSG:4326")
    except Exception:
        pass
    dem = dem.rename({dem.rio.x_dim: 'lon', dem.rio.y_dim: 'lat'})
    if dem.lat[0] > dem.lat[-1]:
        dem = dem.sortby('lat')
    dem_ds = dem.to_dataset(name='elev')
    if 'lat_bnds' not in dem_ds.coords:
        dem_ds['lat_bnds'] = xr.DataArray(build_bounds_from_centers(dem_ds['lat'].values), dims=('lat','bnds'))
    if 'lon_bnds' not in dem_ds.coords:
        dem_ds['lon_bnds'] = xr.DataArray(build_bounds_from_centers(dem_ds['lon'].values), dims=('lon','bnds'))
    tgt_b = tgt.copy()
    if 'lat_bnds' not in tgt_b:
        tgt_b['lat_bnds'] = xr.DataArray(build_bounds_from_centers(tgt_b['lat'].values), dims=('lat','bnds'))
    if 'lon_bnds' not in tgt_b:
        tgt_b['lon_bnds'] = xr.DataArray(build_bounds_from_centers(tgt_b['lon'].values), dims=('lon','bnds'))
    regridder = xe.Regridder(dem_ds, tgt_b, method='conservative_normed')
    elev_tgt = regridder(dem_ds['elev'])
    vals = elev_tgt.values
    if np.ma.isMaskedArray(vals):
        filled = priority_flood_fill(vals.filled(np.nan))
        elev_tgt = xr.DataArray(np.where(np.isnan(vals), np.nan, filled),
                                dims=elev_tgt.dims, coords=elev_tgt.coords, attrs=elev_tgt.attrs)
    else:
        filled = priority_flood_fill(vals)
        elev_tgt = xr.DataArray(filled, dims=elev_tgt.dims, coords=elev_tgt.coords, attrs=elev_tgt.attrs)
    elev_tgt.name = 'elevation'
    elev_tgt.attrs['units'] = 'm'
    elev_tgt.attrs['description'] = 'DEM aggregated to ERA5 grid (area-weighted mean) with in-memory depression filling.'
    return elev_tgt

# Orographic adjustments

def apply_climatology_ratio(pr_regridded: xr.DataArray, ref_clim: xr.DataArray, renorm: bool = True) -> xr.DataArray:
    model_clim = monthly_climatology(pr_regridded)
    ratio = xr.where(model_clim > 0, ref_clim / model_clim, 1.0)
    ratio_t = ratio.sel(month=pr_regridded['time.month'])
    pr_adj = pr_regridded * ratio_t
    if renorm:
        w = np.cos(np.deg2rad(pr_adj['lat']))
        w = w / w.mean()
        for m in range(1, 13):
            sel = pr_adj['time.month'] == m
            s_before = (pr_regridded.where(sel) * w).mean(('lat', 'lon'))
            s_after  = (pr_adj.where(sel) * w).mean(('lat', 'lon'))
            factor = xr.where(s_after > 0, s_before/s_after, 1.0).mean()
            pr_adj = pr_adj.where(~sel, pr_adj * factor)
    return pr_adj

def fit_monthly_plr(pr_clim: xr.DataArray, elev: xr.DataArray) -> xr.DataArray:
    """Fit monthly log-linear precip lapse rate: log(P) = a + b*z_km. Returns b per month (1/km)."""
    gammas = []
    zkm = elev / 1000.0
    for m in range(1, 13):
        Pc = pr_clim.sel(month=m)
        mask = np.isfinite(Pc) & (Pc > 0) & np.isfinite(zkm)
        x = zkm.where(mask).values.ravel()
        y = np.log(Pc.where(mask).values.ravel())
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) < 50:
            gammas.append(0.0)
        else:
            slope, intercept, r, p, se = stats.linregress(x, y)
            gammas.append(slope)
    gamma = xr.DataArray(np.array(gammas), dims=['month'], coords={'month': np.arange(1, 13)})
    gamma.attrs['units'] = '1/km'
    gamma.attrs['description'] = 'Monthly d ln P / dz (log-precip lapse rate)'
    return gamma

def apply_lapse_rate(pr_regridded: xr.DataArray, elev: xr.DataArray, renorm: bool = True) -> xr.DataArray:
    model_clim = monthly_climatology(pr_regridded)
    gamma = fit_monthly_plr(model_clim, elev)
    z = elev
    zbar = z.mean()
    dz = (z - zbar) / 1000.0
    scale = xr.DataArray(np.exp(gamma.values[:, None, None] * dz.values[None, :, :]),
                         dims=('month', 'lat', 'lon'),
                         coords={'month': gamma['month'], 'lat': dz['lat'], 'lon': dz['lon']})
    scale_t = scale.sel(month=pr_regridded['time.month'])
    pr_adj = pr_regridded * scale_t
    if renorm:
        w = np.cos(np.deg2rad(pr_adj['lat']))
        w = w / w.mean()
        for m in range(1, 13):
            sel = pr_adj['time.month'] == m
            s_before = (pr_regridded.where(sel) * w).mean(('lat', 'lon'))
            s_after  = (pr_adj.where(sel) * w).mean(('lat', 'lon'))
            factor = xr.where(s_after > 0, s_before/s_after, 1.0).mean()
            pr_adj = pr_adj.where(~sel, pr_adj * factor)
    return pr_adj

# QDM bias correction (multiplicative)

def ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    x = np.sort(arr)
    n = len(x)
    p = np.linspace(1, n, n) / (n + 1.0)
    return x, p

def qdm_precip(model_hist: xr.DataArray, model_fut: xr.DataArray, obs_hist: xr.DataArray) -> xr.DataArray:
    out = []
    for m in range(1, 13):
        sel_hist = model_hist['time.month'] == m
        sel_fut  = model_fut['time.month']  == m
        sel_obs  = obs_hist['time.month']   == m
        mh = model_hist.where(sel_hist, drop=True)
        mf = model_fut.where(sel_fut, drop=True)
        oh = obs_hist.where(sel_obs, drop=True)
        def _qdm_1d(mh1d, mf1d, oh1d):
            mh1d = mh1d.values.ravel()
            mf1d = mf1d.values.ravel()
            oh1d = oh1d.values.ravel()
            mh1d = mh1d[np.isfinite(mh1d) & (mh1d >= 0)]
            mf1d = mf1d[np.isfinite(mf1d) & (mf1d >= 0)]
            oh1d = oh1d[np.isfinite(oh1d) & (oh1d >= 0)]
            if len(mh1d) < 50 or len(mf1d) < 50 or len(oh1d) < 50:
                return np.full_like(mf1d, np.nan, dtype=float)
            x_mh, p_mh = ecdf(mh1d)
            x_mf, p_mf = ecdf(mf1d)
            x_oh, p_oh = ecdf(oh1d)
            p_grid = np.linspace(0.01, 0.99, 99)
            Qmh = np.interp(p_grid, p_mh, x_mh)
            Qoh = np.interp(p_grid, p_oh, x_oh)
            s = np.where(Qmh > 0, Qoh/Qmh, 1.0)
            p_f = np.interp(mf1d, x_mf, p_mf, left=0.01, right=0.99)
            sf = np.interp(p_f, p_grid, s)
            return mf1d * sf
        adj = xr.apply_ufunc(_qdm_1d, mh, mf, oh,
                             input_core_dims=[['time'], ['time'], ['time']],
                             output_core_dims=[['time']],
                             vectorize=True,
                             dask='parallelized',
                             output_dtypes=[float])
        adj = adj.assign_coords({'time': mf['time']})
        out.append(adj)
    return xr.concat(out, dim='time').sortby('time')

# Validation helper

def precip_elev_corr(pr: xr.DataArray, elev: xr.DataArray) -> float:
    Pc = monthly_climatology(pr).mean('month')
    mask = np.isfinite(Pc) & np.isfinite(elev)
    x = elev.where(mask).values.ravel()
    y = Pc.where(mask).values.ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 50:
        return np.nan
    r, p = stats.pearsonr(x, y)
    return float(r)

# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline(
    gcm_dir: str,
    era5_template: str,
    dem_path: str,
    output_dir: str,
    ref_clim_path: Optional[str] = None,
    obs_hist_path: Optional[str] = None,
    orog_method: str = 'lapse',  # 'lapse' or 'ratio'
    varname: str = 'pr',
    units: str = 'mm/day'
):
    # Adapt paths cross-platform
    gcm_dir = adapt_path(gcm_dir)
    era5_template = adapt_path(era5_template)
    dem_path = adapt_path(dem_path)
    output_dir = adapt_path(output_dir)
    ref_clim_path = adapt_path(ref_clim_path)
    obs_hist_path = adapt_path(obs_hist_path)

    # Path check
    print("\n[PATH CHECK]")
    print("GCM_DIR:", gcm_dir, os.path.isdir(gcm_dir))
    print("ERA5_TEMPLATE:", era5_template, os.path.exists(era5_template))
    print("DEM_PATH:", dem_path, os.path.exists(dem_path))
    print("OUTPUT_DIR:", output_dir)
    if orog_method == 'ratio':
        print("REF_CLIM_PATH:", ref_clim_path, bool(ref_clim_path and os.path.exists(ref_clim_path)))
    if obs_hist_path:
        print("OBS_HIST_PATH:", obs_hist_path, os.path.exists(obs_hist_path))

    os.makedirs(output_dir, exist_ok=True)

    # Optional: initialize ESMPy/ESMF (informational)
    _ = init_esmpy()

    # 1) Target grid (ERA5 template)
    print("\nLoading ERA5 target grid template:", era5_template)
    ds_tgt = xr.open_dataset(era5_template)
    if 'latitude' in ds_tgt.coords:
        ds_tgt = ds_tgt.rename({'latitude': 'lat'})
    if 'longitude' in ds_tgt.coords:
        ds_tgt = ds_tgt.rename({'longitude': 'lon'})
    if 'lat_bnds' not in ds_tgt and 'lat' in ds_tgt:
        ds_tgt['lat_bnds'] = xr.DataArray(build_bounds_from_centers(ds_tgt['lat'].values), dims=('lat','bnds'))
    if 'lon_bnds' not in ds_tgt and 'lon' in ds_tgt:
        ds_tgt['lon_bnds'] = xr.DataArray(build_bounds_from_centers(ds_tgt['lon'].values), dims=('lon','bnds'))
    ds_tgt = ds_tgt[['lat','lon','lat_bnds','lon_bnds']]

    # 2) DEM aggregation + in-memory fill
    print("Aggregating DEM to target grid (area-weighted) and filling depressions in-memory…")
    elev_tgt = aggregate_dem_to_target(dem_path, ds_tgt)

    # 3) Reference climatology (if ratio)
    if orog_method == 'ratio':
        if ref_clim_path is None:
            raise ValueError("ref_clim_path must be provided for climatology ratio method.")
        print("Loading reference climatology:", ref_clim_path)
        ref = xr.open_dataset(ref_clim_path)
        if 'precip' in ref.data_vars:
            ref_clim = ref['precip']
        else:
            ref_clim = list(ref.data_vars.values())[0]
        if not (np.array_equal(ref_clim['lat'], ds_tgt['lat']) and np.array_equal(ref_clim['lon'], ds_tgt['lon'])):
            print("Regridding reference climatology to target grid (bilinear)…")
            regridder_ref = xe.Regridder(ref_clim, ds_tgt, 'bilinear')
            ref_clim = xr.concat([regridder_ref(ref_clim.sel(month=m)) for m in range(1, 13)], dim='month')
            ref_clim['month'] = np.arange(1, 13)
    else:
        ref_clim = None

    # 4) Process each GCM file
    files = sorted(glob.glob(os.path.join(gcm_dir, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {gcm_dir}")

    for f in files:
        print("\nProcessing:", os.path.basename(f))
        ds = xr.open_dataset(f)
        var = varname if varname in ds.data_vars else next((k for k in ['pr','precip','precipitation'] if k in ds.data_vars), None)
        if var is None:
            raise KeyError("Precipitation variable not found. Set VAR_NAME to match your files.")
        da = ds[var]
        renames = {}
        if 'lat' not in da.coords and 'latitude' in da.coords:
            renames['latitude'] = 'lat'
        if 'lon' not in da.coords and 'longitude' in da.coords:
            renames['longitude'] = 'lon'
        if renames:
            da = da.rename(renames)
        # Curvilinear → pre-step bilinear to rectilinear ERA5 grid
        if da['lat'].ndim != 1 or da['lon'].ndim != 1:
            print("Curvilinear grid detected → bilinear pre-step to rectilinear ERA5 grid…")
            regridder_pre = xe.Regridder(da, ds_tgt, method='bilinear')
            da = regridder_pre(da)
        # Add bounds if missing
        if 'lat_bnds' not in da.coords:
            da.coords['lat_bnds'] = xr.DataArray(build_bounds_from_centers(da['lat'].values), dims=('lat','bnds'))
        if 'lon_bnds' not in da.coords:
            da.coords['lon_bnds'] = xr.DataArray(build_bounds_from_centers(da['lon'].values), dims=('lon','bnds'))
        # Units to mm/day
        if 'units' in da.attrs and da.attrs['units'] in ['kg m-2 s-1','kg/m^2/s','kg m-2 s^-1','kg m-2 s**-1']:
            da = da * 86400.0
            da.attrs['units'] = 'mm/day'
        elif units == 'mm/day' and da.attrs.get('units','').lower() in ['mm/day','mm d-1']:
            pass
        else:
            warnings.warn("Unknown precip units; proceeding assuming mm/day.")

        # Conservative regridding to target (FRACAREA normalization)
        print("Regridding to ERA5 grid (conservative_normed)…")
        regridder = xe.Regridder(da, ds_tgt, method='conservative_normed')
        pr_tgt = regridder(da)
        pr_tgt.name = 'pr'
        pr_tgt.attrs['units'] = 'mm/day'
        pr_tgt.attrs['long_name'] = 'Precipitation (regridded)'

        # Orographic adjustment
        if orog_method == 'ratio':
            print("Applying climatology ratio + monthly renormalization…")
            pr_adj = apply_climatology_ratio(pr_tgt, ref_clim, renorm=True)
        else:
            print("Applying lapse-rate scaling + monthly renormalization…")
            pr_adj = apply_lapse_rate(pr_tgt, elev_tgt, renorm=True)
        pr_adj.name = 'pr_orog'
        pr_adj.attrs['long_name'] = 'Precipitation after orographic adjustment'

        # Optional QDM bias correction
        if obs_hist_path is not None:
            print("Applying QDM bias correction (multiplicative)…")
            obs = xr.open_dataset(obs_hist_path)
            vobs = next((obs[k] for k in ['pr','precip','precipitation'] if k in obs.data_vars), list(obs.data_vars.values())[0])
            if not (np.array_equal(vobs['lat'], ds_tgt['lat']) and np.array_equal(vobs['lon'], ds_tgt['lon'])):
                regridder_obs = xe.Regridder(vobs, ds_tgt, method='bilinear')
                vobs = regridder_obs(vobs)
            # Demo mapping using same series for both hist/future; replace with real split in production
            pr_bc = qdm_precip(pr_adj, pr_adj, vobs)
            pr_bc.name = 'pr_downscaled'
            final = pr_bc
        else:
            final = pr_adj.rename('pr_downscaled')

        # Validation diagnostic
        try:
            r_pe = precip_elev_corr(final, elev_tgt)
            print(f"Diag: precip–elevation correlation (annual mean): r = {r_pe:.3f}")
        except Exception:
            pass

        # Save output
        base = os.path.basename(f)
        out_name = base.replace('.nc', '_downscaled.nc')
        out_path = os.path.join(output_dir, out_name)
        print("Writing:", out_path)
        ds_out = xr.Dataset({'pr_downscaled': final, 'elevation': elev_tgt})
        ds_out.to_netcdf(out_path)

    print("\nAll files processed.")

# ==============================
# USER CONFIG: EDIT THESE PATHS
# ==============================
if __name__ == '__main__':
    # You may keep Windows-style paths here; on WSL they will be converted automatically.
    GCM_DIR        = r'mnt/c/users/Diwakar Adhikari/Downloads/Model Selection/GCM/precipitation'         # folder with your GCM .nc files
    ERA5_TEMPLATE  = r'mnt/c/users/Diwakar Adhikari/Documents/ERA5_Land/precipitation/1984H1p.nc' # any NetCDF on ERA5-Land grid
    DEM_PATH       = r'mnt/c/Users/Diwakar Adhikari/OneDrive/OneDrive - Clean up Nepal/Desktop/DEM/MRB_JULY.tif'      # ASTER DEM (GeoTIFF)
    OUTPUT_DIR     = r'mnt/c/users/Diwakar Adhikari/Model Selection/GCM/precipitation/outputs'               # outputs written here

    # --- Method selection ---
    OROG_METHOD    = 'lapse'   # 'lapse' (default) or 'ratio'
    # If using method='ratio', set monthly reference climatology (dims: month,lat,lon).
    REF_CLIM_PATH  = None  # e.g., r'C:\path\to\monthly_reference_climatology.nc'
    # Optional: observed historical daily precipitation (for QDM bias correction)
    OBS_HIST_PATH  = None  # e.g., r'C:\path\to\obs_daily_precip_on_era5grid.nc'

    # Name of precipitation variable in your GCM files (defaults to 'pr'; change if needed)
    VAR_NAME       = 'pr'
    UNITS          = 'mm/day'

    # Run
    run_pipeline(
        gcm_dir=GCM_DIR,
        era5_template=ERA5_TEMPLATE,
        dem_path=DEM_PATH,
        output_dir=OUTPUT_DIR,
        ref_clim_path=REF_CLIM_PATH,
        obs_hist_path=OBS_HIST_PATH,
        orog_method=OROG_METHOD,
        varname=VAR_NAME,
        units=UNITS,
    )
