Characterizing Hydrologic Response & Projecting Climate Change Impacts

A reproducible pipeline for bias‑correcting climate data, orographic downscaling, and hydrologic characterization to assess climate change impacts on water resources.


✨ What this repo does

-Regridding & Orographic Downscaling for Precipitation

-Uses xESMF for first‑order conservative regridding (mass‑conserving) of GCM precipitation to a high‑resolution target grid (e.g., ERA5‑Land scale).
Applies orographic adjustment based on DEM‑derived terrain metrics (elevation, slope, aspect, TPI) and seasonal weights.
Outputs:
pr_regrid → regridded precipitation (mm/day)
pr_orog → orographically adjusted precipitation (mm/day)

-Bias Correction (QDM) for Climate Variables
  -Precipitation → ratio QDM (multiplicative, ≥0, trace handling).
  -Temperature (Tmin/Tmax/Tavg) → additive QDM (delta mapping, preserves sign).

Multi‑model support for CMIP6 historical and future scenarios (SSP2‑4.5, SSP5‑8.5).
