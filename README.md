# End‑to‑End Workflow: ERA5‑Land & GCM → Hydrological Forcings

This repository documents our full pipeline to build hydrological forcings from **ERA5‑Land** and **CMIP6 GCMs**, select models using the **ΔP/ΔT method**, apply **orographic precipitation regridding** (conservative & bilinear), and perform **QDM bias correction**.

## Contents

1.  ERA5‑Land: Download & Extraction
2.  GCMs: Download & Extraction
3.  Model Selection: ΔP / ΔT
4.  Regridding & Orographic Adjustment
    *   (a) First‑order conservative + DEM
    *   (b) Bilinear regridding (precipitation)
5.  Bias Correction: Quantile Delta Mapping (QDM)
6.  Outputs
7.  Reproducibility Notes

***

## 1) ERA5‑Land: Download & Extraction

**Goal:** Produce **daily, basin‑representative** meteorological series suitable for hydrological modelling.

**What we did**

*   Downloaded **ERA5‑Land** hourly NetCDF for study extent (precip, Tmax, Tmin, RH if needed).
*   **Converted UTC → local time (NST, UTC+05:45)** *before* aggregation.
*   **Area‑weighted averaging** over the basin/sub‑basins using polygon overlap fractions.
*   Aggregated **hourly → daily**:
    *   Precipitation: daily **sum**
    *   Tmax: daily **max**
    *   Tmin: daily **min**
*   QC for gaps, duplicates, and unit consistency (mm/day for precip).

**Inputs:** ERA5‑Land hourly `.nc`, basin shapefile(s)  
**Outputs:** Daily basin time series (NetCDF/CSV), local‑time stamped

***

## 2) GCM: Download & Extraction

**Goal:** Prepare historical and future GCM series aligned to the analysis domain.

**What we did**

*   Downloaded **CMIP6** daily variables (historical + SSP2‑4.5 / SSP5‑8.5 as needed).
*   Extracted target region bounding box.
*   Standardized variable names/units (e.g., `pr` to **mm/day**).
*   Kept historical (e.g., **1985–2014**) and future (**2015–2100**) **separate**.

**Inputs:** CMIP6 NetCDFs (per model/scenario)  
**Outputs:** Region‑cropped, unit‑standardized GCM daily `.nc`

***

## 3) Model Selection — ΔP / ΔT

**Goal:** Choose a subset of GCMs that represent regional hydro‑climate change envelopes.

**What we did**

*   Computed **ΔP** and **ΔT** between a baseline (e.g., **1985–2014**) and projection (e.g., **2041–2070**), typically at **annual** and **seasonal (e.g., JJAS)** scales.
*   Ranked/filtered models based on spread and realism (e.g., within observed variability bounds).
*   Selected **Top‑N** models covering the ΔP/ΔT space (wet/dry & warm/hot quadrants).

**Inputs:** GCM daily series (hist & fut), ERA5‑Land diagnostics for context  
**Outputs:** List of selected GCMs (+ small table/figure of ΔP vs ΔT)

***

## 4) Regridding & Orographic Adjustment

**Goal:** Map GCM precipitation to the **ERA5‑Land grid** and account for **topography**.

### 4(a) First‑order conservative + DEM (preferred for precipitation)

*   Regridded GCM `pr` to the **ERA5‑Land rectilinear grid** using **first‑order conservative (area‑weighted)**.
*   Aggregated an external **DEM** onto the same grid (area‑weighted) to obtain grid‑mean elevation.
*   Applied **orographic precipitation adjustment** (e.g., lapse‑rate scaling or climatology ratio) using the DEM‑derived elevation field.
*   Preserved mass through **conservative\_normed** handling where applicable.

**Inputs:** GCM `pr` (daily), ERA5‑Land template grid, DEM (GeoTIFF)  
**Outputs:** GCM precipitation on ERA5‑Land grid with orographic adjustment

### 4(b) Bilinear regridding (temperature)

- Uses **bilinear interpolation** to regrid temperature (tasmin / tasmax) from the native GCM grid to a basin-defined target grid.
- Chosen because temperature is an **intensive variable**, where smooth spatial interpolation is appropriate.
- Well-suited as a **pre-processing step** for converting curvilinear or coarse-resolution model grids to a regular lat–lon grid.
- Produces smoother fields compared to nearest-neighbor methods.

⚠️ Note:
- Bilinear interpolation **does not conserve totals**.
- Should **not** be used for accumulated or extensive variables (e.g., precipitation, runoff).

**Inputs:** Same as 4(a)  
**Outputs:** GCM precipitation on ERA5‑Land grid (bilinear variant)

***

## 5) Bias Correction: Quantile Delta Mapping (QDM)

**Goal:** Correct GCM biases while preserving change signal.

**What we did**

*   Performed **QDM (multiplicative for precipitation; additive or multiplicative chosen per variable)** on a **monthly** conditioning, using:
    *   **Observed baseline:** ERA5‑Land (basin series or ERA5‑grid series)
    *   **Model baseline:** GCM (same period)
    *   **Model future:** GCM (projection period)
*   Produced **bias‑corrected future series** consistent with observed distribution and projected changes.

**Inputs:**

*   Observed (ERA5‑Land) daily series (baseline)
*   GCM daily series (baseline & future, regridded to ERA5‑Land grid)

**Outputs:** Bias‑corrected GCM daily series (per model/scenario)

***

## Outputs

*   **/outputs/era5\_land/**
    *   `basin_daily_*.nc` (or `.csv`): daily local‑time series from ERA5‑Land
*   **/outputs/gcm\_prepped/**
    *   `MODEL_SCENARIO_region_daily.nc`: region‑cropped, unit‑standardized
*   **/outputs/regridded/**
    *   `MODEL_SCENARIO_pr_era5grid_conservative.nc`
    *   `MODEL_SCENARIO_pr_era5grid_bilinear.nc`
*   **/outputs/orographic/**
    *   `MODEL_SCENARIO_pr_orog.nc` (DEM‑informed adjustment)
*   **/outputs/qdm/**
    *   `MODEL_SCENARIO_pr_qdm.nc` (final hydrological forcing candidate)
*   **/docs/**
    *   `deltaP_deltaT_selection.pdf/png` (scatter, table of selected models)

***

## Reproducibility Notes

*   **Time handling:** All daily aggregations are based on **local time (UTC+05:45)**.
*   **Area handling:** Basin values use **area‑weighted averaging** (polygon overlap).
*   **Units:** Precipitation is **mm/day**; check metadata for others.
*   **Separation of periods:** Keep **historical** and **future** files separate to avoid overwriting.
*   **Method toggles:** For precipitation, prefer **conservative**; use **bilinear** intentionally (e.g., pre‑step from curvilinear grids).

***

### Quick Summary (one‑liner)

> ERA5‑Land hourly data were converted to local time and area‑averaged to the basin; CMIP6 GCMs were regridded to the ERA5‑Land grid with orographic adjustment, then bias‑corrected using QDM, and models were selected via ΔP/ΔT coverage.
