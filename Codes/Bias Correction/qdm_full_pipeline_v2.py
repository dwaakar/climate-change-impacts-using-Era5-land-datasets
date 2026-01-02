
# qdm_tmin_single_model.py
# -----------------------------------------------------------
# QDM (additive) bias correction for Tmin using a single model CSV.
# Standardize -> Train (1985–2014) -> Correct future (2015+) -> Validate (2015–2024)
# Outputs: corrected CSV, validation metrics, annual & monthly plots.
# Author: Diwakar Adhikari & M365 Copilot
# Date: 2026-01-02

import os, re, json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========================= USER CONFIG =========================
# Set the paths for your observed and model CSVs.
ROOT = r"youtpath"
OBS_FILENAME = "observed_daily_tmin_per-grid_std.csv"   # observed tmin
MODEL_FILENAME = "MIROC-ES2L_ssp585_std.csv"           # one model CSV (historical+future if available)

# Canonicalization and correction options
CANON_DECIMALS = 3         # grid label rounding: lat{:.3f}_lon{:.3f}
MONTHWISE = True           # recommended True
TRAIN_START = "1985-01-01"
TRAIN_END   = "2014-12-31"
FUTURE_START = "2015-01-01"  # future segment to correct

# Output folders (created under ROOT)
OUT_STD_DIR  = os.path.join(ROOT, "standardized_tmin_single")
OUT_QDM_DIR  = os.path.join(ROOT, "QDM_out_tmin_single")
PLOTS_DIR    = os.path.join(OUT_QDM_DIR, "plots")
LOG_PATH     = os.path.join(OUT_QDM_DIR, "run_log_tmin.txt")
os.makedirs(OUT_STD_DIR, exist_ok=True)
os.makedirs(OUT_QDM_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ========================= HELPERS =========================
def log(msg: str):
    print(msg)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + "\n")

LABEL_RE = re.compile(r"^\s*lat\s*([\-\+]?\d*\.?\d+)\s*_\s*lon\s*([\-\+]?\d*\.?\d+)\s*$", re.IGNORECASE)

def canonical_label(lat: float, lon: float, decimals: int) -> str:
    return f"lat{lat:.{decimals}f}_lon{lon:.{decimals}f}"

def normalize_columns(df: pd.DataFrame, decimals: int):
    mapping = {}
    for col in df.columns:
        if col.lower() == 'date':
            mapping[col] = 'date'
            continue
        m = LABEL_RE.match(col)
        if m:
            lat = float(m.group(1)); lon = float(m.group(2))
            mapping[col] = canonical_label(lat, lon, decimals)
        else:
            mapping[col] = col
    return df.rename(columns=mapping)

def normalize_dates(df: pd.DataFrame):
    if 'date' not in df.columns:
        raise ValueError("CSV missing 'date' column.")
    def to_iso(x):
        if pd.isna(x): return None
        s = str(x).strip()
        fmts = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m/%d/%y", "%Y/%m/%d"]
        for f in fmts:
            try:
                dt = datetime.strptime(s, f); return dt.strftime('%Y-%m-%d')
            except Exception: continue
        dt = pd.to_datetime(s, errors='coerce')
        if pd.isna(dt): return None
        return dt.strftime('%Y-%m-%d')
    df['date'] = df['date'].apply(to_iso)
    df = df.dropna(subset=['date']).copy()
    return df

def ranks_to_probs(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    return (ranks - 1) / (len(x) - 1) if len(x) > 1 else np.zeros_like(ranks)

def empirical_quantile(values: np.ndarray, p: float) -> float:
    if values.size == 0: return np.nan
    arr = np.sort(values)
    p = min(max(p, 0.0), 1.0)
    idx = p * (len(arr) - 1)
    i0 = int(np.floor(idx)); i1 = int(np.ceil(idx))
    if i0 == i1: return float(arr[i0])
    w = idx - i0
    return float((1 - w) * arr[i0] + w * arr[i1])

# ========================= CORE =========================
def standardize_pair(obs_path: str, model_path: str):
    # Observed
    log(f"[INFO] Loading observed: {obs_path}")
    obs_df = pd.read_csv(obs_path)
    obs_df = normalize_dates(obs_df)
    obs_df = normalize_columns(obs_df, CANON_DECIMALS)
    obs_grids = [c for c in obs_df.columns if c.lower() != 'date']
    std_obs_path = os.path.join(OUT_STD_DIR, os.path.splitext(os.path.basename(obs_path))[0] + "_std.csv")
    obs_df[['date'] + obs_grids].to_csv(std_obs_path, index=False)

    # Model
    log(f"[INFO] Loading model: {model_path}")
    mdf = pd.read_csv(model_path)
    mdf = normalize_dates(mdf)
    mdf = normalize_columns(mdf, CANON_DECIMALS)
    # Align model columns to observed grid list order
    aligned_cols = [g for g in obs_grids if g in mdf.columns]
    std_model_path = os.path.join(OUT_STD_DIR, os.path.splitext(os.path.basename(model_path))[0] + "_std.csv")
    mdf[['date'] + aligned_cols].to_csv(std_model_path, index=False)

    # Report
    report = {
        'root': ROOT,
        'observed_std': os.path.basename(std_obs_path),
        'model_std': os.path.basename(std_model_path),
        'canonical_decimals': CANON_DECIMALS,
        'std_grid_cols_count': len(aligned_cols),
    }
    with open(os.path.join(OUT_STD_DIR, "standardization_report_tmin_single.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    log(f"[OK] Standardization saved. Grids aligned: {len(aligned_cols)}")
    return std_obs_path, std_model_path, obs_df, obs_grids

def qdm_additive_month_segment(obs_m: np.ndarray, hist_m: np.ndarray, fut_m: np.ndarray) -> np.ndarray:
    """
    Additive QDM for temperature:
      corr = Q_obs(p) + (Q_fut(p) - Q_hist(p)), with p from fut_m ranks.
    """
    if fut_m.size == 0: return np.array([])
    obs_sorted  = np.sort(obs_m)
    hist_sorted = np.sort(hist_m)
    fut_sorted  = np.sort(fut_m)
    p_fut = ranks_to_probs(fut_m)
    out = np.zeros_like(fut_m, dtype=float)
    for i, p in enumerate(p_fut):
        q_obs  = empirical_quantile(obs_sorted,  p)
        q_hist = empirical_quantile(hist_sorted, p)
        q_fut  = empirical_quantile(fut_sorted,  p)
        out[i] = q_obs + (q_fut - q_hist)
    return out

def run_qdm_tmin_single(model_std_csv: str, obs_df: pd.DataFrame, obs_grids: list):
    # Load standardized model
    mdf = pd.read_csv(model_std_csv)
    obs_df = obs_df.copy()
    obs_df['date'] = pd.to_datetime(obs_df['date'])
    mdf['date'] = pd.to_datetime(mdf['date'])

    # Split windows
    obs_train  = obs_df[(obs_df['date'] >= TRAIN_START) & (obs_df['date'] <= TRAIN_END)]
    hist_train = mdf[(mdf['date'] >= TRAIN_START) & (mdf['date'] <= TRAIN_END)]
    fut_df     = mdf[(mdf['date'] >= FUTURE_START)]

    # Historical QM to obs (optional output; good check)
    hist_corr = hist_train[['date']].copy()
    if MONTHWISE:
        months_hist = hist_train['date'].dt.month.values
        months_obs  = obs_train['date'].dt.month.values
        for g in obs_grids:
            if g not in hist_train.columns: continue
            hc_vals = hist_train[g].values.astype(float)
            ob_vals = obs_train[g].values.astype(float)
            out_vals = np.empty_like(hc_vals, dtype=float)
            for m in range(1, 13):
                idx_h = np.where(months_hist == m)[0]
                idx_o = np.where(months_obs  == m)[0]
                if idx_h.size == 0 or idx_o.size == 0: continue
                p_hist = ranks_to_probs(hc_vals[idx_h])
                obs_sorted = np.sort(ob_vals[idx_o])
                out_vals[idx_h] = np.array([empirical_quantile(obs_sorted, p) for p in p_hist])
            hist_corr[g] = out_vals
    else:
        for g in obs_grids:
            if g not in hist_train.columns: continue
            hc_vals = hist_train[g].values.astype(float)
            ob_vals = obs_train[g].values.astype(float)
            p_hist = ranks_to_probs(hc_vals)
            obs_sorted = np.sort(ob_vals)
            hist_corr[g] = np.array([empirical_quantile(obs_sorted, p) for p in p_hist])

    hist_out_path = os.path.join(OUT_QDM_DIR, "tmin_historical_QDM.csv")
    if len(hist_corr) > 1:
        hist_corr.to_csv(hist_out_path, index=False)
        log(f"[OK] Historical corrected (QM) written: {hist_out_path}")

    # Future correction (additive QDM)
    ssp_corr = fut_df[['date']].copy()
    if MONTHWISE:
        months_fut  = fut_df['date'].dt.month.values
        months_hist = hist_train['date'].dt.month.values
        months_obs  = obs_train['date'].dt.month.values
        for g in obs_grids:
            if g not in fut_df.columns: continue
            fut_vals_all = fut_df[g].values.astype(float)
            out_vals_all = np.empty_like(fut_vals_all, dtype=float)
            for m in range(1, 13):
                idx_f = np.where(months_fut  == m)[0]
                idx_h = np.where(months_hist == m)[0]
                idx_o = np.where(months_obs  == m)[0]
                if idx_f.size == 0:
                    continue
                if idx_h.size == 0 or idx_o.size == 0:
                    # No training in that month -> pass-through
                    out_vals_all[idx_f] = fut_vals_all[idx_f]
                    continue
                ob_m   = obs_train[g].values[idx_o]
                hist_m = hist_train[g].values[idx_h]
                fut_m  = fut_vals_all[idx_f]
                corr_m = qdm_additive_month_segment(ob_m, hist_m, fut_m)
                out_vals_all[idx_f] = corr_m
            ssp_corr[g] = out_vals_all
    else:
        for g in obs_grids:
            if g not in fut_df.columns: continue
            ob_m   = obs_train[g].values.astype(float)
            hist_m = hist_train[g].values.astype(float)
            fut_m  = fut_df[g].values.astype(float)
            corr_m = qdm_additive_month_segment(ob_m, hist_m, fut_m)
            ssp_corr[g] = corr_m

    out_path = os.path.join(OUT_QDM_DIR, "tmin_single_model_QDM.csv")
    ssp_corr.to_csv(out_path, index=False)
    log(f"[OK] QDM (tmin) correction written: {out_path}")
    return out_path, ssp_corr, fut_df

def validate_and_plots_tmin(obs_df: pd.DataFrame, ssp_corr: pd.DataFrame, fut_df: pd.DataFrame, obs_grids: list):
    # Validation (2015–2024): basin mean (simple unweighted mean across grids)
    def basin_mean_daily(df: pd.DataFrame, grids: list) -> pd.Series:
        dates = pd.to_datetime(df['date'])
        used = [g for g in grids if g in df.columns]
        if len(used) == 0:
            return pd.Series([np.nan]*len(df), index=dates)
        vals = df[used].values
        mean_vals = vals.mean(axis=1)
        return pd.Series(mean_vals, index=dates)

    obs_val = obs_df[(pd.to_datetime(obs_df['date']) >= '2015-01-01') &
                     (pd.to_datetime(obs_df['date']) <= '2024-12-31')].copy()
    obs_mean = basin_mean_daily(obs_val, obs_grids)
    raw_val  = fut_df[(fut_df['date'] >= '2015-01-01') & (fut_df['date'] <= '2024-12-31')].copy()
    qdm_val  = ssp_corr[(ssp_corr['date'] >= '2015-01-01') & (ssp_corr['date'] <= '2024-12-31')].copy()

    def mae_rmse(obs_series, other_df):
        if len(other_df) == 0: return None, None, 0
        other_mean = basin_mean_daily(other_df, obs_grids)
        common = obs_series.index.intersection(other_mean.index)
        if len(common) == 0: return None, None, 0
        e = obs_series.loc[common].values - other_mean.loc[common].values
        mae = float(np.nanmean(np.abs(e)))
        rmse = float(np.sqrt(np.nanmean(e**2)))
        return mae, rmse, int(len(common))

    raw_mae, raw_rmse, raw_n = mae_rmse(obs_mean, raw_val)
    qdm_mae, qdm_rmse, qdm_n = mae_rmse(obs_mean, qdm_val)
    metrics = {
        'var': 'tmin', 'unit': '°C',
        'raw_mae': raw_mae, 'raw_rmse': raw_rmse, 'overlap_days_raw': raw_n,
        'qdm_mae': qdm_mae, 'qdm_rmse': qdm_rmse, 'overlap_days_qdm': qdm_n
    }
    val_path = os.path.join(OUT_QDM_DIR, 'validation_metrics_tmin_single.csv')
    pd.DataFrame([metrics]).to_csv(val_path, index=False)
    log(f"[OK] Validation metrics saved: {val_path}")

    # Annual summary plot (mean for temperature)
    def annual_mean(series: pd.Series, dates: pd.Series) -> float:
        df = pd.DataFrame({'date': dates.values, 'x': series.values})
        df['date'] = pd.to_datetime(df['date'])
        annual = df.set_index('date')['x'].resample('YS').mean()
        return float(annual.mean()) if len(annual) > 0 else float('nan')

    obs_annual = annual_mean(obs_mean, obs_mean.index.to_series())
    raw_annual = annual_mean(basin_mean_daily(raw_val, obs_grids), raw_val['date']) if len(raw_val) > 0 else np.nan
    qdm_annual = annual_mean(basin_mean_daily(qdm_val, obs_grids), qdm_val['date']) if len(qdm_val) > 0 else np.nan

    plt.figure(figsize=(6,4))
    plt.bar(['Observed','Raw','QDM'], [obs_annual, raw_annual, qdm_annual], color=['#2E86AB','#E67E22','#27AE60'])
    plt.ylabel('Mean annual Tmin (°C), 2015–2024')
    plt.title('Tmin: Basin-wide annual mean (2015–2024)')
    plt.tight_layout()
    p1 = os.path.join(PLOTS_DIR, "tmin_annual_mean_bars.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    log(f"[OK] Plot saved: {p1}")

    # Monthly climatology plot (mean for temperature)
    def monthly_climatology(series: pd.Series, date_series: pd.Series) -> np.ndarray:
        df = pd.DataFrame({'date': date_series.values, 'x': series.values})
        df['date'] = pd.to_datetime(df['date'])
        monthly_vals = df.set_index('date')['x'].resample('MS').mean()
        clim = monthly_vals.groupby(monthly_vals.index.month).mean()
        return np.array([clim.get(m, np.nan) for m in range(1,13)])

    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    obs_clim = monthly_climatology(obs_mean, obs_mean.index.to_series())
    raw_clim = monthly_climatology(basin_mean_daily(raw_val, obs_grids), raw_val['date']) if len(raw_val) > 0 else np.full(12, np.nan)
    qdm_clim = monthly_climatology(basin_mean_daily(qdm_val, obs_grids), qdm_val['date']) if len(qdm_val) > 0 else np.full(12, np.nan)

    plt.figure(figsize=(8,4))
    plt.plot(months, obs_clim, label='Observed', color='#2E86AB', marker='o')
    if len(raw_val) > 0: plt.plot(months, raw_clim, label='Raw', color='#E67E22', marker='o')
    if len(qdm_val) > 0: plt.plot(months, qdm_clim, label='QDM', color='#27AE60', marker='o')
    plt.grid(alpha=0.3); plt.ylabel('°C'); plt.title('Tmin: Basin monthly climatology (2015–2024)')
    plt.legend(); plt.tight_layout()
    p2 = os.path.join(PLOTS_DIR, "tmin_monthly_climatology.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    log(f"[OK] Plot saved: {p2}")

# ========================= MAIN =========================
def main():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("QDM (tmin) single-model run log\n")
    obs_path = os.path.join(ROOT, OBS_FILENAME)
    model_path = os.path.join(ROOT, MODEL_FILENAME)
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f"Observed file not found: {obs_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    std_obs_path, std_model_path, obs_df, obs_grids = standardize_pair(obs_path, model_path)
    out_path, ssp_corr, fut_df = run_qdm_tmin_single(std_model_path, obs_df, obs_grids)
    validate_and_plots_tmin(obs_df, ssp_corr, fut_df, obs_grids)
    log(f"[DONE] Tmin QDM pipeline complete. Output CSV: {out_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")

        raise
