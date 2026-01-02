# qdm_full_pipeline_v6_qdmviz_gif_per_model.py
# --------------------------------------------------------------
# Standardization -> QDM -> Validation & Plots
# Save a QDM-internals GIF **per model** (no realtime window, no web server)
# Panels in GIF frames: ECDFs, ratio mapping, histogram, hydrograph for monitored grid/month segments
# Author: Diwakar Adhikari & M365 Copilot
# Date: 2025-11-10
# --------------------------------------------------------------
import os
import re
import sys
import glob
import json
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Non-interactive backend; only save images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime

# --------------------------- USER CONFIG ---------------------------
ROOT = r"C:\\Users\\Diwakar Adhikari\\Downloads\\Model Selection\\GCM\\precipitation\\regrid_v4\\QDM\\standardized"
OBS_FILENAME = "observed_era5land_daily_grid_NST_BULK.csv"
CANON_DECIMALS = 3
TRACE_MM = 0.1
RATIO_MAX = 5.0
RATIO_MIN = 0.0
MONTHWISE = True
WEIGHTS_CSV: Optional[str] = None

# --- GIF recording options (per model) ---
RECORD_GIF = True                  # set False to skip GIFs
GIF_FPS = 6                        # frames per second
GIF_MAX_FRAMES = 1500              # safety cap per model
GIF_KEEP_FRAMES = False            # keep the intermediate PNG frames
MONITOR_GRID: Optional[str] = None # e.g., 'lat27.800_lon84.600'; None -> auto first grid
MONITOR_EVERY_N_GRIDS = 25         # record updates every N grids to limit frames
MAX_POINTS_TS = 300                # downsample length in hydrograph panel

# Output folders
OUT_STD_DIR = os.path.join(ROOT, "standardized")
OUT_QDM_DIR = os.path.join(ROOT, "QDM_out")
PLOTS_DIR = os.path.join(OUT_QDM_DIR, "plots")
REPORT_STD = os.path.join(OUT_STD_DIR, "standardization_report.json")
REPORT_QDM = os.path.join(OUT_QDM_DIR, "qdm_report.json")
LOG_PATH = os.path.join(OUT_QDM_DIR, "run_log.txt")

os.makedirs(OUT_STD_DIR, exist_ok=True)
os.makedirs(OUT_QDM_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------- Logging ---------------------------
def log(msg: str):
    print(msg)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + "\n")

# ---------------------- GIF Recorder for QDM Viz ----------------------
class QDMVizRecorder:
    """Record QDM internals frames and write a single GIF per model.
    Panels per frame:
      (A) ECDFs: Obs vs Hist (1985–2014) vs Future (month)
      (B) Ratio mapping R(p) and corrected quantiles Q_obs*R
      (C) Histogram: month-wise Raw vs Corrected
      (D) Hydrograph: Raw vs Corrected (downsampled)
    """
    def __init__(self, model_name: str, fps: int, max_frames: int, keep_frames: bool):
        self.model = model_name
        self.fps = max(1, int(fps))
        self.max_frames = int(max_frames)
        self.keep_frames = bool(keep_frames)
        self.frames_dir = os.path.join(PLOTS_DIR, f"_qdm_frames_{self.model}")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.frame_idx = 0
        # prepare figure
        self.fig = plt.figure(figsize=(11, 7))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[1,1], width_ratios=[1,1])
        self.axes = {
            'ecdf': self.fig.add_subplot(gs[0,0]),
            'ratio': self.fig.add_subplot(gs[0,1]),
            'hist':  self.fig.add_subplot(gs[1,0]),
            'ts':    self.fig.add_subplot(gs[1,1])
        }
        self._init_styles()

    def _init_styles(self):
        for key in ['ecdf','ratio','hist','ts']:
            ax = self.axes[key]
            ax.clear()
            ax.grid(alpha=0.3)
        self.axes['ecdf'].set_title('ECDFs (Obs vs Hist vs Fut)')
        self.axes['ecdf'].set_xlabel('Precipitation (mm/day)')
        self.axes['ecdf'].set_ylabel('Probability')
        self.axes['ratio'].set_title('QDM mapping: R(p) & corrected quantiles')
        self.axes['ratio'].set_xlabel('Probability p')
        self.axes['ratio'].set_ylabel('Value')
        self.axes['hist'].set_title('Distribution in month segment')
        self.axes['hist'].set_xlabel('Precipitation (mm/day)')
        self.axes['hist'].set_ylabel('Frequency')
        self.axes['ts'].set_title('Month segment: Raw vs Corrected (downsampled)')
        self.axes['ts'].set_xlabel('Index (segment)')
        self.axes['ts'].set_ylabel('mm/day')
        self.fig.tight_layout()

    @staticmethod
    def _ecdf(x: np.ndarray):
        x = np.sort(x)
        n = len(x)
        if n == 0:
            return np.array([]), np.array([])
        p = np.linspace(0.0, 1.0, n)
        return x, p

    def update(self, ob_m: np.ndarray, hist_m: np.ndarray, fut_m: np.ndarray,
               corr_m: np.ndarray, month: int, grid_label: str):
        if self.frame_idx >= self.max_frames:
            return
        # ECDFs
        ax = self.axes['ecdf']; ax.clear(); ax.grid(alpha=0.3)
        xo, po = self._ecdf(ob_m); xh, ph = self._ecdf(hist_m); xf, pf = self._ecdf(fut_m)
        ax.plot(xo, po, label='Obs (1985–2014)', color='#2E86AB')
        ax.plot(xh, ph, label='Hist GCM (1985–2014)', color='#E67E22')
        ax.plot(xf, pf, label=f'Future raw (M{month})', color='#7D3C98')
        ax.legend(loc='best'); ax.set_title(f'ECDFs model={self.model} grid={grid_label} month={month}')
        # Ratio mapping
        axr = self.axes['ratio']; axr.clear(); axr.grid(alpha=0.3)
        n = max(len(fut_m), len(hist_m), len(ob_m)); px = np.linspace(0, 1, max(2,n))
        def q(arr, p):
            arr = np.sort(arr)
            if len(arr) == 0:
                return np.full_like(p, np.nan, dtype=float)
            idx = p * (len(arr) - 1)
            i0 = np.floor(idx).astype(int); i1 = np.ceil(idx).astype(int); w = idx - i0
            return (1-w)*arr[i0] + w*arr[i1]
        q_obs = q(ob_m, px); q_hist = q(hist_m, px); q_fut = q(fut_m, px)
        q_hist_safe = np.maximum(q_hist, TRACE_MM)
        ratio = np.where(q_hist_safe>0, q_fut/q_hist_safe, 0.0)
        ratio = np.clip(ratio, RATIO_MIN, RATIO_MAX)
        q_corr = np.maximum(0.0, q_obs * ratio)
        axr.plot(px, ratio, label='R(p)', color='#34495E')
        axr.plot(px, q_obs, label='Q_obs(p)', color='#2E86AB', alpha=0.7)
        axr.plot(px, q_corr, label='Q_corr(p)', color='#27AE60')
        axr.legend(loc='best'); axr.set_title(f'QDM mapping model={self.model} grid={grid_label} month={month}')
        # Histogram
        axh = self.axes['hist']; axh.clear(); axh.grid(alpha=0.3)
        bins = max(10, int(np.sqrt(max(len(fut_m),1))))
        if len(fut_m)>0: axh.hist(fut_m, bins=bins, alpha=0.6, label='Raw fut', color='#7D3C98')
        if len(corr_m)>0: axh.hist(corr_m, bins=bins, alpha=0.6, label='QDM corr', color='#27AE60')
        axh.legend(loc='best'); axh.set_title(f'Distribution model={self.model} grid={grid_label} month={month}')
        # Hydrograph
        axt = self.axes['ts']; axt.clear(); axt.grid(alpha=0.3)
        nseg = len(fut_m); idx = np.arange(nseg)
        if nseg>0:
            if nseg>MAX_POINTS_TS:
                step = max(1, nseg//MAX_POINTS_TS)
                idx = idx[::step]
                fut_plot = fut_m[::step]; corr_plot = corr_m[::step]
            else:
                fut_plot = fut_m; corr_plot = corr_m
            axt.plot(idx, fut_plot, label='Raw fut', color='#7D3C98')
            axt.plot(idx, corr_plot, label='QDM corr', color='#27AE60')
            axt.legend(loc='best')
        axt.set_title(f'Hydrograph model={self.model} grid={grid_label} month={month}')
        # Save frame
        self.fig.tight_layout()
        frame_path = os.path.join(self.frames_dir, f'frame_{self.frame_idx:06d}.png')
        self.fig.savefig(frame_path, dpi=120)
        self.frame_idx += 1

    def finalize_gif(self) -> Optional[str]:
        """Write GIF in PLOTS_DIR as <model>_qdm_internal.gif and cleanup frames if needed."""
        if self.frame_idx == 0:
            return None
        out_gif = os.path.join(PLOTS_DIR, f"{self.model}_qdm_internal.gif")
        try:
            from PIL import Image
            frames = []
            for i in range(self.frame_idx):
                fp = os.path.join(self.frames_dir, f'frame_{i:06d}.png')
                if os.path.isfile(fp):
                    frames.append(Image.open(fp).convert('P'))
            if frames:
                duration = int(1000/max(1,self.fps))  # ms per frame
                frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration, loop=0)
                log(f"[GIF] Saved: {out_gif}")
        except Exception as e:
            log(f"[WARN] GIF creation failed for {self.model}: {e}")
            out_gif = None
        # cleanup frames
        if not self.keep_frames:
            try:
                for i in range(self.frame_idx):
                    fp = os.path.join(self.frames_dir, f'frame_{i:06d}.png')
                    if os.path.isfile(fp): os.remove(fp)
                os.rmdir(self.frames_dir)
            except Exception:
                pass
        return out_gif

# --------------------------- Helpers ---------------------------
LABEL_RE = re.compile(r"^\s*lat\s*([\-\+]?\d*\.?\d+)\s*_\s*lon\s*([\-\+]?\d*\.?\d+)\s*$", re.IGNORECASE)

def parse_label(label: str):
    m = LABEL_RE.match(label)
    if not m:
        return None
    try:
        lat = float(m.group(1)); lon = float(m.group(2))
        return lat, lon
    except Exception:
        return None

def canonical_label(lat: float, lon: float, decimals: int = CANON_DECIMALS) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"lat{fmt.format(round(lat, decimals))}_lon{fmt.format(round(lon, decimals))}"

def normalize_columns(df: pd.DataFrame):
    mapping = {}
    canon_cols = []
    for col in df.columns:
        if col.lower() == 'date':
            continue
        parsed = parse_label(col)
        if parsed is None:
            mapping[col] = col; canon_cols.append(col)
        else:
            clabel = canonical_label(parsed[0], parsed[1], CANON_DECIMALS)
            mapping[col] = clabel; canon_cols.append(clabel)
    return mapping, canon_cols

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

def discover_models(root: str, obs_name: str) -> Dict[str, Dict[str, str]]:
    files = glob.glob(os.path.join(root, '*.csv'))
    models: Dict[str, Dict[str, str]] = {}
    for fp in files:
        base = os.path.basename(fp)
        if base == obs_name: continue
        name_noext = os.path.splitext(base)[0]
        parts = name_noext.split('_')
        if len(parts) < 2: continue
        suffix = parts[-1].lower(); model = '_'.join(parts[:-1])
        if suffix not in ('historical', 'ssp245', 'ssp585'): continue
        models.setdefault(model, {})[suffix] = fp
    return models

def empirical_quantile(values: np.ndarray, p: float) -> float:
    if values.size == 0: return np.nan
    arr = np.sort(values); p = min(max(p, 0.0), 1.0)
    idx = p * (len(arr) - 1)
    i0 = int(np.floor(idx)); i1 = int(np.ceil(idx))
    if i0 == i1: return float(arr[i0])
    w = idx - i0
    return float((1 - w) * arr[i0] + w * arr[i1])

def ranks_to_probs(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    return (ranks - 1) / (len(x) - 1) if len(x) > 1 else np.zeros_like(ranks)

# --------------------------- QDM Core ---------------------------
def qdm_ratio_correct_future(obs_m: np.ndarray, hist_m: np.ndarray, fut_m: np.ndarray,
                             trace_mm: float = TRACE_MM, ratio_max: float = RATIO_MAX) -> np.ndarray:
    if fut_m.size == 0: return np.array([])
    obs_sorted = np.sort(obs_m); hist_sorted = np.sort(hist_m); fut_sorted = np.sort(fut_m)
    p_fut = ranks_to_probs(fut_m)
    out = np.zeros_like(fut_m, dtype=float)
    for i, p in enumerate(p_fut):
        q_obs = empirical_quantile(obs_sorted, p)
        q_hist = empirical_quantile(hist_sorted, p)
        q_fut = empirical_quantile(fut_sorted, p)
        if q_hist < trace_mm:
            ratio = min(ratio_max, (q_fut / max(trace_mm, q_hist)))
        else:
            ratio = q_fut / q_hist
        ratio = max(RATIO_MIN, ratio)
        out[i] = max(0.0, q_obs * ratio)
    return out

def qm_hist_to_obs(obs_m: np.ndarray, hist_m: np.ndarray, hist_vals_m: np.ndarray) -> np.ndarray:
    if hist_vals_m.size == 0: return np.array([])
    obs_sorted = np.sort(obs_m)
    p_hist = ranks_to_probs(hist_vals_m)
    out = np.zeros_like(hist_vals_m, dtype=float)
    for i, p in enumerate(p_hist):
        q_obs = empirical_quantile(obs_sorted, p)
        out[i] = max(0.0, q_obs)
    return out

# ------------------------ Standardization ------------------------
def standardize_all() -> Tuple[str, pd.DataFrame, List[str], Dict[str, Dict[str, str]], Dict]:
    obs_path = os.path.join(ROOT, OBS_FILENAME)
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f"Observed file not found: {obs_path}")
    log(f"[INFO] Loading observed: {obs_path}")
    obs_df = pd.read_csv(obs_path)
    obs_df = normalize_dates(obs_df)
    obs_map, _ = normalize_columns(obs_df)
    obs_df = obs_df.rename(columns=obs_map)
    obs_grids = [c for c in obs_df.columns if c.lower() != 'date']
    obs_grid_set = set(obs_grids)
    std_obs_path = os.path.join(OUT_STD_DIR, os.path.splitext(os.path.basename(obs_path))[0] + "_std.csv")
    obs_df[['date'] + obs_grids].to_csv(std_obs_path, index=False)

    models = discover_models(ROOT, OBS_FILENAME)
    log(f"[INFO] Discovered {len(models)} model(s) with historical/SSP files.")
    report = {
        'root': ROOT,
        'observed_std': os.path.basename(std_obs_path),
        'canonical_decimals': CANON_DECIMALS,
        'files': []
    }

    for model, scen_map in models.items():
        for scen, path in scen_map.items():
            df = pd.read_csv(path)
            orig_cols = list(df.columns)
            df = normalize_dates(df)
            mapping, _ = normalize_columns(df)
            df = df.rename(columns=mapping)
            grid_cols = [c for c in df.columns if c.lower() != 'date']
            intersect = sorted(list(obs_grid_set.intersection(grid_cols)))
            missing_in_model = sorted(list(obs_grid_set.difference(grid_cols)))
            extra_in_model = sorted(list(set(grid_cols).difference(obs_grid_set)))
            aligned_cols = [g for g in obs_grids if g in intersect]
            std_df = df[['date'] + aligned_cols].copy()
            base = os.path.splitext(os.path.basename(path))[0]
            std_path = os.path.join(OUT_STD_DIR, base + "_std.csv")
            std_df.to_csv(std_path, index=False)
            report['files'].append({
                'file': os.path.basename(path),
                'model': model,
                'scenario': scen,
                'original_cols_count': len(orig_cols),
                'original_grid_cols_count': sum(1 for c in orig_cols if c.lower() != 'date'),
                'std_grid_cols_count': len(aligned_cols),
                'dropped_grids_count': len(missing_in_model),
                'extra_grids_count': len(extra_in_model),
            })
            log(f"[STD] {model} {scen}: std_grids={len(aligned_cols)} dropped={len(missing_in_model)} extra={len(extra_in_model)}")
    with open(REPORT_STD, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    log(f"[OK] Standardization report: {REPORT_STD}")
    return std_obs_path, obs_df, obs_grids, models, report

# --------------------- Weights & Basin mean ---------------------
def load_weights(obs_grids: List[str]) -> Optional[np.ndarray]:
    if WEIGHTS_CSV is None:
        return None
    try:
        wdf = pd.read_csv(WEIGHTS_CSV)
        wmap = dict(zip(wdf.iloc[:,0].astype(str), wdf.iloc[:,1].astype(float)))
        weights = np.array([wmap.get(g, np.nan) for g in obs_grids], dtype=float)
        if np.isnan(weights).any():
            log("[WARN] Some weights missing; falling back to equal weights.")
            return None
        s = np.sum(weights)
        if s <= 0:
            log("[WARN] Non-positive weights sum; falling back to equal weights.")
            return None
        return weights / s
    except Exception as e:
        log(f"[WARN] Failed to load weights: {e}; using equal weights.")
        return None

def basin_mean_daily(df: pd.DataFrame, grids: List[str], weights: Optional[np.ndarray]) -> pd.Series:
    if 'date' not in df.columns:
        raise ValueError("DataFrame missing 'date' column for basin_mean_daily().")
    dates = pd.to_datetime(df['date'])
    used_grids = [g for g in grids if g in df.columns]
    if len(used_grids) == 0:
        return pd.Series([float('nan')]*len(df), index=dates)
    w = None
    if weights is not None:
        try:
            idxs = [grids.index(g) for g in used_grids]
            w = weights[idxs]; s = float(w.sum()); w = (w / s) if s > 0 else None
        except Exception:
            w = None
    vals = df[used_grids].values
    mean_vals = vals.mean(axis=1) if w is None else (vals * w.reshape(1, -1)).sum(axis=1)
    return pd.Series(mean_vals, index=dates)

# ----------------------- QDM & Validation -----------------------
def run_qdm_for_model(model: str, scen_map: Dict[str, str], obs_df: pd.DataFrame, obs_grids: List[str], weights: Optional[np.ndarray], recorder: Optional[QDMVizRecorder]):
    def load_std(suffix: str) -> Optional[pd.DataFrame]:
        if suffix not in scen_map:
            return None
        base = os.path.splitext(os.path.basename(scen_map[suffix]))[0]
        std_path = os.path.join(OUT_STD_DIR, base + "_std.csv")
        if not os.path.isfile(std_path):
            return None
        return pd.read_csv(std_path)

    hist_df = load_std('historical')
    ssp245_df = load_std('ssp245')
    ssp585_df = load_std('ssp585')
    if hist_df is None:
        return {'model': model, 'status': 'skipped (no historical)', 'metrics': []}

    # Dates
    obs_df = obs_df.copy(); obs_df['date'] = pd.to_datetime(obs_df['date'])
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    if ssp245_df is not None: ssp245_df['date'] = pd.to_datetime(ssp245_df['date'])
    if ssp585_df is not None: ssp585_df['date'] = pd.to_datetime(ssp585_df['date'])

    # Training window
    obs_train = obs_df[(obs_df['date'] >= '1985-01-01') & (obs_df['date'] <= '2014-12-31')]
    hist_train = hist_df[(hist_df['date'] >= '1985-01-01') & (hist_df['date'] <= '2014-12-31')]

    # Historical corrected
    hist_corr = obs_train[['date']].copy()

    # Monitoring grid selection
    monitor_grid = MONITOR_GRID if (MONITOR_GRID and MONITOR_GRID in obs_grids) else (obs_grids[0] if len(obs_grids)>0 else None)

    # Monthwise or single distribution
    if MONTHWISE:
        months_hist = hist_train['date'].dt.month.values
        months_obs  = obs_train['date'].dt.month.values
        # Historical QM
        for g in obs_grids:
            hc_vals = hist_train[g].values.astype(float)
            ob_vals = obs_train[g].values.astype(float)
            out_vals = np.empty_like(hc_vals, dtype=float)
            for m in range(1, 13):
                idx = np.where(months_hist == m)[0]
                idxo = np.where(months_obs == m)[0]
                if idx.size == 0 or idxo.size == 0: continue
                out_vals[idx] = qm_hist_to_obs(ob_vals[idxo], hc_vals[idx], hc_vals[idx])
            hist_corr[g] = out_vals
    else:
        for g in obs_grids:
            hc_vals = hist_train[g].values.astype(float)
            ob_vals = obs_train[g].values.astype(float)
            hist_corr[g] = qm_hist_to_obs(ob_vals, hc_vals, hc_vals)

    # Write historical corrected
    hist_out_path = os.path.join(OUT_QDM_DIR, f"{model}_historical_QDM.csv")
    hist_corr.to_csv(hist_out_path, index=False)

    # Future corrections; record frames for monitored grid
    def correct_future(ssp_df: Optional[pd.DataFrame], tag: str) -> Optional[pd.DataFrame]:
        if ssp_df is None: return None
        ssp_corr = ssp_df[['date']].copy()
        if MONTHWISE:
            months_fut = ssp_df['date'].dt.month.values
            months_hist = hist_train['date'].dt.month.values
            months_obs  = obs_train['date'].dt.month.values
            for gi, g in enumerate(obs_grids):
                fut_vals_all = ssp_df[g].values.astype(float)
                out_vals_all = np.empty_like(fut_vals_all, dtype=float)
                for m in range(1, 13):
                    idx_f = np.where(months_fut == m)[0]
                    idx_h = np.where(months_hist == m)[0]
                    idx_o = np.where(months_obs == m)[0]
                    if idx_f.size == 0: continue
                    if idx_h.size == 0 or idx_o.size == 0:
                        out_vals_all[idx_f] = fut_vals_all[idx_f]
                        continue
                    ob_m = obs_train[g].values[idx_o]
                    hist_m = hist_train[g].values[idx_h]
                    fut_m = fut_vals_all[idx_f]
                    corr_m = qdm_ratio_correct_future(ob_m, hist_m, fut_m, TRACE_MM, RATIO_MAX)
                    out_vals_all[idx_f] = corr_m
                    # record only for monitored grid and stride
                    if RECORD_GIF and recorder and (g == monitor_grid) and (gi % MONITOR_EVERY_N_GRIDS == 0):
                        recorder.update(ob_m, hist_m, fut_m, corr_m, month=m, grid_label=g)
                ssp_corr[g] = out_vals_all
        else:
            for gi, g in enumerate(obs_grids):
                ob_m = obs_train[g].values.astype(float)
                hist_m = hist_train[g].values.astype(float)
                fut_m = ssp_df[g].values.astype(float)
                p_fut = ranks_to_probs(fut_m)
                obs_sorted = np.sort(ob_m); hist_sorted = np.sort(hist_m); fut_sorted = np.sort(fut_m)
                out_vals = np.zeros_like(fut_m, dtype=float)
                for i, p in enumerate(p_fut):
                    q_obs = empirical_quantile(obs_sorted, p)
                    q_hist = empirical_quantile(hist_sorted, p)
                    q_fut = empirical_quantile(fut_sorted, p)
                    if q_hist < TRACE_MM:
                        ratio = min(RATIO_MAX, q_fut / max(TRACE_MM, q_hist))
                    else:
                        ratio = q_fut / q_hist
                    ratio = max(RATIO_MIN, ratio)
                    out_vals[i] = max(0.0, q_obs * ratio)
                ssp_corr[g] = out_vals
                if RECORD_GIF and recorder and (g == monitor_grid) and (gi % MONITOR_EVERY_N_GRIDS == 0):
                    recorder.update(ob_m, hist_m, fut_m, out_vals, month=0, grid_label=g)
        out_path = os.path.join(OUT_QDM_DIR, f"{model}_{tag}_QDM.csv")
        ssp_corr.to_csv(out_path, index=False)
        return ssp_corr

    ssp245_corr = correct_future(ssp245_df, 'ssp245')
    ssp585_corr = correct_future(ssp585_df, 'ssp585')

    # Validation 2015–2024
    obs_val = obs_df[(obs_df['date'] >= '2015-01-01') & (obs_df['date'] <= '2024-12-31')].copy()
    obs_val_mean = basin_mean_daily(obs_val, obs_grids, weights)

    def compute_metrics(tag: str):
        def subset(df):
            if df is None: return None
            tmp = df.copy(); tmp['date'] = pd.to_datetime(tmp['date'])
            return tmp[(tmp['date'] >= '2015-01-01') & (tmp['date'] <= '2024-12-31')]
        raw_df  = subset(ssp245_df if tag=='ssp245' else ssp585_df if tag=='ssp585' else None)
        corr_df = subset(ssp245_corr if tag=='ssp245' else ssp585_corr if tag=='ssp585' else None)
        out = {'scenario': tag, 'raw_mae_mmday': None, 'raw_rmse_mmday': None, 'qdm_mae_mmday': None, 'qdm_rmse_mmday': None, 'overlap_days_raw': 0, 'overlap_days_qdm': 0}
        obs_series = obs_val_mean.copy(); obs_series.index = pd.to_datetime(obs_series.index)
        if raw_df is not None and len(raw_df)>0:
            raw_mean = basin_mean_daily(raw_df, obs_grids, weights)
            common = obs_series.index.intersection(raw_mean.index)
            if len(common)>0:
                e = obs_series.loc[common].values - raw_mean.loc[common].values
                out['raw_mae_mmday'] = float(np.nanmean(np.abs(e)))
                out['raw_rmse_mmday'] = float(np.sqrt(np.nanmean(e**2)))
                out['overlap_days_raw'] = int(len(common))
        if corr_df is not None and len(corr_df)>0:
            corr_mean = basin_mean_daily(corr_df, obs_grids, weights)
            common = obs_series.index.intersection(corr_mean.index)
            if len(common)>0:
                e = obs_series.loc[common].values - corr_mean.loc[common].values
                out['qdm_mae_mmday'] = float(np.nanmean(np.abs(e)))
                out['qdm_rmse_mmday'] = float(np.sqrt(np.nanmean(e**2)))
                out['overlap_days_qdm'] = int(len(common))
        # Summary plots
        if (raw_df is not None and len(raw_df)>0) or (corr_df is not None and len(corr_df)>0):
            def annual_mean(series: pd.Series, dates: pd.Series) -> float:
                df = pd.DataFrame({'date': dates.values, 'x': series.values}); df['date'] = pd.to_datetime(df['date'])
                annual = df.set_index('date')['x'].resample('YS').sum(); return float(annual.mean()) if len(annual)>0 else float('nan')
            obs_annual = annual_mean(obs_series, obs_series.index.to_series())
            raw_annual = annual_mean(basin_mean_daily(raw_df, obs_grids, weights), raw_df['date']) if raw_df is not None and len(raw_df)>0 else np.nan
            qdm_annual = annual_mean(basin_mean_daily(corr_df, obs_grids, weights), corr_df['date']) if corr_df is not None and len(corr_df)>0 else np.nan
            plt.figure(figsize=(6,4))
            plt.bar(['Observed','Raw','QDM'], [obs_annual, raw_annual, qdm_annual], color=['#2E86AB','#E67E22','#27AE60'])
            plt.ylabel('Mean annual total (mm/year, 2015–2024)'); plt.title(f'{model} {tag}: Basin-wide annual mean')
            plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{model}_{tag}_annual_mean_bars.png"), dpi=200); plt.close()
            def monthly_climatology(series: pd.Series, date_series: pd.Series) -> np.ndarray:
                df = pd.DataFrame({'date': date_series.values, 'x': series.values}); df['date'] = pd.to_datetime(df['date'])
                monthly_totals = df.set_index('date')['x'].resample('MS').sum(); clim = monthly_totals.groupby(monthly_totals.index.month).mean()
                return np.array([clim.get(m, np.nan) for m in range(1,13)])
            months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            obs_clim = monthly_climatology(obs_series, obs_series.index.to_series())
            raw_clim = monthly_climatology(basin_mean_daily(raw_df, obs_grids, weights), raw_df['date']) if raw_df is not None and len(raw_df)>0 else np.full(12, np.nan)
            qdm_clim = monthly_climatology(basin_mean_daily(corr_df, obs_grids, weights), corr_df['date']) if corr_df is not None and len(corr_df)>0 else np.full(12, np.nan)
            plt.figure(figsize=(8,4))
            plt.plot(months, obs_clim, label='Observed', color='#2E86AB', marker='o')
            if raw_df is not None and len(raw_df)>0: plt.plot(months, raw_clim, label='Raw', color='#E67E22', marker='o')
            if corr_df is not None and len(corr_df)>0: plt.plot(months, qdm_clim, label='QDM', color='#27AE60', marker='o')
            plt.grid(alpha=0.3); plt.ylabel('Climatology (mm/month)'); plt.title(f'{model} {tag}: Basin monthly climatology (2015–2024)')
            plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{model}_{tag}_monthly_climatology.png"), dpi=200); plt.close()
        return out

    val_report = {'model': model, 'metrics': []}
    val_report['metrics'].append(compute_metrics('ssp245'))
    val_report['metrics'].append(compute_metrics('ssp585'))
    return val_report

# ----------------------------- Main -----------------------------
def main():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("QDM full pipeline run log\n")
    log("[INFO] Starting QDM full pipeline (standardize -> QDM -> validate)")

    std_obs_path, obs_df, obs_grids, models, std_report = standardize_all()
    log(f"[OK] Standardization complete. Observed standardized: {std_obs_path}")

    weights = load_weights(obs_grids)
    if weights is None: log("[INFO] Basin average uses equal weights across grids.")
    else: log("[INFO] Basin average uses provided area weights (normalized).")

    qdm_report = {'root': ROOT, 'trace_mm': TRACE_MM, 'ratio_max': RATIO_MAX, 'monthwise': MONTHWISE, 'models': []}

    # Per-model loop with recorder
    for i, (model, scen_map) in enumerate(models.items(), start=1):
        log(f"[INFO] ({i}/{len(models)}) Model: {model}")
        recorder = QDMVizRecorder(model, GIF_FPS, GIF_MAX_FRAMES, GIF_KEEP_FRAMES) if RECORD_GIF else None
        val = run_qdm_for_model(model, scen_map, obs_df, obs_grids, weights, recorder)
        qdm_report['models'].append(val)
        # finalize GIF
        if recorder:
            gif_path = recorder.finalize_gif()
            if gif_path:
                log(f"[OK] GIF for {model}: {gif_path}")
        # close fig
        if recorder and recorder.fig:
            plt.close(recorder.fig)

    with open(REPORT_QDM, 'w', encoding='utf-8') as f:
        json.dump(qdm_report, f, indent=2)
    log(f"[OK] QDM report saved: {REPORT_QDM}")

    # Validation tables
    try:
        rows = []
        for m in qdm_report.get('models', []):
            model_name = m.get('model', '')
            for met in m.get('metrics', []) or []:
                rows.append({
                    'model': model_name,
                    'scenario': met.get('scenario'),
                    'raw_mae_mmday': met.get('raw_mae_mmday'),
                    'raw_rmse_mmday': met.get('raw_rmse_mmday'),
                    'qdm_mae_mmday': met.get('qdm_mae_mmday'),
                    'qdm_rmse_mmday': met.get('qdm_rmse_mmday'),
                    'overlap_days_raw': met.get('overlap_days_raw'),
                    'overlap_days_qdm': met.get('overlap_days_qdm')
                })
        if rows:
            dfm = pd.DataFrame(rows)
            csv_path = os.path.join(OUT_QDM_DIR, 'validation_metrics.csv')
            dfm.to_csv(csv_path, index=False)
            txt_path = os.path.join(OUT_QDM_DIR, 'validation_metrics.txt')
            with open(txt_path, 'w', encoding='utf-8') as tf:
                tf.write('Validation metrics (2015-2024)\n')
                tf.write('model, scenario, raw_mae_mmday, raw_rmse_mmday, qdm_mae_mmday, qdm_rmse_mmday, overlap_days_raw, overlap_days_qdm\n')
                for r in rows:
                    tf.write(', '.join(str(r.get(k, '')) for k in ['model','scenario','raw_mae_mmday','raw_rmse_mmday','qdm_mae_mmday','qdm_rmse_mmday','overlap_days_raw','overlap_days_qdm']) + '\n')
            log(f"[OK] Validation tables saved to: {csv_path} and {txt_path}")
        else:
            log("[WARN] No validation rows to write (no models/scenarios found).")
    except Exception as e:
        log(f"[WARN] Could not write validation CSV/TXT: {e}")

    log(f"[OK] Outputs written to: {OUT_STD_DIR} and {OUT_QDM_DIR}")
    log(f"[OK] Plots written to: {PLOTS_DIR}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
