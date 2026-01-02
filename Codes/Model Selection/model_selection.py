# -*- coding: utf-8 -*-
"""
Model Selection — MAE-based Shape Ranking, Annual Bias, ΔP–ΔT Envelope (Tenfie default)
-----------------------------------------------------------------------------

This script implements a three-step, literature-aligned GCM selection workflow
for each scenario (SSP245, SSP585), using:

Step 1: Monthly "shape" retention via MAE ranks (no seasons, no weights)
        - For each model, compute MAE across the 12 months against reference
          for precipitation and temperature separately.
        - Rank by MAE(P) and MAE(T), average ranks → composite; select Top-24.

Step 2: Annual bias realism (only Step-1 Top-24 are eligible)
        - Compute historical annual means (1985–2014) from scenario annual files.
        - Bias_P% and Bias_T (°C); rank by absolute biases, average → composite; Top-16.

Step 3: Future-change envelope (ΔP–ΔT)
        (A) Tenfie-style default: percentile targets (10/90, and 50/50 median),
            computed against the FULL scenario ensemble; nearest unique picks
            by Euclidean distance from the Top-16 pool (no quadrant enforcement).
        (B) Optional diagnostic: quadrant coverage (median split of pool),
            percentile corners from the POOL; nearest by Euclidean.

Outputs: CSVs and plots for rankings and selections.

Notes:
- Euclidean distance is used (standard in envelope methods).
- Percentile targets follow Tenfie (2022) and Lutz (2016) logic.
- Skill diagnostics (WFDEI-style) are intentionally omitted here and can be
  computed post-selection as per user’s workflow.
"""
from pathlib import Path
import json
import warnings
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ============================== CONFIGURATION ================================ #
BASE_DIR = Path(r"C:/Users/Diwakar Adhikari/Downloads/GCMs/Model Selection")
OUTDIR = BASE_DIR / "outputs_mae_tenfie_V2"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Periods
HIST_START, HIST_END = 1985, 2014
FUT_START, FUT_END = 2071, 2100

SCENARIOS = ["ssp245", "ssp585"]
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Envelope toggles
PRODUCE_QUADRANT_VARIANT = True   # Tenfie default produced regardless; this adds a diagnostic variant
INCLUDE_MEDIAN_TENFIE = True       # Tenfie includes median (50th–50th)

# File names (expected under BASE_DIR)
FILE_NAMES = {
    "observed_pr": "Observed_pr.csv",                 # monthly by year (Year, Jan..Dec)
    "observed_tas": "Observed_avg_temperature.csv",   # monthly by year (Year, Jan..Dec)
    "reference": "reference.csv",                     # rows: Precipitation, Temperature

    # Historical monthly climatology (scenario-specific; index=Model, columns Jan..Dec)
    "pr_hist_monthly_245": "pr_ssp245_monthly_1985_2014.csv",
    "tas_hist_monthly_245": "tas_ssp245_monthly_1985_2014.csv",
    "pr_hist_monthly_585": "pr_ssp585_monthly_1985_2014.csv",
    "tas_hist_monthly_585": "tas_ssp585_monthly_1985_2014.csv",

    # Annual model files (per scenario; index=Model, columns=years)
    "pr_ssp245_annual": "pr_ssp245_annual.csv",
    "tas_ssp245_annual": "tas_ssp245_annual.csv",
    "pr_ssp585_annual": "pr_ssp585_annual.csv",
    "tas_ssp585_annual": "tas_ssp585_annual.csv",
}
FILES: Dict[str, Path] = {k: BASE_DIR / v for k, v in FILE_NAMES.items()}

# ================================ UTILITIES ================================= #
def _safe_read_csv(pathlike: Path) -> pd.DataFrame:
    path = Path(pathlike)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def _ensure_month_columns(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    months_present = [m for m in MONTHS if m in df.columns]
    if len(months_present) != 12:
        raise ValueError(f"{source_name} must contain all months Jan..Dec. Found: {months_present}")
    return df[MONTHS].apply(pd.to_numeric, errors="coerce")

def build_model_color_map(model_lists: List[List[str]]) -> Dict[str, str]:
    """Stable mapping: model -> hex color."""
    all_models = []
    for lst in model_lists: all_models.extend(lst)
    uniq = list(dict.fromkeys(all_models))
    palettes = []
    for cmap in ["tab20","tab20b","tab20c"]:
        cm = plt.get_cmap(cmap)
        palettes.extend([to_hex(cm(i/20.0)) for i in range(20)])
    colors = palettes.copy()
    if len(uniq) > len(colors):
        extra = len(uniq) - len(colors)
        for i in range(extra):
            hue = (i / max(extra, 1))
            cm = plt.get_cmap("hsv")
            colors.append(to_hex(cm(hue)))
    mapping = {m: colors[i % len(colors)] for i, m in enumerate(uniq)}
    pd.Series(mapping, name="color").to_csv(OUTDIR / "model_colors.csv")
    return mapping

# ============================== READING HELPERS ============================== #
def read_reference_monthly(fn: Path) -> Dict[str, pd.Series]:
    df = _safe_read_csv(fn)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index(df.columns[0])
    df = _ensure_month_columns(df, fn.name)
    ref_pr = df.loc["Precipitation"].astype(float)
    ref_tas = df.loc["Temperature"].astype(float)
    return {"pr": ref_pr, "tas": ref_tas}

def read_model_monthly_climatology(fn: Path) -> pd.DataFrame:
    df = _safe_read_csv(fn)
    if "Model" in df.columns:
        df = df.set_index("Model")
    else:
        df = df.set_index(df.columns[0])
    df = _ensure_month_columns(df, fn.name)
    return df

def read_observed_monthly(fn: Path) -> pd.DataFrame:
    df = _safe_read_csv(fn)
    year_col_candidates = [c for c in df.columns if str(c).lower() in ("year", "years")]
    if not year_col_candidates:
        raise ValueError(f"{fn.name} must have a 'year' column.")
    df = df.set_index(year_col_candidates[0])
    df.index = df.index.astype(int)
    df = _ensure_month_columns(df, fn.name)
    return df

def read_annual_gcm(fn: Path) -> pd.DataFrame:
    df = _safe_read_csv(fn)
    if "Model" in df.columns:
        df = df.set_index("Model")
    else:
        df = df.set_index(df.columns[0])
    year_cols = [c for c in df.columns if str(c).isdigit()]
    if not year_cols:
        raise ValueError(f"{fn.name} must have numeric year columns (1985..2100).")
    df = df[year_cols].apply(pd.to_numeric, errors="coerce")
    df.columns = df.columns.astype(int)
    return df

# =============================== STEP-1 (MAE) =============================== #
def mae_monthly_shape_ranking(pr_mon: pd.DataFrame, tas_mon: pd.DataFrame, ref: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute MAE across 12 months for P and T vs reference; rank and average ranks."""
    # Ensure same set of models
    common = pr_mon.index.intersection(tas_mon.index)
    pr_mon = pr_mon.loc[common, MONTHS]
    tas_mon = tas_mon.loc[common, MONTHS]

    # MAE across months
    mae_pr = (pr_mon.sub(ref["pr"], axis=1).abs()).mean(axis=1)
    mae_tas = (tas_mon.sub(ref["tas"], axis=1).abs()).mean(axis=1)

    # Ranks (lower MAE is better)
    rank_pr = mae_pr.rank(method="min", ascending=True)
    rank_tas = mae_tas.rank(method="min", ascending=True)

    composite = (rank_pr + rank_tas) / 2.0

    out = pd.DataFrame({
        "MAE_P": mae_pr,
        "MAE_T": mae_tas,
        "rank_P": rank_pr,
        "rank_T": rank_tas,
        "composite_rank": composite
    })
    return out.sort_values("composite_rank")

# =============================== STEP-2 (Annual) ============================= #
def observed_historical_means(obs_pr: pd.DataFrame, obs_tas: pd.DataFrame) -> Tuple[float, float]:
    pr_hist = obs_pr.loc[HIST_START:HIST_END]
    tas_hist = obs_tas.loc[HIST_START:HIST_END]
    p_yearly = pr_hist.sum(axis=1)   # annual precip totals
    t_yearly = tas_hist.mean(axis=1) # annual mean temp
    return float(p_yearly.mean()), float(t_yearly.mean())

def compute_annual_bias_table(pr_hist_mod: pd.Series, tas_hist_mod: pd.Series, p_obs_hist: float, t_obs_hist: float) -> pd.DataFrame:
    out = pd.DataFrame({
        "Bias_P_percent": (pr_hist_mod - p_obs_hist) / (p_obs_hist + 1e-6) * 100.0,
        "Bias_T_degC": (tas_hist_mod - t_obs_hist),
    })
    out["abs_Bias_P_percent"] = out["Bias_P_percent"].abs()
    out["abs_Bias_T_degC"]    = out["Bias_T_degC"].abs()
    out["rank_P"]  = out["abs_Bias_P_percent"].rank(method="min", ascending=True)
    out["rank_T"] = out["abs_Bias_T_degC"].rank(method="min", ascending=True)
    out["composite_rank_50_50"] = (out["rank_P"] + out["rank_T"]) / 2.0
    return out.sort_values("composite_rank_50_50")

# ============================ STEP-3 (ΔP–ΔT Envelope) ======================== #
def historical_future_means_annual(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    hist = df.loc[:, HIST_START:HIST_END].mean(axis=1)
    fut  = df.loc[:, FUT_START:FUT_END].mean(axis=1)
    return hist, fut

def future_changes_annual(pr_df: pd.DataFrame, tas_df: pd.DataFrame) -> pd.DataFrame:
    pr_hist, pr_fut = historical_future_means_annual(pr_df)
    tas_hist, tas_fut = historical_future_means_annual(tas_df)
    d = pd.DataFrame({"P_hist": pr_hist, "P_fut": pr_fut, "T_hist": tas_hist, "T_fut": tas_fut})
    eps = 1e-6
    d["dP_percent"] = (d["P_fut"] - d["P_hist"]) / (d["P_hist"].abs() + eps) * 100.0
    d["dT_degC"]    = d["T_fut"] - d["T_hist"]
    return d

def envelope_targets_percentiles(df: pd.DataFrame, include_median: bool) -> Dict[str, Tuple[float, float]]:
    p10P, p50P, p90P = np.percentile(df["dP_percent"].dropna(), [10, 50, 90])
    p10T, p50T, p90T = np.percentile(df["dT_degC"].dropna(), [10, 50, 90])
    targets = {
        "Cold-Dry": (p10P, p10T),
        "Cold-Wet": (p90P, p10T),
        "Warm-Dry": (p10P, p90T),
        "Warm-Wet": (p90P, p90T),
    }
    if include_median:
        targets["Median"] = (p50P, p50T)
    return targets

def pick_nearest_unique(sub: pd.DataFrame, targets: Dict[str, Tuple[float,float]], used: set) -> Dict[str, str]:
    picks = {}
    for label, (tx, ty) in targets.items():
        cand = sub.copy()
        cand["distance"] = np.sqrt((cand["dP_percent"] - tx)**2 + (cand["dT_degC"] - ty)**2)
        cand = cand.sort_values("distance", ascending=True)
        pick = None
        for idx in cand.index:
            if idx not in used:
                pick = idx
                break
        if pick is None and not cand.empty:
            pick = cand.index[0]
        if pick is not None:
            picks[label] = pick
            used.add(pick)
    return picks

def envelope_tenfie_default(chg_all: pd.DataFrame, pool_models: List[str], include_median: bool) -> Dict[str, str]:
    """Tenfie-style: percentile targets from FULL ensemble; selection from pool (Top-16)."""
    sub_pool = chg_all.loc[pool_models, ["dP_percent","dT_degC"]].dropna()
    if sub_pool.empty:
        return {}
    targets_full = envelope_targets_percentiles(chg_all, include_median)
    used = set()
    return pick_nearest_unique(sub_pool, targets_full, used)

# def envelope_quadrant_variant(chg_pool: pd.DataFrame) -> Dict[str, str]:
#     """Median-split quadrants inside POOL; corners from POOL percentiles; 4 picks (no median)."""
#     sub = chg_pool[["dP_percent","dT_degC"]].dropna()
#     if sub.empty:
#         return {}
#     # corners from POOL
#     targets_pool = envelope_targets_percentiles(sub, include_median=False)
#     # quadrant masks via medians
#     p_med, t_med = sub["dP_percent"].median(), sub["dT_degC"].median()
#     quad_masks = {
#         "Cold-Dry": (sub["dP_percent"] <= p_med) & (sub["dT_degC"] <= t_med),
#         "Cold-Wet": (sub["dP_percent"] >= p_med) & (sub["dT_degC"] <= t_med),
#         "Warm-Dry": (sub["dP_percent"] <= p_med) & (sub["dT_degC"] >= t_med),
#         "Warm-Wet": (sub["dP_percent"] >= p_med) & (sub["dT_degC"] >= t_med),
#     }
#     picks, used = {}, set()
#     for label in ["Cold-Dry","Cold-Wet","Warm-Dry","Warm-Wet"]:
#         cand = sub[quad_masks[label]].copy()
#         if cand.empty:
#             cand = sub.copy()
#         tx, ty = targets_pool[label]
#         cand["distance"] = np.sqrt((cand["dP_percent"] - tx)**2 + (cand["dT_degC"] - ty)**2)
#         cand = cand.sort_values("distance", ascending=True)
#         pick = None
#         for idx in cand.index:
#             if idx not in used:
#                 pick = idx
#                 break
#         if pick is None and not cand.empty:
#             pick = cand.index[0]
#         if pick is not None:
#             picks[label] = pick
#             used.add(pick)
#     return picks

def flatten_picks(picks: Dict[str, str]) -> List[str]:
    return list(dict.fromkeys(picks.values()))

# ================================= PLOTTING ================================= #
def plot_lines_group(pr_mon: pd.DataFrame, tas_mon: pd.DataFrame, ref: Dict[str, pd.Series],
                     group_models: List[str], colors: Dict[str,str], title_suffix: str, out_prefix: str, legend: bool):
    # Precipitation
    plt.figure(figsize=(9,6))
    plt.plot(MONTHS, ref["pr"].values, color="black", lw=3, label="Reference")
    for m in group_models:
        if m in pr_mon.index:
            y = pr_mon.loc[m, MONTHS].values
            plt.plot(MONTHS, y, color=colors.get(m, "#888888"), lw=1.5)
    plt.title(f"Monthly Climatology (P) — {title_suffix}")
    plt.ylabel("Precipitation (mm)")
    plt.grid(alpha=0.3)
    if legend:
        handles = [plt.Line2D([],[],color="black",lw=3,label="Reference")]
        for m in group_models:
            handles.append(plt.Line2D([],[],color=colors.get(m, "#888888"), lw=2, label=m))
        plt.legend(handles=handles, fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(OUTDIR / f"{out_prefix}_pr.png", dpi=200); plt.close()

    # Temperature
    plt.figure(figsize=(9,6))
    plt.plot(MONTHS, ref["tas"].values, color="black", lw=3, label="Reference")
    for m in group_models:
        if m in tas_mon.index:
            y = tas_mon.loc[m, MONTHS].values
            plt.plot(MONTHS, y, color=colors.get(m, "#888888"), lw=1.5)
    plt.title(f"Monthly Climatology (T) — {title_suffix}")
    plt.ylabel("Temperature (°C)")
    plt.grid(alpha=0.3)
    if legend:
        handles = [plt.Line2D([],[],color="black",lw=3,label="Reference")]
        for m in group_models:
            handles.append(plt.Line2D([],[],color=colors.get(m, "#888888"), lw=2, label=m))
        plt.legend(handles=handles, fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(OUTDIR / f"{out_prefix}_tas.png", dpi=200); plt.close()

def plot_dP_dT(df: pd.DataFrame, models: List[str], colors: Dict[str,str], title: str, outpng: Path,
               corner_marks: Dict[str, Tuple[float,float]] = None):
    sub = df.loc[models, ["dP_percent","dT_degC"]].copy().dropna()
    outpng.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,6))
    plt.axvline(0, color="silver", ls="--", lw=1)
    plt.axhline(0, color="silver", ls="--", lw=1)
    for m, row in sub.iterrows():
        plt.scatter(row["dP_percent"], row["dT_degC"], s=80, color=colors.get(m, "#888888"))
        plt.text(row["dP_percent"]+0.3, row["dT_degC"]+0.03, m, fontsize=9, color=colors.get(m, "#888888"))
    if corner_marks:
        for lbl, (px, py) in corner_marks.items():
            plt.scatter(px, py, s=60, marker="x", color="black")
            plt.text(px+0.3, py+0.03, lbl, fontsize=9, color="black")
    plt.xlabel("ΔP (%) — (2071–2100 vs 1985–2014)")
    plt.ylabel("ΔT (°C) — (2071–2100 vs 1985–2014)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(outpng, dpi=200); plt.close()

# ================================ WORKFLOW ================================== #
if __name__ == "__main__":
    log_msgs: List[str] = []
    print("Step 0: Setup & file validation")

    minimally_required = [
        FILES["observed_pr"], FILES["observed_tas"], FILES["reference"],
        FILES["pr_hist_monthly_245"], FILES["tas_hist_monthly_245"],
        FILES["pr_hist_monthly_585"], FILES["tas_hist_monthly_585"],
        FILES["pr_ssp245_annual"], FILES["tas_ssp245_annual"],
        FILES["pr_ssp585_annual"], FILES["tas_ssp585_annual"],
    ]
    for path in minimally_required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    # Load: reference + observed monthly (by year)
    ref = read_reference_monthly(FILES["reference"])
    obs_pr = read_observed_monthly(FILES["observed_pr"])  # monthly by year
    obs_tas = read_observed_monthly(FILES["observed_tas"])  # monthly by year
    P_obs_hist, T_obs_hist = observed_historical_means(obs_pr, obs_tas)

    # Build color map over union of models present in monthly & annual files
    print("Building consistent model color map…")
    model_sets: List[List[str]] = []
    # monthly (per scenario)
    for scen in SCENARIOS:
        prm = read_model_monthly_climatology(FILES[f"pr_hist_monthly_{'245' if '245' in scen else '585'}"])
        tasm = read_model_monthly_climatology(FILES[f"tas_hist_monthly_{'245' if '245' in scen else '585'}"])
        model_sets.append(list(prm.index)); model_sets.append(list(tasm.index))
    # annual
    for scen in SCENARIOS:
        for var in ["pr", "tas"]:
            key = f"{var}_{scen}_annual"
            if FILES.get(key) and FILES[key].exists():
                try:
                    df = read_annual_gcm(FILES[key])
                    model_sets.append(list(df.index))
                except Exception:
                    pass
    colors = build_model_color_map(model_sets)

    per_scen_top24: Dict[str, List[str]] = {}
    per_scen_top16: Dict[str, List[str]] = {}
    selections: Dict[str, Dict[str, List[str]]] = {}

    # ------------------------------ STEP 1 & 2 per scenario -------------------- #
    for scen in SCENARIOS:
        print(f"\n[{scen}] Step 1: Monthly shape (MAE ranks) → Top-24")
        suf = "245" if "245" in scen else "585"
        pr_hist_mon_s = read_model_monthly_climatology(FILES[f"pr_hist_monthly_{suf}"])
        tas_hist_mon_s = read_model_monthly_climatology(FILES[f"tas_hist_monthly_{suf}"])

        mae_df_s = mae_monthly_shape_ranking(pr_hist_mon_s, tas_hist_mon_s, ref)
        mae_df_s.to_csv(OUTDIR / f"step1_mae_monthly_shape_{scen}.csv")

        top24_s = list(mae_df_s.head(24).index)
        per_scen_top24[scen] = top24_s
        pd.Series(top24_s, name=f"Top24_Step1_{scen}").to_csv(OUTDIR / f"top24_step1_{scen}.csv")

        # Plots: All vs Top-24 vs reference
        plot_lines_group(pr_hist_mon_s, tas_hist_mon_s, ref, list(pr_hist_mon_s.index), colors,
                         title_suffix=f"All Models — {scen.upper()}", out_prefix=f"lines_all_{scen}", legend=False)
        plot_lines_group(pr_hist_mon_s, tas_hist_mon_s, ref, top24_s, colors,
                         title_suffix=f"Top-24 (MAE shape) — {scen.upper()}", out_prefix=f"lines_top24_{scen}", legend=False)

        print(f"[{scen}] Step 2: Annual bias realism → Top-16")
        pr_ann = read_annual_gcm(FILES[f"pr_{scen}_annual"])  # precip totals per year
        tas_ann = read_annual_gcm(FILES[f"tas_{scen}_annual"])  # annual mean tas per year

        avail_models_s = [m for m in top24_s if (m in pr_ann.index) and (m in tas_ann.index)]
        pd.DataFrame({"Model": avail_models_s}).to_csv(OUTDIR / f"step2_availability_{scen}.csv", index=False)

        # Historical means (1985–2014)
        pr_hist_mod_s  = pr_ann.loc[avail_models_s, HIST_START:HIST_END].mean(axis=1)
        tas_hist_mod_s = tas_ann.loc[avail_models_s, HIST_START:HIST_END].mean(axis=1)

        bias_df_s = compute_annual_bias_table(pr_hist_mod_s, tas_hist_mod_s, P_obs_hist, T_obs_hist)
        bias_df_s.to_csv(OUTDIR / f"step2_annual_bias_summary_{scen}.csv")

        top16_s = list(bias_df_s.head(16).index)
        per_scen_top16[scen] = top16_s
        pd.Series(top16_s, name=f"Top16_Step2_{scen}").to_csv(OUTDIR / f"top16_step2_{scen}.csv")

        plot_lines_group(pr_hist_mon_s, tas_hist_mon_s, ref, top16_s, colors,
                         title_suffix=f"Top-16 (Annual bias) — {scen.upper()}", out_prefix=f"lines_top16_{scen}", legend=False)

    # ------------------------------ STEP 3 ------------------------------------- #
    print("\nStep 3: ΔP–ΔT Envelope per scenario (Tenfie default + optional quadrant variant)")
    for scen in SCENARIOS:
        print(f"Processing scenario: {scen}")
        try:
            pr_df = read_annual_gcm(FILES[f"pr_{scen}_annual"])  # annual totals
            tas_df = read_annual_gcm(FILES[f"tas_{scen}_annual"])  # annual mean tas
        except Exception as e:
            warnings.warn(f"{scen}: annual files missing or unreadable: {e}")
            continue

        pool = per_scen_top16.get(scen, [])
        pool_avail = [m for m in pool if (m in pr_df.index) and (m in tas_df.index)]
        if len(pool_avail) < 4:
            print(f"[{scen}] Fewer than 4 models in Top-16; expanding to Top-24.")
            pool_avail = [m for m in per_scen_top24.get(scen, []) if (m in pr_df.index) and (m in tas_df.index)]
        if len(pool_avail) < 4:
            print(f"[{scen}] Fewer than 4 models in Top-24; expanding to all models.")
            pool_avail = [m for m in pr_df.index if m in tas_df.index]

        # Tenfie-style default: targets from FULL ensemble; selection from pool
        chg_all = future_changes_annual(pr_df, tas_df)
        chg_all.to_csv(OUTDIR / f"step3_future_changes_all_{scen}.csv")

        picks_tenfie = envelope_tenfie_default(chg_all, pool_avail, INCLUDE_MEDIAN_TENFIE)
        pd.Series(picks_tenfie, name="Model").to_csv(OUTDIR / f"step3_selection_tenfie_{scen}.csv")
        sel_tenfie = flatten_picks(picks_tenfie)
        selections.setdefault(scen, {})["tenfie_default"] = sel_tenfie

        # Plots for Tenfie set
        suf = "245" if "245" in scen else "585"
        pr_hist_mon_s = read_model_monthly_climatology(FILES[f"pr_hist_monthly_{suf}"])
        tas_hist_mon_s = read_model_monthly_climatology(FILES[f"tas_hist_monthly_{suf}"])
        plot_lines_group(pr_hist_mon_s, tas_hist_mon_s, ref, sel_tenfie, colors,
                         title_suffix=f"Selected (Tenfie: corners + median) — {scen.upper()}",
                         out_prefix=f"lines_selected_tenfie_{scen}", legend=True)
        corner_marks_full = envelope_targets_percentiles(chg_all, INCLUDE_MEDIAN_TENFIE)
        plot_dP_dT(chg_all, sel_tenfie, colors,
                   title=f"ΔP vs ΔT — Selected (Tenfie) {scen.upper()}",
                   outpng=OUTDIR / f"dP_dT_selected_tenfie_{scen}.png",
                   corner_marks=corner_marks_full)

        # Optional quadrant variant inside POOL
        # if PRODUCE_QUADRANT_VARIANT:
        #     chg_pool = chg_all.loc[pool_avail, ["dP_percent","dT_degC"]].dropna()
        #     picks_quad = envelope_quadrant_variant(chg_pool)
        #     pd.Series(picks_quad, name="Model").to_csv(OUTDIR / f"step3_selection_quadrant_{scen}.csv")
        #     sel_quad = flatten_picks(picks_quad)
        #     selections.setdefault(scen, {})["quadrant_variant"] = sel_quad

        #     plot_lines_group(pr_hist_mon_s, tas_hist_mon_s, ref, sel_quad, colors,
        #                      title_suffix=f"Selected (Quadrant coverage) — {scen.upper()}",
        #                      out_prefix=f"lines_selected_quadrant_{scen}", legend=True)
        #     corner_marks_pool = envelope_targets_percentiles(chg_pool, include_median=False)
        #     plot_dP_dT(chg_all, sel_quad, colors,
        #                title=f"ΔP vs ΔT — Selected (Quadrant) {scen.upper()}",
        #                outpng=OUTDIR / f"dP_dT_selected_quadrant_{scen}.png",
        #                corner_marks=corner_marks_pool)

    # ------------------------------ MANIFEST ---------------------------------- #
    manifest = {
        "BASE_DIR": str(BASE_DIR),
        "OUTDIR": str(OUTDIR),
        "HIST_PERIOD": [HIST_START, HIST_END],
        "FUT_PERIOD": [FUT_START, FUT_END],
        "SCENARIOS": SCENARIOS,
        "STEP1": {
            "method": "Monthly MAE vs reference; rank P and T; average ranks",
            "top24_per_scenario": {k: len(v) for k, v in per_scen_top24.items()}
        },
        "STEP2": {
            "method": "Annual bias realism (P% and T°C); rank absolute biases; average ranks",
            "top16_per_scenario": {k: len(v) for k, v in per_scen_top16.items()}
        },
        "STEP3": {
            "tenfie_default": "Percentile targets from FULL ensemble; Euclidean; corners + median; select within Top-16 pool",
            "quadrant_variant": "Median-split quadrants in pool; corners from POOL percentiles; Euclidean",
            "selections": selections
        },
        "DISTANCE_METRIC": "Euclidean (ΔP–ΔT space)",
        "SKILL_DIAGNOSTICS": "Not computed here (to be done post-selection as per workflow)"
    }
    with open(OUTDIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nAll done. Outputs saved under:", OUTDIR.as_posix())
