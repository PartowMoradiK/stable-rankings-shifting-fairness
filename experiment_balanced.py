"""
EXPERIMENT 2: Is the group-level signal weak because of sampling noise?
------------------------------------------------------------------------
Idea: in a random SHAP sample, minority groups are small, so group-level
means are noisy. Here we draw a STRATIFIED sample (equal size per race x sex
cell) and vary the per-cell size N. If the group signal gets stronger as N
grows, the method works but needs enough samples per group.

Only XGBoost (fast). Same sources / targets / seeds as experiment 1.
One SHAP call per state-year (4 cells x N_MAX rows); smaller N are subsamples,
so all N levels are compared on exactly the same models and data.

Setup (Colab or Jupyter):
    !pip install -q folktables xgboost shap fairlearn
Estimated runtime: ~30-45 minutes.

Outputs (OUT folder):
    balanced_results.csv   one row per shift x seed x group x N
    balanced_summary.txt   main answer
    fig_rho_vs_n.png       signal strength vs samples per group
    fig_snr_vs_n.png       signal-to-noise ratio vs samples per group
"""

import os, time, warnings
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, binomtest
from folktables import ACSDataSource, ACSPublicCoverage
from fairlearn.metrics import MetricFrame, false_positive_rate

warnings.filterwarnings("ignore")

# ======================= SETTINGS =======================
SOURCES = [("CA", "2014"), ("TX", "2014"), ("NY", "2014")]
TARGET_STATES = ["CA", "TX", "NY", "FL", "MI", "GA", "OH", "PA", "WA", "AZ"]
TARGET_YEARS = ["2014", "2015", "2016", "2017", "2018"]
SEEDS = [0, 1, 2]
N_LEVELS = [125, 250, 500, 1000]     # rows per race x sex cell
N_MAX = max(N_LEVELS)
GROUPS = ["race", "sex", "race_x_sex"]
OUT = "outputs_balanced"             # Colab + Drive: "/content/drive/MyDrive/paper_outputs_balanced"
QUICK_TEST = False
# ========================================================

if QUICK_TEST:
    SOURCES, TARGET_STATES, TARGET_YEARS, SEEDS = SOURCES[:1], ["CA", "TX"], ["2014", "2015"], [0]
os.makedirs(OUT, exist_ok=True)
_src, _data = {}, {}


def load(state, year):
    if (state, year) not in _data:
        if year not in _src:
            _src[year] = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        X, y, _ = ACSPublicCoverage.df_to_pandas(_src[year].get_data(states=[state], download=True))
        X = X.astype(float)
        race = np.where(X["RAC1P"].values == 1, "White", "Non-white")
        sex = np.where(X["SEX"].values == 1, "Male", "Female")
        cell = np.char.add(np.char.add(race, "|"), sex)
        _data[(state, year)] = (X, y.iloc[:, 0].astype(int).values,
                                {"race": race, "sex": sex, "race_x_sex": cell})
    return _data[(state, year)]


def stratified_shap(model, X, cell, seed):
    """SHAP on up to N_MAX random rows from each race x sex cell.
    Returns |SHAP|, cell labels and within-cell draw order (for nested subsampling)."""
    r = np.random.default_rng(seed)
    idx, rank = [], []
    for c in np.unique(cell):
        members = np.where(cell == c)[0]
        pick = r.permutation(members)[:N_MAX]
        idx.append(pick)
        rank.append(np.arange(len(pick)))
    idx, rank = np.concatenate(idx), np.concatenate(rank)
    sv = np.abs(shap.TreeExplainer(model).shap_values(X.iloc[idx]))
    return sv, cell[idx], rank


def cell_means(sv, cells, rank, n):
    keep = rank < n
    return {c: sv[keep & (cells == c)].mean(axis=0) for c in np.unique(cells[keep])}


def group_means(cm, group):
    """Combine race x sex cells into race or sex groups (equal weight per cell = balanced)."""
    if group == "race_x_sex":
        return cm
    pos = 0 if group == "race" else 1
    out = {}
    for c, v in cm.items():
        out.setdefault(c.split("|")[pos], []).append(v)
    return {k: np.mean(v, axis=0) for k, v in out.items()}


def disparity(gm):
    m = np.vstack(list(gm.values()))
    return m.max(axis=0) - m.min(axis=0)


def global_mean(cm, cell_full):
    """Population-weighted global |SHAP| from cell means (unbiased despite stratification)."""
    cells, counts = np.unique(cell_full, return_counts=True)
    w = dict(zip(cells, counts / counts.sum()))
    return sum(w[c] * cm[c] for c in cm)


def fpr_gap(y, p, g):
    return float(MetricFrame(metrics=false_positive_rate, y_true=y, y_pred=p,
                             sensitive_features=g).difference())


# ======================= MAIN LOOP =======================
ckpt = f"{OUT}/balanced_results.csv"
rows = pd.read_csv(ckpt).to_dict("records") if os.path.exists(ckpt) else []
done = {(r["source"], r["seed"]) for r in rows}
t0 = time.time()

for s_state, s_year in SOURCES:
    src = f"{s_state}{s_year}"
    Xs, ys, gs = load(s_state, s_year)
    for seed in SEEDS:
        if (src, seed) in done:
            continue
        print(f"[{time.time()-t0:6.0f}s] source={src} seed={seed}")
        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                  random_state=seed, eval_metric="logloss", n_jobs=-1).fit(Xs, ys)
        ps = model.predict(Xs)
        s_sv, s_cells, s_rank = stratified_shap(model, Xs, gs["race_x_sex"], seed)
        s_fpr = {g: fpr_gap(ys, ps, gs[g]) for g in GROUPS}
        s_stats = {}
        for n in N_LEVELS:
            cm = cell_means(s_sv, s_cells, s_rank, n)
            s_stats[n] = (global_mean(cm, gs["race_x_sex"]),
                          {g: disparity(group_means(cm, g)) for g in GROUPS})

        for t_state in TARGET_STATES:
            for t_year in TARGET_YEARS:
                if (t_state, t_year) == (s_state, s_year):
                    continue
                Xt, yt, gt = load(t_state, t_year)
                pt = model.predict(Xt)
                t_sv, t_cells, t_rank = stratified_shap(model, Xt, gt["race_x_sex"], seed + 1000)
                stype = "temporal" if t_state == s_state else "geographic" if t_year == s_year else "both"
                t_fpr = {g: fpr_gap(yt, pt, gt[g]) for g in GROUPS}
                for n in N_LEVELS:
                    cm = cell_means(t_sv, t_cells, t_rank, n)
                    glob = global_mean(cm, gt["race_x_sex"])
                    s_glob, s_disp = s_stats[n]
                    gchange = 1 - float(np.dot(s_glob, glob) / (np.linalg.norm(s_glob) * np.linalg.norm(glob)))
                    for g in GROUPS:
                        rows.append({
                            "source": src, "seed": seed, "group": g, "n_per_cell": n,
                            "target_state": t_state, "target_year": int(t_year), "shift_type": stype,
                            "delta_fpr_gap": abs(t_fpr[g] - s_fpr[g]),
                            "global_shap_change": gchange,
                            "group_disp_change": float(np.abs(disparity(group_means(cm, g)) - s_disp[g]).sum()),
                        })
        pd.DataFrame(rows).to_csv(ckpt, index=False)

res = pd.DataFrame(rows)
print(f"\nDone: {len(res)} rows, {time.time()-t0:.0f}s")

# ======================= ANALYSIS =======================
# 1) correlation with fairness change, per config, per N
corr = []
for (src, seed, g, n), d in res.groupby(["source", "seed", "group", "n_per_cell"]):
    rg = spearmanr(d.global_shap_change, d.delta_fpr_gap)[0]
    rr = spearmanr(d.group_disp_change, d.delta_fpr_gap)[0]
    corr.append({"source": src, "seed": seed, "group": g, "n_per_cell": n,
                 "rho_global": rg, "rho_group": rr, "group_wins": rr > rg})
corr = pd.DataFrame(corr)

# 2) signal-to-noise: variation across targets vs variation across seeds (same target)
snr = []
for (g, n), d in res.groupby(["group", "n_per_cell"]):
    for sig in ["group_disp_change", "global_shap_change"]:
        noise = d.groupby(["source", "target_state", "target_year"])[sig].std().mean()
        signal = d.groupby(["source", "seed"])[sig].std().mean()
        snr.append({"group": g, "n_per_cell": n, "signal": sig, "snr": signal / (noise + 1e-12)})
snr = pd.DataFrame(snr)

agg = (corr.groupby(["group", "n_per_cell"])
           .agg(rho_global=("rho_global", "mean"), rho_group=("rho_group", "mean"),
                win_rate=("group_wins", "mean"), wins=("group_wins", "sum"), n_configs=("group_wins", "size"))
           .reset_index())
agg["sign_test_p"] = [binomtest(int(w), int(k), 0.5).pvalue for w, k in zip(agg.wins, agg.n_configs)]
snr_wide = snr.pivot_table(index=["group", "n_per_cell"], columns="signal", values="snr").reset_index()

trend = []
for g, d in agg.groupby("group"):
    trend.append(f"  {g:>10}: rho_group {d.rho_group.iloc[0]:.3f} (N={d.n_per_cell.iloc[0]}) -> "
                 f"{d.rho_group.iloc[-1]:.3f} (N={d.n_per_cell.iloc[-1]}), "
                 f"global stays ~{d.rho_global.mean():.3f}")

summary = ["QUESTION: does the group-level signal improve when minority groups get enough SHAP samples?",
           f"Rows: {len(res)} | configs per (group, N): {agg.n_configs.iloc[0]}", "",
           "=== Correlation with |change in FPR gap| by samples per race x sex cell ===",
           agg.round(3).to_string(index=False), "",
           "=== Signal-to-noise ratio (variation across targets / across seeds) ===",
           snr_wide.round(2).to_string(index=False), "",
           "=== Trend ===", *trend, "",
           "How to read:",
           "  - rho_group rising with N  -> group signal was limited by sampling noise",
           "  - rho_group > rho_global at large N with sign_test_p < 0.05 -> group signal wins once noise is controlled",
           "  - rho_group flat and below global -> noise is not the explanation"]
open(f"{OUT}/balanced_summary.txt", "w").write("\n".join(summary))
corr.to_csv(f"{OUT}/balanced_correlations.csv", index=False)
print("\n".join(summary))

# ======================= FIGURES =======================
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for g, d in agg.groupby("group"):
    line, = ax.plot(d.n_per_cell, d.rho_group, marker="o", label=f"group signal ({g})")
    ax.plot(d.n_per_cell, d.rho_global, ls="--", color=line.get_color(), alpha=0.6, label=f"global ({g})")
ax.set_xscale("log", base=2); ax.set_xticks(N_LEVELS); ax.set_xticklabels(N_LEVELS)
ax.set_xlabel("SHAP samples per race x sex cell"); ax.set_ylabel("Mean Spearman rho with |change in FPR gap|")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/fig_rho_vs_n.png", dpi=250); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for g, d in snr_wide.groupby("group"):
    ax.plot(d.n_per_cell, d.group_disp_change, marker="o", label=g)
ax.set_xscale("log", base=2); ax.set_xticks(N_LEVELS); ax.set_xticklabels(N_LEVELS)
ax.axhline(1, ls=":", c="gray")
ax.set_xlabel("SHAP samples per race x sex cell"); ax.set_ylabel("Signal-to-noise ratio (group signal)")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig_snr_vs_n.png", dpi=250); plt.close()

print(f"\nFiles saved in {OUT}/")
