"""
FULL EXPERIMENT (for the paper)
Can group-level explanations catch what global explanations miss?
Folktables ACSPublicCoverage, natural geographic + temporal shift.

Colab:
    Cell 1:  !pip install folktables xgboost shap fairlearn
    Cell 2:  paste this file and run

Robustness grid:
    source states x models x seeds, evaluated on all target state-years,
    for 3 group definitions: race, sex, race x sex (intersectional).

Outputs (in ./outputs):
    all_results.csv        every shift x config
    correlations.csv       Spearman per config (main evidence)
    summary.txt            aggregated answer to the research question
    table_main.tex         LaTeX table ready for Overleaf
    fig_main_scatter.png   global vs group signal (pooled)
    fig_win_rate.png       how often group signal beats global, per group type
    fig_source_disparity.png
Partial results are saved after every source/model/seed, so a Colab
disconnect does not lose finished work (set RESUME=True to continue).
"""

import os, time, warnings
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from folktables import ACSDataSource, ACSPublicCoverage
from fairlearn.metrics import MetricFrame, false_positive_rate, demographic_parity_difference

warnings.filterwarnings("ignore")

# ======================= SETTINGS =======================
SOURCES = [("CA", "2014"), ("TX", "2014"), ("NY", "2014")]
TARGET_STATES = ["CA", "TX", "NY", "FL", "MI", "GA", "OH", "PA", "WA", "AZ"]
TARGET_YEARS = ["2014", "2015", "2016", "2017", "2018"]
MODELS = ["xgb", "rf"]
SEEDS = [0, 1, 2]
GROUPS = ["race", "sex", "race_x_sex"]
SHAP_SAMPLE = 2000
TOP_K = 5
OUT = "/content/drive/MyDrive/paper_outputs"
RESUME = True
QUICK_TEST = False   # True = tiny run (2 targets, 1 seed) to check everything works
# ========================================================

if QUICK_TEST:
    SOURCES, TARGET_STATES, TARGET_YEARS, SEEDS = SOURCES[:1], ["CA", "TX"], ["2014", "2015"], [0]

os.makedirs(OUT, exist_ok=True)
_sources, _data = {}, {}


def load(state, year):
    key = (state, year)
    if key not in _data:
        if year not in _sources:
            _sources[year] = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
        acs = _sources[year].get_data(states=[state], download=True)
        X, y, _ = ACSPublicCoverage.df_to_pandas(acs)
        X = X.astype(float)
        race = np.where(X["RAC1P"].values == 1, "White", "Non-white")
        sex = np.where(X["SEX"].values == 1, "Male", "Female")
        _data[key] = (X, y.iloc[:, 0].astype(int).values,
                      {"race": race, "sex": sex, "race_x_sex": np.char.add(np.char.add(race, "|"), sex)})
    return _data[key]


def make_model(name, seed):
    if name == "xgb":
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                 random_state=seed, eval_metric="logloss", n_jobs=-1)
    return RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20,
                                  random_state=seed, n_jobs=-1)


def abs_shap(model, X, seed):
    r = np.random.default_rng(seed)
    idx = r.choice(len(X), size=min(SHAP_SAMPLE, len(X)), replace=False)
    sv = shap.TreeExplainer(model).shap_values(X.iloc[idx])
    if isinstance(sv, list):          # older shap + RF: list per class
        sv = sv[1]
    if sv.ndim == 3:                  # newer shap + RF: (n, f, classes)
        sv = sv[:, :, 1]
    return np.abs(sv), idx


def group_disparity(abs_sv, g):
    """Per-feature spread (max - min) of mean |SHAP| across groups."""
    means = np.vstack([abs_sv[g == k].mean(axis=0) for k in np.unique(g) if (g == k).sum() >= 30])
    return means.max(axis=0) - means.min(axis=0)


def cosine_dist(u, v):
    return 1 - float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def jaccard_topk(u, v, k=TOP_K):
    a, b = set(np.argsort(u)[-k:]), set(np.argsort(v)[-k:])
    return len(a & b) / len(a | b)


def fair(y, p, g):
    return (float(MetricFrame(metrics=false_positive_rate, y_true=y, y_pred=p,
                              sensitive_features=g).difference()),
            float(demographic_parity_difference(y, p, sensitive_features=g)))


# ======================= MAIN LOOP =======================
ckpt = f"{OUT}/all_results.csv"
done = set()
rows = []
if RESUME and os.path.exists(ckpt):
    prev = pd.read_csv(ckpt)
    rows = prev.to_dict("records")
    done = set(zip(prev.source, prev.model, prev.seed))
    print(f"Resuming: {len(done)} configs already done")

source_disp_example = None
t0 = time.time()
for s_state, s_year in SOURCES:
    src_name = f"{s_state}{s_year}"
    Xs, ys, gs = load(s_state, s_year)
    for m in MODELS:
        for seed in SEEDS:
            if (src_name, m, seed) in done:
                continue
            print(f"[{time.time()-t0:6.0f}s] source={src_name} model={m} seed={seed}")
            model = make_model(m, seed).fit(Xs, ys)
            ps = model.predict(Xs)
            s_sv, s_idx = abs_shap(model, Xs, seed)
            s_glob = s_sv.mean(axis=0)
            s_disp = {g: group_disparity(s_sv, gs[g][s_idx]) for g in GROUPS}
            s_fair = {g: fair(ys, ps, gs[g]) for g in GROUPS}
            if source_disp_example is None and m == "xgb":
                source_disp_example = (list(Xs.columns), s_disp["race"], src_name)

            for t_state in TARGET_STATES:
                for t_year in TARGET_YEARS:
                    if (t_state, t_year) == (s_state, s_year):
                        continue
                    Xt, yt, gt = load(t_state, t_year)
                    pt = model.predict(Xt)
                    t_sv, t_idx = abs_shap(model, Xt, seed)
                    t_glob = t_sv.mean(axis=0)
                    stype = ("temporal" if t_state == s_state else
                             "geographic" if t_year == s_year else "both")
                    for g in GROUPS:
                        fpr, dp = fair(yt, pt, gt[g])
                        disp = group_disparity(t_sv, gt[g][t_idx])
                        rows.append({
                            "source": src_name, "model": m, "seed": seed, "group": g,
                            "target_state": t_state, "target_year": int(t_year), "shift_type": stype,
                            "accuracy": accuracy_score(yt, pt),
                            "fpr_gap": fpr, "dp_diff": dp,
                            "delta_fpr_gap": abs(fpr - s_fair[g][0]),
                            "delta_dp": abs(dp - s_fair[g][1]),
                            "global_shap_change": cosine_dist(s_glob, t_glob),
                            "global_topk_jaccard": jaccard_topk(s_glob, t_glob),
                            "group_disp_change": float(np.abs(disp - s_disp[g]).sum()),
                        })
            pd.DataFrame(rows).to_csv(ckpt, index=False)   # checkpoint

res = pd.DataFrame(rows)
print(f"\nAll experiments done: {len(res)} rows, {time.time()-t0:.0f}s")

# ======================= ANALYSIS =======================
corr_rows = []
for (src, m, seed, g), d in res.groupby(["source", "model", "seed", "group"]):
    for target in ["delta_fpr_gap", "delta_dp"]:
        r_glob, p_glob = spearmanr(d.global_shap_change, d[target])
        r_grp, p_grp = spearmanr(d.group_disp_change, d[target])
        corr_rows.append({"source": src, "model": m, "seed": seed, "group": g, "fairness_metric": target,
                          "rho_global": r_glob, "p_global": p_glob,
                          "rho_group": r_grp, "p_group": p_grp,
                          "group_wins": r_grp > r_glob, "n_shifts": len(d)})
corr = pd.DataFrame(corr_rows)
corr.to_csv(f"{OUT}/correlations.csv", index=False)

agg = (corr.groupby(["group", "fairness_metric"])
           .agg(rho_global=("rho_global", "mean"), rho_group=("rho_group", "mean"),
                win_rate=("group_wins", "mean"), n_configs=("group_wins", "size"))
           .reset_index())
by_model = corr.groupby(["model", "fairness_metric"])[["rho_global", "rho_group", "group_wins"]].mean()
by_shift = (res.groupby(["group", "shift_type"])
               .apply(lambda d: pd.Series({
                   "rho_global": spearmanr(d.global_shap_change, d.delta_fpr_gap)[0],
                   "rho_group": spearmanr(d.group_disp_change, d.delta_fpr_gap)[0]}))
               .reset_index())

summary = [
    "RESEARCH QUESTION: does group-level explanation disparity track fairness change",
    "better than global explanation change under natural distribution shift?",
    f"Rows: {len(res)} | configs: {len(corr)//2} | shifts per config: {corr.n_shifts.iloc[0]}",
    "", "=== Mean Spearman rho by group type (higher = better signal) ===", agg.round(3).to_string(index=False),
    "", "=== By model ===", by_model.round(3).to_string(),
    "", "=== By shift type (FPR gap, pooled over configs) ===", by_shift.round(3).to_string(index=False),
    "", "win_rate = share of configs where the group signal beat the global signal.",
    "Rule of thumb: win_rate > 0.7 and rho_group clearly > rho_global = supports your hypothesis.",
]
open(f"{OUT}/summary.txt", "w").write("\n".join(summary))
print("\n".join(summary))

# LaTeX table for the paper
tex = agg[agg.fairness_metric == "delta_fpr_gap"][["group", "rho_global", "rho_group", "win_rate", "n_configs"]].copy()
tex.columns = ["Group", r"$\rho$ global", r"$\rho$ group", "Win rate", "Configs"]
tex["Group"] = tex["Group"].str.replace("_x_", r" $\times$ ", regex=False)
open(f"{OUT}/table_main.tex", "w").write(tex.to_latex(index=False, float_format="%.3f", escape=False))

# ======================= FIGURES =======================
d = res[res.group == "race"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for a, sig, title in [(ax[0], "global_shap_change", "Global SHAP change (cosine distance)"),
                      (ax[1], "group_disp_change", "Group-level SHAP disparity change")]:
    for t, mk in [("temporal", "o"), ("geographic", "s"), ("both", "^")]:
        dd = d[d.shift_type == t]
        a.scatter(dd[sig], dd.delta_fpr_gap, marker=mk, s=14, alpha=0.5, label=t)
    a.set_xlabel(title)
ax[0].set_ylabel("|Change in FPR gap| (race)")
ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig_main_scatter.png", dpi=250); plt.close()

w = agg[agg.fairness_metric == "delta_fpr_gap"]
plt.figure(figsize=(6, 4))
plt.bar(w.group, w.win_rate)
plt.axhline(0.5, ls="--", c="gray")
plt.ylabel("Share of configs where group signal wins"); plt.ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_win_rate.png", dpi=250); plt.close()

if source_disp_example:
    feats, disp, name = source_disp_example
    top = np.argsort(disp)[-10:]
    plt.figure(figsize=(7, 4.5))
    plt.barh([feats[i] for i in top], disp[top])
    plt.xlabel("Spread of mean |SHAP| across race groups")
    plt.title(f"Group explanation disparity at source ({name}, XGBoost)")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_source_disparity.png", dpi=250); plt.close()

print(f"\nDone. Everything is in ./{OUT}/  (download the folder from the Colab file panel)")
