#!/usr/bin/env python3
"""
Reproducibility check for the chokepoint slowdown notebook.

Loads final_feature_vector.csv, then:
  1. retrains logistic regression on the three ratio features
  2. retrains histogram gradient boosting (with the inner-time-split grid search)
  3. reruns k-means on the 28 chokepoint profiles
  4. prints everything and compares to the numbers quoted in the notebook

Usage:  python headline.py [data/final_feature_vector.csv]

EXPECTED COLUMNS (the notebook's `model_data` after the feature step):
  portid, portname, date, n_total, traffic_lag1, mean_7day, mean_28day,
  std_28day, slowdown
Optional (recomputed if absent): ratio_7_28, ratio_lag_28, cv_28, doy_sin, doy_cos
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/final_feature_vector.csv"

# Numbers quoted in the notebook, used only for PASS/FAIL comparison.
EXPECTED = {
    "n_rows": 78244,
    "n_pos": 2269,
    "n_test": 18156,
    "n_test_pos": 653,
    "lr_ap": 0.438,
    "gb_ap": 0.415,
    "lr_recall3": 0.824,
    "persist_recall3": 0.527,
    "gb_recall3": 0.804,
}


def banner(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def check(label, got, want, tol=0.0015):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got:.4f}, notebook says {want}")
    return ok


def recall_at_k(frame, score_col, K, seed=0):
    """Same as the notebook: shuffle so ties break randomly, rank per date."""
    d = frame.sample(frac=1, random_state=seed)
    rank = d.groupby("date")[score_col].rank(method="first", ascending=False)
    sel = rank <= K
    return d.loc[sel, "y"].sum() / d["y"].sum(), d.loc[sel, "y"].mean()


# --------------------------------------------------------------------------
# Load and prepare
# --------------------------------------------------------------------------
banner("0. LOAD")
required = ["portid", "portname", "date", "n_total", "traffic_lag1",
            "mean_7day", "mean_28day", "std_28day", "slowdown"]
model_data = pd.read_csv(PATH)
missing = [c for c in required if c not in model_data.columns]
if missing:
    sys.exit(f"CSV is missing required columns: {missing}\nFound: {list(model_data.columns)}")

model_data["date"] = pd.to_datetime(model_data["date"])
# Same row order as the notebook (sorted by chokepoint then date) so that
# the random tie-breaking in Recall@K reproduces exactly.
model_data = model_data.sort_values(["portid", "date"]).reset_index(drop=True)

if "ratio_7_28" not in model_data:
    model_data["ratio_7_28"] = model_data["mean_7day"] / model_data["mean_28day"]
if "ratio_lag_28" not in model_data:
    model_data["ratio_lag_28"] = model_data["traffic_lag1"] / model_data["mean_28day"]
if "cv_28" not in model_data:
    model_data["cv_28"] = model_data["std_28day"] / model_data["mean_28day"]
doy = model_data["date"].dt.dayofyear
if "doy_sin" not in model_data:
    model_data["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
if "doy_cos" not in model_data:
    model_data["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

ratio_cols = ["ratio_7_28", "ratio_lag_28", "cv_28"]
cyc_cols = ["doy_sin", "doy_cos"]
chokepoint_features = pd.get_dummies(model_data["portid"].astype(str), prefix="cp", dtype=int)
flex_cols = ratio_cols + cyc_cols + list(chokepoint_features.columns)

X = pd.concat([model_data[ratio_cols + cyc_cols], chokepoint_features], axis=1)
X["mean_7day"] = model_data["mean_7day"]      # needed for the persistence score
X["mean_28day"] = model_data["mean_28day"]
y = model_data["slowdown"].astype(int)
dates = model_data["date"]

print(f"rows: {len(model_data):,}   positives: {int(y.sum()):,}   "
      f"chokepoints: {model_data['portid'].nunique()}")
print(f"date range: {dates.min().date()} -> {dates.max().date()}")
print(f"NaN/inf in ratio features: "
      f"{int(model_data[ratio_cols].isna().sum().sum())} / "
      f"{int(np.isinf(model_data[ratio_cols]).sum().sum())}")
check("usable rows", len(model_data), EXPECTED["n_rows"], tol=0)
check("total positives", int(y.sum()), EXPECTED["n_pos"], tol=0)

# --------------------------------------------------------------------------
# Split (locked): cutoff 2025-01-01, 7-day purge
# --------------------------------------------------------------------------
banner("1. SPLIT")
cutoff = pd.Timestamp("2025-01-01")
purge = pd.Timedelta(days=7)
train = dates < cutoff - purge
gap = (dates >= cutoff - purge) & (dates < cutoff)
test = dates >= cutoff
split_table = pd.DataFrame([
    {"set": n, "first": dates[m].min().date(), "last": dates[m].max().date(),
     "dates": dates[m].nunique(), "rows": int(m.sum()), "positives": int(y[m].sum())}
    for n, m in [("Train", train), ("Purge gap", gap), ("Test", test)]
])
print(split_table.to_string(index=False))
check("test rows", int(test.sum()), EXPECTED["n_test"], tol=0)
check("test positives", int(y[test].sum()), EXPECTED["n_test_pos"], tol=0)

# --------------------------------------------------------------------------
# Baselines and logistic regression
# --------------------------------------------------------------------------
banner("2. LOGISTIC REGRESSION (3 ratio features)")
template = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))
simple = clone(template).fit(X.loc[train, ratio_cols], y[train])
lr = simple[-1]
print("intercept:", round(float(lr.intercept_[0]), 3))
print("standardized coefficients:")
print(pd.Series(lr.coef_[0], index=ratio_cols).round(3).to_string())

# reproducibility: second fit must give identical predictions
simple2 = clone(template).fit(X.loc[train, ratio_cols], y[train])
same = np.allclose(simple.predict_proba(X.loc[test, ratio_cols]),
                   simple2.predict_proba(X.loc[test, ratio_cols]))
print(f"refit gives identical test predictions: {same}")

prior = DummyClassifier(strategy="prior").fit(X.loc[train, ratio_cols], y[train])
prior_ap = average_precision_score(y[test], prior.predict_proba(X.loc[test, ratio_cols])[:, 1])
print(f"prior AP: {prior_ap:.4f}   test positive rate: {y[test].mean():.4f}")

# --------------------------------------------------------------------------
# Gradient boosting with honest inner-time-split tuning
# --------------------------------------------------------------------------
banner("3. GRADIENT BOOSTING (tuning on inner time split, then refit)")
val_start = dates[train].max() - pd.Timedelta(days=365)
inner_tr = train & (dates < val_start - pd.Timedelta(days=7))
inner_va = train & (dates >= val_start)
print(f"inner train rows: {int(inner_tr.sum()):,}   inner val rows: {int(inner_va.sum()):,} "
      f"(val positives: {int(y[inner_va].sum())})")

grid = [dict(max_depth=d, learning_rate=lrt, max_iter=n, min_samples_leaf=msl)
        for d, lrt, n, msl in itertools.product([3, 6], [0.05, 0.1], [100, 300], [50, 200])]


def make_gb(params):
    return HistGradientBoostingClassifier(**params, class_weight="balanced",
                                          early_stopping=False, random_state=0)


val_scores = []
for i, p in enumerate(grid, 1):
    m = make_gb(p).fit(X.loc[inner_tr, flex_cols], y[inner_tr])
    s = average_precision_score(y[inner_va], m.predict_proba(X.loc[inner_va, flex_cols])[:, 1])
    val_scores.append(s)
    print(f"  config {i:2d}/{len(grid)}  {p}  val AP = {s:.4f}")

best = grid[int(np.argmax(val_scores))]
print("\nbest params:", best, "| validation AP:", round(max(val_scores), 3))

gb = make_gb(best).fit(X.loc[train, flex_cols], y[train])
gb2 = make_gb(best).fit(X.loc[train, flex_cols], y[train])
same_gb = np.allclose(gb.predict_proba(X.loc[test, flex_cols]),
                      gb2.predict_proba(X.loc[test, flex_cols]))
print(f"refit gives identical test predictions: {same_gb}")

# --------------------------------------------------------------------------
# Test-set scoring
# --------------------------------------------------------------------------
banner("4. TEST-SET RESULTS")
ev = pd.DataFrame({
    "date": dates[test].to_numpy(),
    "y": y[test].to_numpy(),
    "persistence": (1 - X.loc[test, "mean_7day"] / X.loc[test, "mean_28day"]).to_numpy(),
    "logreg": simple.predict_proba(X.loc[test, ratio_cols])[:, 1],
    "gb": gb.predict_proba(X.loc[test, flex_cols])[:, 1],
})
methods = ["persistence", "logreg", "gb"]
aps = {m: average_precision_score(ev["y"], ev[m]) for m in methods}
for m in methods:
    print(f"{m:12s} AP = {aps[m]:.3f}")
print()
recalls = {}
for K in [1, 3, 5]:
    for m in methods:
        r, p = recall_at_k(ev, m, K)
        recalls[(K, m)] = r
        print(f"K={K}  {m:12s} recall={r:.3f}  precision={p:.3f}")

print("\nComparison with notebook numbers:")
check("LogReg AP", aps["logreg"], EXPECTED["lr_ap"], tol=0.0006)
check("GB AP", aps["gb"], EXPECTED["gb_ap"], tol=0.0006)
check("LogReg Recall@3", recalls[(3, "logreg")], EXPECTED["lr_recall3"], tol=0.0006)
check("Persistence Recall@3", recalls[(3, "persistence")], EXPECTED["persist_recall3"], tol=0.0006)
check("GB Recall@3", recalls[(3, "gb")], EXPECTED["gb_recall3"], tol=0.0006)

# --------------------------------------------------------------------------
# K-means on chokepoint profiles (training rows only)
# --------------------------------------------------------------------------
banner("5. K-MEANS (28 chokepoint profiles, training rows only)")
tr = model_data.loc[train]
prof = tr.groupby("portid").agg(
    log_mean=("n_total", lambda s: np.log1p(s.mean())),
    med_cv=("cv_28", "median"),
    zero_frac=("n_total", lambda s: (s == 0).mean()),
)
print(f"profile shape: {prof.shape}")
print("raw variance share per feature (what dominates distance without scaling):")
print((prof.var() / prof.var().sum()).round(3).to_string())

Z = StandardScaler().fit_transform(prof)
print("\nsilhouette by k:")
for k in [2, 3, 4, 5]:
    labels = KMeans(k, n_init=20, random_state=0).fit_predict(Z)
    sizes = np.bincount(labels).tolist()
    print(f"  k={k}: silhouette = {silhouette_score(Z, labels):.3f}   cluster sizes = {sizes}")

k = 3
prof["cluster"] = KMeans(k, n_init=20, random_state=0).fit_predict(Z)
prof2 = KMeans(k, n_init=20, random_state=0).fit_predict(Z)
print(f"\nrerun gives identical k=3 labels: {bool((prof['cluster'].to_numpy() == prof2).all())}")

names = model_data.drop_duplicates("portid").set_index("portid")["portname"]
print(f"\nk={k} clusters:")
for c, g in prof.groupby("cluster"):
    print(f"\n  cluster {c} ({len(g)} chokepoints)")
    print(f"    mean profile (raw units): "
          f"log_mean={g['log_mean'].mean():.2f}, med_cv={g['med_cv'].mean():.2f}, "
          f"zero_frac={g['zero_frac'].mean():.3f}")
    print("    members:", ", ".join(names.loc[g.index].tolist()))

banner("DONE")