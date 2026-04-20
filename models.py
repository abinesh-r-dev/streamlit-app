"""
models.py — ML training, augmentation, and evaluation.

Key design decisions:
- @st.cache_resource  : trained model objects (sklearn estimators). Global, shared
                        across reruns and sessions. No serialization overhead.
                        Models are trained ONCE per unique (n_aug, noise_scale, seed).
- @st.cache_data      : pure data returns (DataFrames, arrays, dicts of numbers).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.interpolate import RBFInterpolator
from scipy.cluster.hierarchy import linkage, fcluster

from config import MINERAL_FEATURES
from data import load_irel, load_basin
from radiation import compute as rad_compute


# ── Data augmentation ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def augment(n_aug: int = 140, noise_scale: float = 0.18, seed: int = 42) -> pd.DataFrame:
    """Gaussian-noise + Mix-up augmentation of the 10-sample IREL dataset."""
    irel = load_irel()
    cols = MINERAL_FEATURES + ["Monazite"]
    orig = irel[cols].values.astype(float)
    stds = orig.std(axis=0, ddof=1) + 1e-6
    rng  = np.random.default_rng(seed)

    rows = []
    for _ in range(n_aug):
        i = rng.integers(0, len(orig))
        s = orig[i].copy()
        if rng.random() < 0.30:
            j = rng.integers(0, len(orig))
            lam = rng.beta(0.4, 0.4)
            s = lam * orig[i] + (1.0 - lam) * orig[j]
        s += rng.normal(0.0, noise_scale * stds)
        s  = np.clip(s, 0.0, None)
        s[MINERAL_FEATURES.index("Heavies")] = s[:MINERAL_FEATURES.index("Heavies")].sum()
        rows.append(s)

    aug = pd.DataFrame(rows, columns=cols)
    aug["src"] = "augmented"
    orig_df = irel[cols].copy()
    orig_df["src"] = "original"
    return pd.concat([orig_df, aug], ignore_index=True)


# ── Model training — cache_resource so models are shared and never re-pickled ──
@st.cache_resource(show_spinner=False)
def train_models(n_aug: int, noise_scale: float, seed: int) -> dict:
    """
    Train RF, GB and KNN on the full augmented dataset.
    @st.cache_resource stores the result globally — the same model object is
    returned on every subsequent call with the same arguments, with zero overhead.
    """
    irel   = load_irel()
    df_aug = augment(n_aug, noise_scale, seed)
    X_orig = irel[MINERAL_FEATURES].values.astype(float)
    y_orig = irel["Monazite"].values.astype(float)
    X_aug  = df_aug.loc[df_aug["src"] == "augmented", MINERAL_FEATURES].values
    y_aug  = df_aug.loc[df_aug["src"] == "augmented", "Monazite"].values

    X_all  = np.vstack([X_orig, X_aug])
    y_all  = np.concatenate([y_orig, y_aug])
    scaler = StandardScaler().fit(X_all)
    Xs     = scaler.transform(X_all)

    rf  = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3,
                                random_state=42, n_jobs=-1).fit(Xs, y_all)
    gb  = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.04,
                                    subsample=0.8, random_state=42).fit(Xs, y_all)
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(Xs, y_all)

    return dict(rf=rf, gb=gb, knn=knn, scaler=scaler,
                X_orig=X_orig, y_orig=y_orig)


# ── LOO evaluation — cache_resource, same rationale ───────────────────────────
@st.cache_resource(show_spinner=False)
def loo_evaluate(n_aug: int, noise_scale: float, seed: int) -> tuple:
    """
    Leave-One-Out cross-validation on the 10 original samples.
    Uses lighter LOO fold models (n_estimators=100) for speed; the full-quality
    models in train_models() are used for actual predictions.
    Returns (results_dict, y_orig_array).
    """
    irel   = load_irel()
    df_aug = augment(n_aug, noise_scale, seed)
    X_orig = irel[MINERAL_FEATURES].values.astype(float)
    y_orig = irel["Monazite"].values.astype(float)
    X_aug  = df_aug.loc[df_aug["src"] == "augmented", MINERAL_FEATURES].values
    y_aug  = df_aug.loc[df_aug["src"] == "augmented", "Monazite"].values

    preds = {"Random Forest": [], "Gradient Boosting": [], "KNN": []}
    for tr, te in LeaveOneOut().split(X_orig):
        Xa  = np.vstack([X_orig[tr], X_aug])
        ya  = np.concatenate([y_orig[tr], y_aug])
        sc  = StandardScaler().fit(Xa)
        Xs  = sc.transform(Xa)
        Xte = sc.transform(X_orig[te])
        preds["Random Forest"].append(
            RandomForestRegressor(n_estimators=100, max_depth=4,
                                  min_samples_leaf=3, random_state=42,
                                  n_jobs=-1).fit(Xs, ya).predict(Xte)[0])
        preds["Gradient Boosting"].append(
            GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                      learning_rate=0.05, random_state=42
                                      ).fit(Xs, ya).predict(Xte)[0])
        preds["KNN"].append(
            KNeighborsRegressor(n_neighbors=4, weights="distance"
                                ).fit(Xs, ya).predict(Xte)[0])

    results = {}
    for name, p_list in preds.items():
        p = np.clip(np.array(p_list), 0.0, None)
        results[name] = dict(
            preds=p,
            r2=round(r2_score(y_orig, p), 4),
            mae=round(mean_absolute_error(y_orig, p), 5),
            rmse=round(float(np.sqrt(mean_squared_error(y_orig, p))), 5),
        )
    return results, y_orig


# ── Bootstrap CI — cache_data (returns plain floats) ──────────────────────────
@st.cache_data(show_spinner=False)
def bootstrap_ci(x_tuple: tuple, model_name: str,
                 n_aug: int, noise_scale: float,
                 seed: int, n_boot: int = 500) -> tuple[float, float, float]:
    irel   = load_irel()
    df_aug = augment(n_aug, noise_scale, seed)
    X_orig = irel[MINERAL_FEATURES].values.astype(float)
    y_orig = irel["Monazite"].values.astype(float)
    X_aug  = df_aug.loc[df_aug["src"] == "augmented", MINERAL_FEATURES].values
    y_aug  = df_aug.loc[df_aug["src"] == "augmented", "Monazite"].values
    X_new  = np.array([x_tuple])
    rng    = np.random.default_rng(seed + 99)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X_orig), len(X_orig))
        Xb  = np.vstack([X_orig[idx], X_aug])
        yb  = np.concatenate([y_orig[idx], y_aug])
        sc  = StandardScaler().fit(Xb)
        m   = (GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                         learning_rate=0.05, random_state=42)
               if model_name == "Gradient Boosting" else
               KNeighborsRegressor(n_neighbors=4, weights="distance")
               if model_name == "KNN" else
               RandomForestRegressor(n_estimators=80, max_depth=4,
                                     min_samples_leaf=3, random_state=42, n_jobs=-1))
        m.fit(sc.transform(Xb), yb)
        boots.append(max(0.0, float(m.predict(sc.transform(X_new))[0])))

    bp = np.array(boots)
    return float(bp.mean()), float(np.percentile(bp, 5)), float(np.percentile(bp, 95))


# ── PCA projection ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def pca_project(n_aug: int, noise_scale: float, seed: int) -> dict:
    df_aug = augment(n_aug, noise_scale, seed)
    Xp     = StandardScaler().fit_transform(df_aug[MINERAL_FEATURES].values)
    pca    = PCA(n_components=2)
    Xpc    = pca.fit_transform(Xp)
    return dict(coords=Xpc, var=pca.explained_variance_ratio_,
                source=df_aug["src"].values, monazite=df_aug["Monazite"].values)


# ── Hierarchical clustering ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cluster_stations(occ: float, n_clusters: int = 2, method: str = "ward") -> dict:
    basin = load_basin()
    m     = rad_compute(basin["monazite"].values, basin["k40"].values, occ)
    Xs    = StandardScaler().fit_transform(
        np.column_stack([basin["monazite"].values, m["a_th"], m["gamma"]]))
    Z     = linkage(Xs, method=method)
    cids  = fcluster(Z, t=n_clusters, criterion="maxclust")
    return dict(Z=Z, cluster_ids=cids, stations=basin["station"].tolist(),
                ra_eq=m["ra_eq"], gamma=m["gamma"], monazite=basin["monazite"].values)


# ── Spatial interpolation ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def spatial_grid(occ: float, resolution: int = 50) -> dict:
    basin = load_basin()
    m     = rad_compute(basin["monazite"].values, basin["k40"].values, occ)
    pts   = basin[["lat", "lon"]].values
    lat_g = np.linspace(pts[:,0].min()-.05, pts[:,0].max()+.05, resolution)
    lon_g = np.linspace(pts[:,1].min()-.05, pts[:,1].max()+.05, resolution)
    lg, llg = np.meshgrid(lat_g, lon_g)
    z     = np.clip(
        RBFInterpolator(pts, m["ra_eq"], kernel="thin_plate_spline", smoothing=5.0
                        )(np.column_stack([lg.ravel(), llg.ravel()])
                          ).reshape(resolution, resolution), 0.0, None)
    return dict(lat=lat_g, lon=lon_g, z=z)


# ── Anomaly detection ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def anomaly_scores(n_aug: int, noise_scale: float, seed: int,
                   contamination: float = 0.1) -> dict:
    df_aug = augment(n_aug, noise_scale, seed)
    irel   = load_irel()
    sc     = StandardScaler().fit(df_aug[MINERAL_FEATURES].values)
    iso    = IsolationForest(n_estimators=200, contamination=contamination,
                             random_state=42)
    iso.fit(sc.transform(df_aug[MINERAL_FEATURES].values))
    Xo     = sc.transform(irel[MINERAL_FEATURES].values)
    return dict(stations=irel["station"].tolist(),
                scores=iso.decision_function(Xo),
                labels=["Anomaly" if p==-1 else "Normal" for p in iso.predict(Xo)],
                monazite=irel["Monazite"].values)


# ── Permutation importance ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def permutation_importance(n_aug: int, noise_scale: float, seed: int) -> dict:
    trained = train_models(n_aug, noise_scale, seed)
    knn, sc = trained["knn"], trained["scaler"]
    Xo, yo  = trained["X_orig"], trained["y_orig"]
    Xs      = sc.transform(Xo)
    base    = mean_absolute_error(yo, knn.predict(Xs))
    rng     = np.random.default_rng(seed + 7)
    imp = {}
    for j, feat in enumerate(MINERAL_FEATURES):
        Xp       = Xs.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp[feat] = mean_absolute_error(yo, knn.predict(Xp)) - base
    return dict(features=list(imp.keys()), importance=list(imp.values()))


# ── Regression helper ──────────────────────────────────────────────────────────
def fit_regression(x_vals, y_vals, model_type: str = "OLS"):
    X = np.asarray(x_vals).reshape(-1, 1)
    y = np.asarray(y_vals)
    mdl = {"OLS": LinearRegression(), "Ridge (α=0.1)": Ridge(alpha=0.1),
           "Lasso (α=0.01)": Lasso(alpha=0.01)}.get(model_type, LinearRegression())
    mdl.fit(X, y)
    y_hat = mdl.predict(X)
    slope = float(mdl.coef_[0] if hasattr(mdl.coef_, "__len__") else mdl.coef_)
    return mdl, r2_score(y, y_hat), slope, float(mdl.intercept_), y_hat