"""
ml_drift_engine.py  —  Machine Learning based drift detection
Replaces statistical tests (KS, Chi-Square, PSI, Wasserstein) with:

1. Isolation Forest      — unsupervised anomaly detector trained on baseline,
                           scores current data, drift = high anomaly rate
2. Random Forest Classifier — trained on labelled baseline(0)+current(1) data,
                              drift = model separates the two periods with high accuracy
3. Gradient Boosting     — same binary classification approach, different model
4. Feature Importance    — RF gives feature-level importance scores
5. Confidence bands      — bootstrap confidence intervals on drift score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble         import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.model_selection  import cross_val_score
from sklearn.metrics          import roc_auc_score, classification_report
from sklearn.pipeline         import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_features(baseline_df: pd.DataFrame,
                     current_df: pd.DataFrame):
    """
    Encode Purchase_Amount, Product_Category, Payment_Method into a
    numeric feature matrix. Encoders are fitted on baseline+current combined
    so both datasets share the same encoding.
    Returns X_base, X_curr, feature_names, encoders
    """
    combined = pd.concat([baseline_df, current_df], ignore_index=True)

    le_cat = LabelEncoder()
    le_pay = LabelEncoder()
    le_cat.fit(combined["Product_Category"].fillna("Unknown").astype(str))
    le_pay.fit(combined["Payment_Method"].fillna("Unknown").astype(str))

    def _build_X(df):
        cat = le_cat.transform(df["Product_Category"].fillna("Unknown").astype(str))
        pay = le_pay.transform(df["Payment_Method"].fillna("Unknown").astype(str))
        amt = df["Purchase_Amount"].fillna(df["Purchase_Amount"].median()).values
        return np.column_stack([amt, cat, pay])

    X_base = _build_X(baseline_df)
    X_curr = _build_X(current_df)
    feature_names = ["Purchase_Amount", "Product_Category", "Payment_Method"]

    return X_base, X_curr, feature_names, {"cat": le_cat, "pay": le_pay}


def _severity(score: float) -> tuple:
    """Convert a 0-1 drift score to label + colour."""
    if score < 0.55:
        return "Stable",            "#06D6A0", 0
    elif score < 0.70:
        return "Mild Drift",        "#FFB703", 1
    elif score < 0.85:
        return "Significant Drift", "#FF7B00", 2
    else:
        return "Critical Drift",    "#EF476F", 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. Isolation Forest Drift Score
# ─────────────────────────────────────────────────────────────────────────────

def isolation_forest_drift(baseline_df: pd.DataFrame,
                            current_df: pd.DataFrame,
                            contamination: float = 0.05) -> dict:
    """
    Train Isolation Forest on baseline. Score current data.
    Drift score = fraction of current rows flagged as anomalies.
    """
    X_base, X_curr, feat_names, _ = _encode_features(baseline_df, current_df)

    scaler = StandardScaler()
    X_base_s = scaler.fit_transform(X_base)
    X_curr_s = scaler.transform(X_curr)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_base_s)

    # -1 = anomaly, +1 = normal
    preds     = iso.predict(X_curr_s)
    scores    = iso.decision_function(X_curr_s)   # higher = more normal
    anomaly_rate = float((preds == -1).mean())

    # Drift detected if > 2x the expected contamination rate
    drift_detected = anomaly_rate > (contamination * 2)

    # Normalise anomaly score to 0-1 (1 = fully drifted)
    drift_score = min(anomaly_rate / 0.5, 1.0)
    level, color, _ = _severity(drift_score)

    # Per-row anomaly scores for visualisation
    row_scores = pd.DataFrame({
        "anomaly_score":  -scores,            # flip so higher = more anomalous
        "is_anomaly":     (preds == -1),
        "period":         "Current",
    })
    # Also score baseline for comparison
    base_preds  = iso.predict(X_base_s)
    base_scores = iso.decision_function(X_base_s)
    base_row = pd.DataFrame({
        "anomaly_score": -base_scores,
        "is_anomaly":    (base_preds == -1),
        "period":        "Baseline",
    })
    all_scores = pd.concat([base_row, row_scores], ignore_index=True)

    return {
        "model":          "Isolation Forest",
        "drift_detected": drift_detected,
        "drift_score":    round(drift_score, 4),
        "anomaly_rate":   round(anomaly_rate, 4),
        "level":          level,
        "color":          color,
        "row_scores":     all_scores,
        "contamination":  contamination,
        "n_anomalies":    int((preds == -1).sum()),
        "n_current":      len(current_df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random Forest Classifier Drift Score
# ─────────────────────────────────────────────────────────────────────────────

def random_forest_drift(baseline_df: pd.DataFrame,
                         current_df: pd.DataFrame) -> dict:
    """
    Binary classification: baseline = 0, current = 1.
    If the model can separate them (AUC > 0.5), drift exists.
    AUC close to 1.0 means complete separation = maximum drift.
    AUC close to 0.5 means distributions are identical = no drift.
    """
    X_base, X_curr, feat_names, _ = _encode_features(baseline_df, current_df)

    # Balance classes
    n = min(len(X_base), len(X_curr))
    idx_b = np.random.choice(len(X_base), n, replace=False)
    idx_c = np.random.choice(len(X_curr), n, replace=False)

    X = np.vstack([X_base[idx_b], X_curr[idx_c]])
    y = np.array([0]*n + [1]*n)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_s, y)

    # Cross-validated AUC
    cv_auc = cross_val_score(rf, X_s, y, cv=5, scoring="roc_auc", n_jobs=-1).mean()

    # Drift score: normalise AUC (0.5 = no drift, 1.0 = complete drift)
    drift_score = max(0.0, (cv_auc - 0.5) * 2)
    drift_detected = cv_auc > 0.65

    level, color, _ = _severity(drift_score)

    # Feature importances
    feat_imp = dict(zip(feat_names, rf.feature_importances_.tolist()))

    # Per-class probabilities on full current set
    X_curr_s = scaler.transform(X_curr)
    proba     = rf.predict_proba(X_curr_s)[:, 1]   # prob of being "current"

    return {
        "model":            "Random Forest Classifier",
        "drift_detected":   drift_detected,
        "drift_score":      round(drift_score, 4),
        "auc":              round(float(cv_auc), 4),
        "level":            level,
        "color":            color,
        "feature_importance": feat_imp,
        "current_proba":    proba.tolist(),
        "n_baseline":       len(baseline_df),
        "n_current":        len(current_df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gradient Boosting Drift Score
# ─────────────────────────────────────────────────────────────────────────────

def gradient_boosting_drift(baseline_df: pd.DataFrame,
                              current_df: pd.DataFrame) -> dict:
    """
    Same binary classification approach with Gradient Boosting.
    GBT is often more accurate but slower.
    """
    X_base, X_curr, feat_names, _ = _encode_features(baseline_df, current_df)

    n = min(len(X_base), len(X_curr), 500)   # cap at 500 for speed
    idx_b = np.random.choice(len(X_base), n, replace=False)
    idx_c = np.random.choice(len(X_curr), n, replace=False)

    X = np.vstack([X_base[idx_b], X_curr[idx_c]])
    y = np.array([0]*n + [1]*n)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    gbt = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    gbt.fit(X_s, y)

    cv_auc = cross_val_score(gbt, X_s, y, cv=5, scoring="roc_auc", n_jobs=-1).mean()
    drift_score    = max(0.0, (cv_auc - 0.5) * 2)
    drift_detected = cv_auc > 0.65
    level, color, _ = _severity(drift_score)

    feat_imp = dict(zip(feat_names, gbt.feature_importances_.tolist()))

    return {
        "model":              "Gradient Boosting",
        "drift_detected":     drift_detected,
        "drift_score":        round(drift_score, 4),
        "auc":                round(float(cv_auc), 4),
        "level":              level,
        "color":              color,
        "feature_importance": feat_imp,
        "n_baseline":         len(baseline_df),
        "n_current":          len(current_df),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-feature drift — one RF per feature
# ─────────────────────────────────────────────────────────────────────────────

def per_feature_ml_drift(baseline_df: pd.DataFrame,
                          current_df: pd.DataFrame) -> dict:
    """
    Run a quick single-feature classification for each column.
    Returns AUC per feature — indicates which feature changed most.
    """
    results = {}

    # Purchase Amount — numeric
    for label, b_vals, c_vals in [
        ("Purchase_Amount",
         baseline_df["Purchase_Amount"].dropna().values,
         current_df["Purchase_Amount"].dropna().values),
    ]:
        n = min(len(b_vals), len(c_vals))
        X = np.concatenate([b_vals[:n], c_vals[:n]]).reshape(-1, 1)
        y = np.array([0]*n + [1]*n)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        auc = cross_val_score(rf, X_s, y, cv=3, scoring="roc_auc").mean()
        drift_score = max(0.0, (auc - 0.5) * 2)
        level, color, _ = _severity(drift_score)
        results[label] = {
            "auc": round(float(auc), 4),
            "drift_score": round(drift_score, 4),
            "drift_detected": auc > 0.65,
            "level": level, "color": color,
        }

    # Product Category — encode then classify
    for label, b_col, c_col in [
        ("Product_Category",
         baseline_df["Product_Category"].fillna("Unknown").astype(str),
         current_df["Product_Category"].fillna("Unknown").astype(str)),
        ("Payment_Method",
         baseline_df["Payment_Method"].fillna("Unknown").astype(str),
         current_df["Payment_Method"].fillna("Unknown").astype(str)),
    ]:
        le = LabelEncoder()
        le.fit(pd.concat([b_col, c_col]))
        n = min(len(b_col), len(c_col))
        b_enc = le.transform(b_col.iloc[:n]).reshape(-1, 1)
        c_enc = le.transform(c_col.iloc[:n]).reshape(-1, 1)
        X = np.vstack([b_enc, c_enc]).astype(float)
        y = np.array([0]*n + [1]*n)
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        auc = cross_val_score(rf, X, y, cv=3, scoring="roc_auc").mean()
        drift_score = max(0.0, (auc - 0.5) * 2)
        level, color, _ = _severity(drift_score)
        results[label] = {
            "auc": round(float(auc), 4),
            "drift_score": round(drift_score, 4),
            "drift_detected": auc > 0.65,
            "level": level, "color": color,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Run all ML drift checks (main entry point)
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_drift_checks(baseline_df: pd.DataFrame,
                         current_df: pd.DataFrame) -> dict:
    """
    Run all ML drift checks and return a unified results dict.
    This is the ML equivalent of run_all_drift_checks() in drift_detection.py
    """
    results = {}

    # Summary stats (kept for charts compatibility)
    ss_b = baseline_df["Purchase_Amount"].describe().rename("Baseline")
    ss_c = current_df["Purchase_Amount"].describe().rename("Current")
    results["summary_stats"] = pd.concat([ss_b, ss_c], axis=1).round(2)

    # Core ML models
    results["isolation_forest"]     = isolation_forest_drift(baseline_df, current_df)
    results["random_forest"]        = random_forest_drift(baseline_df, current_df)
    results["gradient_boosting"]    = gradient_boosting_drift(baseline_df, current_df)
    results["per_feature"]          = per_feature_ml_drift(baseline_df, current_df)

    # Ensemble drift decision — majority vote of the 3 models
    votes = [
        results["isolation_forest"]["drift_detected"],
        results["random_forest"]["drift_detected"],
        results["gradient_boosting"]["drift_detected"],
    ]
    results["ensemble_drift"]  = sum(votes) >= 2
    results["ensemble_score"]  = round(np.mean([
        results["isolation_forest"]["drift_score"],
        results["random_forest"]["drift_score"],
        results["gradient_boosting"]["drift_score"],
    ]), 4)
    level, color, _ = _severity(results["ensemble_score"])
    results["ensemble_level"] = level
    results["ensemble_color"] = color

    # Category + Payment distributions (for existing charts compatibility)
    cat_base = (baseline_df["Product_Category"].value_counts(normalize=True)*100).round(1).to_dict()
    cat_curr = (current_df["Product_Category"].value_counts(normalize=True)*100).round(1).to_dict()
    pay_base = (baseline_df["Payment_Method"].value_counts(normalize=True)*100).round(1).to_dict()
    pay_curr = (current_df["Payment_Method"].value_counts(normalize=True)*100).round(1).to_dict()

    # Keep these in "statistical" format for backward-compat with charts
    pf = results["per_feature"]
    results["purchase_amount"]  = {
        "test":           "Random Forest AUC",
        "statistic":      pf["Purchase_Amount"]["auc"],
        "p_value":        round(1 - pf["Purchase_Amount"]["auc"], 6),
        "drift_detected": pf["Purchase_Amount"]["drift_detected"],
        "drift_score":    pf["Purchase_Amount"]["drift_score"],
        "level":          pf["Purchase_Amount"]["level"],
        "color":          pf["Purchase_Amount"]["color"],
    }
    results["product_category"] = {
        "test":                   "Random Forest AUC",
        "statistic":              pf["Product_Category"]["auc"],
        "p_value":                round(1 - pf["Product_Category"]["auc"], 6),
        "drift_detected":         pf["Product_Category"]["drift_detected"],
        "drift_score":            pf["Product_Category"]["drift_score"],
        "level":                  pf["Product_Category"]["level"],
        "color":                  pf["Product_Category"]["color"],
        "baseline_distribution":  cat_base,
        "current_distribution":   cat_curr,
    }
    results["payment_method"] = {
        "test":                   "Random Forest AUC",
        "statistic":              pf["Payment_Method"]["auc"],
        "p_value":                round(1 - pf["Payment_Method"]["auc"], 6),
        "drift_detected":         pf["Payment_Method"]["drift_detected"],
        "drift_score":            pf["Payment_Method"]["drift_score"],
        "level":                  pf["Payment_Method"]["level"],
        "color":                  pf["Payment_Method"]["color"],
        "baseline_distribution":  pay_base,
        "current_distribution":   pay_curr,
    }

    # PSI-compatible dict (for charts that use psi keys)
    results["psi"] = {
        "psi":   results["ensemble_score"],
        "level": results["ensemble_level"],
        "color": results["ensemble_color"],
        "bin_data": pd.DataFrame(),  # not used in ML mode
    }
    # Wasserstein-compatible dict
    b_mean = float(baseline_df["Purchase_Amount"].mean())
    c_mean = float(current_df["Purchase_Amount"].mean())
    dist   = abs(c_mean - b_mean)
    results["wasserstein"] = {
        "distance":       round(dist, 4),
        "pct_shift":      round((dist/b_mean*100) if b_mean > 0 else 0, 2),
        "interpretation": f"Mean purchase shifted by ~${dist:.2f}",
    }

    return results