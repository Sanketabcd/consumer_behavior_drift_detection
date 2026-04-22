"""
prediction_engine.py  —  Random Forest purchase amount predictor
Returns prediction + full explainability:
  - Feature importances
  - Per-category and per-payment average spend from baseline
  - Confidence intervals via tree variance
  - How the selected combo compares to all other combos
"""

import pandas as pd
import numpy as np
from sklearn.ensemble         import RandomForestRegressor
from sklearn.preprocessing    import OneHotEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
import warnings
warnings.filterwarnings("ignore")


def train_and_predict(baseline_df: pd.DataFrame,
                      user_category: str,
                      user_payment: str):
    """
    Train RF on baseline, predict purchase amount, return rich explainability.

    Returns
    -------
    predicted_amt : float
    expected_std  : float
    explain       : dict with full breakdown for the UI
    """
    df = baseline_df.dropna(
        subset=["Purchase_Amount", "Product_Category", "Payment_Method"]
    )
    if len(df) < 5:
        return None, None, None

    X = df[["Product_Category", "Payment_Method"]]
    y = df["Purchase_Amount"]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["Product_Category", "Payment_Method"])
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10))
    ])
    model.fit(X, y)

    input_df = pd.DataFrame([{
        "Product_Category": user_category,
        "Payment_Method":   user_payment,
    }])
    predicted_amt = float(model.predict(input_df)[0])

    # ── Confidence interval via tree variance ─────────────────────────────────
    X_t = preprocessor.transform(input_df)
    tree_preds = np.array([
        tree.predict(X_t)[0]
        for tree in model.named_steps["regressor"].estimators_
    ])
    expected_std = float(np.std(tree_preds))
    ci_low  = float(np.percentile(tree_preds, 10))
    ci_high = float(np.percentile(tree_preds, 90))

    # ── Feature importances ───────────────────────────────────────────────────
    feat_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["regressor"].feature_importances_
    # Group by original column
    cat_imp = float(sum(
        imp for nm, imp in zip(feat_names, importances)
        if nm.startswith("cat__Product_Category")
    ))
    pay_imp = float(sum(
        imp for nm, imp in zip(feat_names, importances)
        if nm.startswith("cat__Payment_Method")
    ))

    # ── Per-category average in baseline ──────────────────────────────────────
    cat_avgs = (
        df.groupby("Product_Category")["Purchase_Amount"]
          .agg(["mean", "count", "std"])
          .rename(columns={"mean":"avg","count":"n","std":"sd"})
          .round(2)
    )
    cat_avgs["sd"] = cat_avgs["sd"].fillna(0)

    # ── Per-payment average in baseline ───────────────────────────────────────
    pay_avgs = (
        df.groupby("Payment_Method")["Purchase_Amount"]
          .agg(["mean", "count", "std"])
          .rename(columns={"mean":"avg","count":"n","std":"sd"})
          .round(2)
    )
    pay_avgs["sd"] = pay_avgs["sd"].fillna(0)

    # ── All combo predictions (for comparison table) ──────────────────────────
    all_cats  = df["Product_Category"].unique().tolist()
    all_pays  = df["Payment_Method"].unique().tolist()
    combos = pd.DataFrame(
        [{"Product_Category": c, "Payment_Method": p}
         for c in all_cats for p in all_pays]
    )
    combos["Predicted_Amount"] = model.predict(
        combos[["Product_Category", "Payment_Method"]]
    ).round(2)
    combos = combos.sort_values("Predicted_Amount", ascending=False).reset_index(drop=True)

    # ── Baseline stats for selected combo ─────────────────────────────────────
    combo_mask = (
        (df["Product_Category"] == user_category) &
        (df["Payment_Method"]   == user_payment)
    )
    combo_actual = df[combo_mask]["Purchase_Amount"]
    combo_actual_mean = float(combo_actual.mean()) if len(combo_actual) > 0 else None
    combo_n          = int(len(combo_actual))

    # ── Global baseline stats ─────────────────────────────────────────────────
    global_mean = float(df["Purchase_Amount"].mean())
    global_med  = float(df["Purchase_Amount"].median())

    # ── Rank of this prediction among all combos ──────────────────────────────
    rank = int(
        combos[
            (combos["Product_Category"] == user_category) &
            (combos["Payment_Method"]   == user_payment)
        ].index[0]
    ) + 1 if len(combos) > 0 else 1
    total_combos = len(combos)

    explain = {
        "predicted_amt":    round(predicted_amt, 2),
        "ci_low":           round(ci_low, 2),
        "ci_high":          round(ci_high, 2),
        "expected_std":     round(expected_std, 2),
        "cat_importance":   round(cat_imp * 100, 1),
        "pay_importance":   round(pay_imp * 100, 1),
        "cat_avgs":         cat_avgs,
        "pay_avgs":         pay_avgs,
        "all_combos":       combos,
        "combo_actual_mean":combo_actual_mean,
        "combo_n":          combo_n,
        "global_mean":      round(global_mean, 2),
        "global_median":    round(global_med, 2),
        "rank":             rank,
        "total_combos":     total_combos,
        "user_category":    user_category,
        "user_payment":     user_payment,
        "n_trees":          100,
        "n_training_rows":  len(df),
    }

    return predicted_amt, expected_std, explain