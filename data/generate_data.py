"""
generate_data.py
----------------
This script generates a synthetic e-commerce dataset that simulates
consumer purchasing behavior — including a deliberate "drift" in the
second half of the data so our detection algorithms have something to find.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────────
N_BASELINE = 1000   # Number of transactions in the baseline period
N_CURRENT  = 1000   # Number of transactions in the current period

START_DATE_BASELINE = datetime(2023, 1, 1)
START_DATE_CURRENT  = datetime(2023, 7, 1)


def generate_baseline():
    """
    Generate the BASELINE period dataset.
    This represents 'normal' consumer behaviour that we use as our reference.
    """
    dates = [START_DATE_BASELINE + timedelta(days=np.random.randint(0, 180))
             for _ in range(N_BASELINE)]

    customer_ids = [f"C{np.random.randint(1000, 5000):04d}" for _ in range(N_BASELINE)]

    # Category distribution for the baseline period
    categories = np.random.choice(
        ["Electronics", "Clothing", "Groceries", "Books", "Home & Garden"],
        size=N_BASELINE,
        p=[0.30, 0.25, 0.20, 0.15, 0.10]   # ← baseline probabilities
    )

    # Purchase amounts follow a log-normal distribution (realistic for retail)
    purchase_amounts = np.round(np.random.lognormal(mean=4.0, sigma=0.8, size=N_BASELINE), 2)

    payment_methods = np.random.choice(
        ["Credit Card", "Debit Card", "PayPal", "Cash"],
        size=N_BASELINE,
        p=[0.40, 0.30, 0.20, 0.10]
    )

    df = pd.DataFrame({
        "Date":            dates,
        "Customer_ID":     customer_ids,
        "Product_Category": categories,
        "Purchase_Amount": purchase_amounts,
        "Payment_Method":  payment_methods,
        "Period":          "Baseline"
    })
    return df.sort_values("Date").reset_index(drop=True)


def generate_current():
    """
    Generate the CURRENT period dataset.
    We intentionally introduce DRIFT here:
      - Purchase amounts are significantly higher (mean shift).
      - Category distribution has changed (Electronics surges, Books drop).
      - Payment method mix has shifted toward digital wallets.
    """
    dates = [START_DATE_CURRENT + timedelta(days=np.random.randint(0, 180))
             for _ in range(N_CURRENT)]

    customer_ids = [f"C{np.random.randint(1000, 5000):04d}" for _ in range(N_CURRENT)]

    # ← DRIFT: Electronics now dominates; Books almost disappear
    categories = np.random.choice(
        ["Electronics", "Clothing", "Groceries", "Books", "Home & Garden"],
        size=N_CURRENT,
        p=[0.50, 0.20, 0.15, 0.05, 0.10]
    )

    # ← DRIFT: Higher average spend (mean shifted from 4.0 → 4.6)
    purchase_amounts = np.round(np.random.lognormal(mean=4.6, sigma=0.9, size=N_CURRENT), 2)

    # ← DRIFT: PayPal usage increases
    payment_methods = np.random.choice(
        ["Credit Card", "Debit Card", "PayPal", "Cash"],
        size=N_CURRENT,
        p=[0.35, 0.25, 0.35, 0.05]
    )

    df = pd.DataFrame({
        "Date":            dates,
        "Customer_ID":     customer_ids,
        "Product_Category": categories,
        "Purchase_Amount": purchase_amounts,
        "Payment_Method":  payment_methods,
        "Period":          "Current"
    })
    return df.sort_values("Date").reset_index(drop=True)


def main():
    baseline_df = generate_baseline()
    current_df  = generate_current()

    # Save individual period files
    out_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_df.to_csv(os.path.join(out_dir, "baseline_data.csv"), index=False)
    current_df.to_csv(os.path.join(out_dir, "current_data.csv"),  index=False)

    # Also save a combined file (handy for time-series views)
    combined = pd.concat([baseline_df, current_df], ignore_index=True)
    combined.to_csv(os.path.join(out_dir, "combined_data.csv"), index=False)

    print(f"✅ Baseline rows : {len(baseline_df)}")
    print(f"✅ Current rows  : {len(current_df)}")
    print(f"✅ Combined rows : {len(combined)}")
    print("Data saved to data/ folder.")


if __name__ == "__main__":
    main()
    
    
