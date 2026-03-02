import os
import csv
import numpy as np

here = os.path.dirname(__file__)
csv_path = os.path.join(here, "robust_identity_runs_len_normalized.csv")

# 8 numeric features
feature_names = [
    "A_len",
    "B_len",
    "A_popularity",
    "B_popularity",
    "A_abstraction",
    "B_abstraction",
    "A_polysemy",
    "B_polysemy",
]

# -------------------------------
# LOAD CSV
# -------------------------------
X_rows = []
y_rows = []

with open(csv_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader, start=1):
        try:
            features = [float(row[name]) for name in feature_names]
            label = float(row["identity_success"])
        except (ValueError, KeyError):
            continue

        X_rows.append(features)
        y_rows.append(label)

if not X_rows:
    raise RuntimeError("No usable rows found in CSV.")

X = np.array(X_rows, dtype=float)
y = np.array(y_rows, dtype=float)

# -------------------------------
# STANDARDIZE FEATURES (z-score)
# -------------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)

# Avoid division by zero
X_std_safe = np.where(X_std == 0, 1, X_std)

X_z = (X - X_mean) / X_std_safe

# -------------------------------
# STANDARDIZE TARGET
# -------------------------------
y_mean = y.mean()
y_std = y.std()

y_z = (y - y_mean) / (y_std if y_std != 0 else 1)

# -------------------------------
# RUN REGRESSION ON Z-SCORES
# -------------------------------
ones = np.ones((X_z.shape[0], 1))
X_design = np.hstack([ones, X_z])    # includes intercept

beta_z, residuals, rank, s = np.linalg.lstsq(X_design, y_z, rcond=None)

intercept_z = beta_z[0]
coefs_z = beta_z[1:]

# -------------------------------
# PRINT RESULTS
# -------------------------------
print("\n========== STANDARDIZED LINEAR REGRESSION ==========\n")
print(f"Samples:             {X.shape[0]}")
print(f"Features:            {X.shape[1]}")
print("Target & features z-scored (mean=0, std=1)\n")

print("Standardized coefficients (directly comparable):")
for name, coef in zip(feature_names, coefs_z):
    print(f"  {name:15s} -> {coef:+.4f}")

print(f"\nIntercept (std space): {intercept_z:+.4f}")
print("\n====================================================\n")

# -------------------------------
# OPTIONAL: ALSO PRINT RAW COEFS
# -------------------------------
raw_beta, _, _, _ = np.linalg.lstsq(
    np.hstack([np.ones((X.shape[0], 1)), X]), y, rcond=None
)

print("Unstandardized coefficients (reference only):")
for name, coef in zip(feature_names, raw_beta[1:]):
    print(f"  {name:15s} -> {coef:+.4f}")

print(f"\nUnstd intercept: {raw_beta[0]:+.4f}")
print("\n====================================================\n")
