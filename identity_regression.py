import os
import csv
import numpy as np

here = os.path.dirname(__file__)
csv_path = os.path.join(here, "robust_identity_runs.csv")

# Use lowercase feature names as our canonical keys
feature_names = [
    "a_popularity",
    "b_popularity",
    "a_abstraction",
    "b_abstraction",
    "a_polysemy",
    "b_polysemy",
]

X_rows = []
y_rows = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print("Raw fieldnames:", reader.fieldnames)

    for i, raw_row in enumerate(reader, start=1):
        # Normalize keys: strip spaces and lowercase
        row = {}
        for k, v in raw_row.items():
            if k is None:
                continue
            norm_k = k.strip().lower()
            row[norm_k] = v

        try:
            features = []
            for name in feature_names:
                val = row.get(name, "")
                if val is None:
                    raise ValueError(f"{name} is None")
                val = val.strip()
                if val == "":
                    raise ValueError(f"{name} is empty")
                features.append(float(val))

            # Normalize identity_success column name as well
            label_str = (
                row.get("identity_success")
                or row.get("identity_success ".strip())
                or row.get("identity success")  # just in case
            )
            if label_str is None:
                raise KeyError("identity_success column not found (even after normalization)")
            label_str = label_str.strip()

            # Be forgiving about formats and capitalization
            ls = label_str.lower()
            if ls in ("true", "yes"):
                label = 1
            elif ls in ("false", "no"):
                label = 0
            else:
                label = int(float(label_str))

        except (KeyError, ValueError) as e:
            if i <= 10:
                print(f"Skipping row {i}: {e}. Raw row={raw_row}")
            continue

        X_rows.append(features)
        y_rows.append(label)

if not X_rows:
    raise RuntimeError("No valid data rows found in CSV after normalization. Check header capitalization/spacing and data values.")

X = np.array(X_rows)
y = np.array(y_rows, dtype=float)

ones = np.ones((X.shape[0], 1), dtype=float)
X_design = np.hstack([ones, X])

beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)

intercept = beta[0]
coefs = beta[1:]

y_pred = X_design @ beta
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

print("\n========== LINEAR REGRESSION (NUMPY) ==========\n")
print(f"Number of samples:   {X.shape[0]}")
print(f"Number of features:  {X.shape[1]}")
print(f"R² score:            {r2:.4f}\n")

print("Coefficients:")
for name, coef in zip(feature_names, coefs):
    print(f"  {name:15s} -> {coef:+.4f}")

print(f"\nIntercept: {intercept:+.4f}")
print("\n===============================================\n")
