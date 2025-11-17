import csv
import os

here = os.path.dirname(__file__)

# Input and output CSV paths
input_path = os.path.join(here, "robust_identity_runs_fixed.csv")
output_path = os.path.join(here, "robust_identity_runs_len_normalized.csv")

A_LEN_COL = "A_len"
B_LEN_COL = "B_len"

def main():
    # First pass: read all rows and collect length values
    rows = []
    all_lengths = []

    with open(input_path, "r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError("Input CSV has no header row.")

        if A_LEN_COL not in fieldnames or B_LEN_COL not in fieldnames:
            raise RuntimeError(
                f"Input CSV must contain columns {A_LEN_COL!r} and {B_LEN_COL!r}. "
                f"Found columns: {fieldnames}"
            )

        for row in reader:
            rows.append(row)

            # Parse lengths as floats
            try:
                a_len = float(row[A_LEN_COL])
                b_len = float(row[B_LEN_COL])
            except (KeyError, ValueError) as e:
                raise RuntimeError(f"Non-numeric A_len/B_len in row {row}: {e}")

            all_lengths.append(a_len)
            all_lengths.append(b_len)

    if not all_lengths:
        raise RuntimeError("No length values found in CSV.")

    min_len = min(all_lengths)
    max_len = max(all_lengths)

    print(f"Min length across A_len/B_len: {min_len}")
    print(f"Max length across A_len/B_len: {max_len}")

    # Avoid division by zero if all lengths are identical
    denom = max_len - min_len
    if denom == 0:
        print(
            "Warning: All lengths are identical. "
            "Setting all normalized lengths to 0.0."
        )

    # Second pass: write new CSV with normalized lengths
    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            a_len = float(row[A_LEN_COL])
            b_len = float(row[B_LEN_COL])

            if denom == 0:
                a_norm = 0.0
                b_norm = 0.0
            else:
                a_norm = (a_len - min_len) / denom
                b_norm = (b_len - min_len) / denom

            row[A_LEN_COL] = f"{a_norm:.6f}"
            row[B_LEN_COL] = f"{b_norm:.6f}"

            writer.writerow(row)

    print(f"Normalized CSV written to: {output_path}")

if __name__ == "__main__":
    main()
