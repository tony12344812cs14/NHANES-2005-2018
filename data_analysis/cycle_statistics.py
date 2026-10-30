
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs/features/features_2005_2018.csv.gz"
OUTPUT_DIR = ROOT / "outputs/data_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load final dataset
df = pd.read_csv(
    DATA_PATH,
    usecols=["cycle", "depression"]
)

# Validate labels
assert df["depression"].notna().all(), "Missing target labels"
assert df["depression"].isin([0, 1]).all(), "Invalid target labels"

# Calculate per-cycle statistics
summary = (
    df.groupby("cycle", sort=True)["depression"]
    .agg(
        Total="count",
        Depression="sum"
    )
    .reset_index()
)

summary["Depression"] = summary["Depression"].astype(int)
summary["Non_Depression"] = (
    summary["Total"] - summary["Depression"]
)

summary["Depression_Rate"] = (
    summary["Depression"] / summary["Total"] * 100
).round(2)

# Overall statistics
total = len(df)
positive = int(df["depression"].sum())

overall = pd.DataFrame([{
    "cycle": "Overall",
    "Total": total,
    "Depression": positive,
    "Non_Depression": total - positive,
    "Depression_Rate": round(positive / total * 100, 2)
}])

result = pd.concat([summary, overall], ignore_index=True)

# Export results
output_path = OUTPUT_DIR / "cycle_statistics.csv"
result.to_csv(output_path, index=False)

print(result.to_string(index=False))
print(f"\nSaved to: {output_path}")
