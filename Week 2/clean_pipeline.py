"""
clean_pipeline.py
------------------
End-to-end cleaning & preprocessing pipeline for the raw shipment dataset
produced by generate_raw_data.py. Mirrors a realistic logistics-analytics
preprocessing workflow: standardize -> deduplicate -> handle missing values
-> fix invalid values -> treat outliers -> engineer features -> normalize.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 120)

df = pd.read_csv("raw_shipments.csv")
print("STEP 0 | Raw shape:", df.shape)

# ---------------------------------------------------------------
# STEP 1: Standardize column values (categorical inconsistency)
# ---------------------------------------------------------------
carrier_map = {
    "fedex": "FedEx", "fed ex": "FedEx", "FEDEX": "FedEx",
    "ups": "UPS", "u.p.s.": "UPS",
    "dhl": "DHL", "d.h.l": "DHL",
    "bluedart": "Blue Dart", "blue dart": "Blue Dart",
    "delhivery": "Delhivery",
}
def normalize_carrier(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    return carrier_map.get(x.lower(), x)

df["carrier"] = df["carrier"].apply(normalize_carrier)

# Standardize mixed date formats to ISO 8601
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", format="mixed")

print("\nSTEP 1 | Standardized carrier values:", sorted(df['carrier'].dropna().unique()))

# ---------------------------------------------------------------
# STEP 2: Remove duplicate records
# ---------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "shipment_id"], keep="first")
# also drop exact duplicate shipment_ids (same record re-scraped)
df = df.drop_duplicates(subset="shipment_id", keep="first")
print(f"\nSTEP 2 | Removed {before - len(df)} duplicate rows -> shape {df.shape}")

# ---------------------------------------------------------------
# STEP 3: Fix invalid / impossible values
# ---------------------------------------------------------------
neg_weight_count = (df["weight_kg"] < 0).sum()
df["weight_kg"] = df["weight_kg"].abs()  # negative weight is a sign-entry error, not a true negative
print(f"\nSTEP 3 | Corrected {neg_weight_count} negative weight_kg entries (sign error)")

# ---------------------------------------------------------------
# STEP 4: Handle missing values
# ---------------------------------------------------------------
missing_before = df.isna().sum()

# Numeric columns: median imputation (robust to skew/outliers vs. mean)
for col in ["weight_kg", "delivery_time_days", "shipping_cost_inr"]:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Categorical columns: mode imputation for low-missing-rate categoricals
for col in ["carrier", "destination_city"]:
    mode_val = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(mode_val)

missing_after = df.isna().sum()
print("\nSTEP 4 | Missing values BEFORE imputation:\n", missing_before[missing_before > 0])
print("\nSTEP 4 | Missing values AFTER imputation:\n", missing_after.sum(), "total remaining")

# ---------------------------------------------------------------
# STEP 5: Outlier detection & treatment (IQR method)
# ---------------------------------------------------------------
def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

outlier_report = {}
for col in ["distance_km", "shipping_cost_inr", "weight_kg", "delivery_time_days"]:
    low, high = iqr_bounds(df[col])
    n_outliers = ((df[col] < low) | (df[col] > high)).sum()
    outlier_report[col] = {"lower": round(low, 2), "upper": round(high, 2), "count": int(n_outliers)}
    # cap (winsorize) rather than delete, to preserve sample size for a business dataset
    df[col] = df[col].clip(lower=low, upper=high)

print("\nSTEP 5 | Outlier bounds (IQR) and counts capped:")
for col, r in outlier_report.items():
    print(f"  {col}: bounds=({r['lower']}, {r['upper']}), outliers_capped={r['count']}")

# ---------------------------------------------------------------
# STEP 6: Feature engineering
# ---------------------------------------------------------------
df["cost_per_km"] = (df["shipping_cost_inr"] / df["distance_km"]).round(2)
df["order_month"] = df["order_date"].dt.month
df["is_delayed"] = (df["status"].isin(["Delayed", "Returned"])).astype(int)

# ---------------------------------------------------------------
# STEP 7: Normalization / scaling
# ---------------------------------------------------------------
# Min-Max normalization (0-1) for distance/weight/cost -> comparable scale for
# models sensitive to magnitude (e.g., k-NN, clustering, neural nets)
minmax_cols = ["distance_km", "weight_kg", "shipping_cost_inr", "delivery_time_days"]
for col in minmax_cols:
    mn, mx = df[col].min(), df[col].max()
    df[f"{col}_minmax"] = ((df[col] - mn) / (mx - mn)).round(4)

# Z-score standardization (mean 0, std 1) -> useful for regression-type models
for col in minmax_cols:
    mu, sigma = df[col].mean(), df[col].std()
    df[f"{col}_zscore"] = ((df[col] - mu) / sigma).round(4)

print("\nSTEP 7 | Sample of normalized columns:")
print(df[["distance_km", "distance_km_minmax", "distance_km_zscore"]].head(5).to_string())

# ---------------------------------------------------------------
# STEP 8: Final validation & export
# ---------------------------------------------------------------
final_missing = df.isna().sum().sum()
final_dupes = df.duplicated(subset="shipment_id").sum()
print(f"\nSTEP 8 | Final validation -> missing values: {final_missing}, duplicate IDs: {final_dupes}")
print("Final cleaned shape:", df.shape)

df.to_csv("cleaned_shipments.csv", index=False)

summary = {
    "raw_rows": 1015,
    "cleaned_rows": len(df),
    "duplicates_removed": before - len(df) if 'before' in dir() else None,
    "missing_values_imputed": int(missing_before.sum()),
    "outliers_capped": sum(r["count"] for r in outlier_report.values()),
    "negative_values_corrected": int(neg_weight_count),
}
print("\n=== PIPELINE SUMMARY ===")
for k, v in summary.items():
    print(f"{k}: {v}")
