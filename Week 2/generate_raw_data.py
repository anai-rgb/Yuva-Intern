"""
generate_raw_data.py
---------------------
Simulates a raw shipment-level logistics dataset, modeled on the structure of
publicly available logistics/supply-chain datasets such as:
  - DataCo Smart Supply Chain for Big Data Analysis (Kaggle)
  - Kaggle "Logistics and Supply Chain" shipment datasets
  - US DOT / BTS Freight Analysis Framework (FAF) shipment records

The generator intentionally injects the kinds of data-quality problems that
show up in real-world logistics extracts: missing values, duplicate rows,
inconsistent categorical labels, outliers, and mixed date formats. This raw
file is the INPUT to the cleaning pipeline (clean_pipeline.py).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 1000

carriers_clean = ["FedEx", "UPS", "DHL", "Blue Dart", "Delhivery"]
# inconsistent/dirty variants that a real system export would contain
carrier_variants = {
    "FedEx": ["FedEx", "fedex", "FEDEX", "Fed Ex"],
    "UPS": ["UPS", "ups", "U.P.S."],
    "DHL": ["DHL", "dhl", "D.H.L"],
    "Blue Dart": ["Blue Dart", "BlueDart", "blue dart"],
    "Delhivery": ["Delhivery", "delhivery", "DELHIVERY"],
}

cities = ["Mumbai", "Delhi", "Nagpur", "Bengaluru", "Chennai", "Kolkata",
          "Pune", "Hyderabad", "Ahmedabad", "Jaipur"]

statuses = ["Delivered", "In Transit", "Delayed", "Cancelled", "Returned"]

rows = []
for i in range(N):
    shipment_id = f"SHP{10000 + i}"
    origin = rng.choice(cities)
    destination = rng.choice([c for c in cities if c != origin])

    # base distance correlated to a synthetic "route length"
    distance_km = rng.normal(650, 300)
    distance_km = max(distance_km, 15)

    weight_kg = rng.gamma(shape=2.0, scale=8.0)  # right-skewed, like real freight weights

    carrier_clean = rng.choice(carriers_clean)
    carrier_raw = rng.choice(carrier_variants[carrier_clean])

    # delivery time loosely tied to distance, with noise
    base_days = distance_km / 250 + rng.normal(0, 1.0)
    delivery_time_days = max(base_days, 0.5)

    cost_inr = distance_km * rng.uniform(3.5, 6.0) + weight_kg * rng.uniform(10, 25)

    status = rng.choice(statuses, p=[0.62, 0.15, 0.12, 0.06, 0.05])

    # mixed date formats to simulate multi-source extraction
    date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
    date_fmt = rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"])
    date_str = date.strftime(date_fmt)

    rows.append({
        "shipment_id": shipment_id,
        "order_date": date_str,
        "origin_city": origin,
        "destination_city": destination,
        "carrier": carrier_raw,
        "distance_km": round(distance_km, 2),
        "weight_kg": round(weight_kg, 2),
        "delivery_time_days": round(delivery_time_days, 2),
        "shipping_cost_inr": round(cost_inr, 2),
        "status": status,
    })

df = pd.DataFrame(rows)

# --- inject missing values (MCAR-style, ~ various rates per column) ---
def inject_missing(col, frac):
    idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

inject_missing("weight_kg", 0.06)
inject_missing("shipping_cost_inr", 0.04)
inject_missing("delivery_time_days", 0.05)
inject_missing("carrier", 0.02)
inject_missing("destination_city", 0.01)

# --- inject outliers ---
outlier_idx = rng.choice(df.index, size=8, replace=False)
df.loc[outlier_idx[:4], "distance_km"] = df.loc[outlier_idx[:4], "distance_km"] * rng.uniform(6, 9)
df.loc[outlier_idx[4:], "shipping_cost_inr"] = df.loc[outlier_idx[4:], "shipping_cost_inr"] * rng.uniform(8, 12)

# a few impossible / negative values (data entry errors)
bad_idx = rng.choice(df.index, size=5, replace=False)
df.loc[bad_idx, "weight_kg"] = -df.loc[bad_idx, "weight_kg"]

# --- inject duplicate rows (system re-sync / re-scrape artifact) ---
dupes = df.sample(15, random_state=7)
df = pd.concat([df, dupes], ignore_index=True)

# shuffle rows to look like a raw export
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv("raw_shipments.csv", index=False)
print("Raw dataset generated:", df.shape)
print(df.head(8).to_string())
print("\nMissing values per column:\n", df.isna().sum())
print("\nDuplicate rows (exact):", df.duplicated().sum())
