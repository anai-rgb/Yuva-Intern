"""
Week 3 Task - Step 2: Exploratory Data Analysis (EDA)
Computes central tendencies, spread, distributions, and correlations
for the simulated logistics dataset.
"""

import pandas as pd

pd.set_option("display.width", 120)
df = pd.read_csv("/home/claude/logistics/logistics_data.csv", parse_dates=["ship_date"])

numeric_cols = ["distance_km", "shipment_volume_kg", "delivery_time_hours",
                 "transport_cost_usd", "cost_per_kg", "customer_satisfaction"]

print("=== Descriptive statistics ===")
desc = df[numeric_cols].describe().T
desc["skew"] = df[numeric_cols].skew()
print(desc.round(2))

print("\n=== On-time delivery rate (overall) ===")
print(f"{df['on_time'].mean()*100:.1f}%")

print("\n=== On-time delivery rate by transport mode ===")
print((df.groupby("transport_mode")["on_time"].mean() * 100).round(1))

print("\n=== On-time delivery rate by region ===")
print((df.groupby("region")["on_time"].mean() * 100).round(1))

print("\n=== Average cost per kg by transport mode ===")
print(df.groupby("transport_mode")["cost_per_kg"].mean().round(3))

print("\n=== Correlation matrix ===")
corr = df[numeric_cols].corr()
print(corr.round(2))

print("\n=== Monthly shipment volume & avg delivery time ===")
monthly = df.groupby("month").agg(
    shipments=("shipment_id", "count"),
    avg_delivery_hours=("delivery_time_hours", "mean"),
    avg_cost=("transport_cost_usd", "mean"),
    on_time_rate=("on_time", "mean"),
).round(2)
print(monthly)

desc.round(2).to_csv("/home/claude/logistics/summary_stats.csv")
corr.round(2).to_csv("/home/claude/logistics/correlation_matrix.csv")
monthly.to_csv("/home/claude/logistics/monthly_summary.csv")
