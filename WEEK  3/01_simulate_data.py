"""
Week 3 Task - Step 1: Data Simulation
Simulates a hypothetical logistics dataset for a mid-size distribution
company operating across multiple regions and transport modes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # number of shipment records

regions = ["North", "South", "East", "West", "Central"]
region_probs = [0.22, 0.20, 0.18, 0.22, 0.18]

modes = ["Road", "Rail", "Air", "Sea"]
mode_probs = [0.55, 0.15, 0.10, 0.20]

# Base characteristics per transport mode (speed & cost profile)
mode_profile = {
    "Road": {"speed_kmph": 55, "cost_per_km": 0.9, "delay_std": 6},
    "Rail": {"speed_kmph": 65, "cost_per_km": 0.55, "delay_std": 10},
    "Air":  {"speed_kmph": 700, "cost_per_km": 4.2, "delay_std": 3},
    "Sea":  {"speed_kmph": 35, "cost_per_km": 0.25, "delay_std": 20},
}

dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

records = []
for i in range(1, N + 1):
    region = np.random.choice(regions, p=region_probs)
    mode = np.random.choice(modes, p=mode_probs)
    profile = mode_profile[mode]

    distance_km = np.clip(np.random.gamma(shape=4.0, scale=180) *
                           (3.0 if mode == "Air" else 1.0) *
                           (0.6 if mode == "Road" else 1.0), 20, 12000)

    shipment_volume_kg = np.clip(np.random.lognormal(mean=6.0, sigma=0.9), 5, 20000)

    # base delivery time from distance/speed + handling time + random ops delay
    base_hours = distance_km / profile["speed_kmph"]
    handling_hours = np.random.gamma(shape=2.0, scale=3.0)
    delay_hours = np.abs(np.random.normal(loc=profile["delay_std"] * 0.6,
                                           scale=profile["delay_std"]))
    delivery_time_hours = base_hours + handling_hours + delay_hours

    # transport cost: distance-based + volume surcharge + fuel volatility noise
    fuel_noise = np.random.normal(1.0, 0.08)
    transport_cost = (distance_km * profile["cost_per_km"] +
                       shipment_volume_kg * 0.03) * fuel_noise
    transport_cost = max(transport_cost, 5)

    # promised delivery window (SLA) in hours, mode-dependent
    sla_hours = {"Road": 24, "Rail": 30, "Air": 14, "Sea": 50}[mode]
    on_time = delivery_time_hours <= sla_hours

    ship_date = np.random.choice(dates)

    # customer satisfaction score (1-5), inversely related to delay & cost, plus noise
    delay_ratio = delivery_time_hours / sla_hours
    sat_score = np.clip(5.5 - 1.8 * delay_ratio + np.random.normal(0, 0.5), 1, 5)

    records.append({
        "shipment_id": f"SHP{i:05d}",
        "ship_date": pd.Timestamp(ship_date),
        "region": region,
        "transport_mode": mode,
        "distance_km": round(distance_km, 1),
        "shipment_volume_kg": round(shipment_volume_kg, 1),
        "delivery_time_hours": round(delivery_time_hours, 2),
        "sla_hours": sla_hours,
        "on_time": on_time,
        "transport_cost_usd": round(transport_cost, 2),
        "customer_satisfaction": round(sat_score, 2),
    })

df = pd.DataFrame(records)
df["month"] = df["ship_date"].dt.to_period("M").astype(str)
df["cost_per_kg"] = (df["transport_cost_usd"] / df["shipment_volume_kg"]).round(3)

df.to_csv("/home/claude/logistics/logistics_data.csv", index=False)
print(df.shape)
print(df.head())
print(df.dtypes)
