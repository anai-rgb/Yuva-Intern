"""
Week 3 Task - Step 3: Visualizations
Generates the chart set used in the DOCX report.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
OUT = "/home/claude/logistics"

df = pd.read_csv(f"{OUT}/logistics_data.csv", parse_dates=["ship_date"])
df["month_dt"] = df["ship_date"].dt.to_period("M").dt.to_timestamp()

# ---------------------------------------------------------------
# 1. Histogram: distribution of delivery times
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df["delivery_time_hours"], bins=40, kde=True, color="#3b6ea5")
plt.axvline(df["delivery_time_hours"].mean(), color="crimson", linestyle="--",
            label=f"Mean = {df['delivery_time_hours'].mean():.1f} h")
plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery Time (hours)")
plt.ylabel("Number of Shipments")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_delivery_time_hist.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Boxplot: delivery time by transport mode
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
order = df.groupby("transport_mode")["delivery_time_hours"].median().sort_values().index
sns.boxplot(data=df, x="transport_mode", y="delivery_time_hours", order=order,
            hue="transport_mode", legend=False, palette="Set2")
plt.title("Delivery Time by Transport Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Delivery Time (hours)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_delivery_time_by_mode.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Scatter: distance vs transport cost, colored by mode
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="distance_km", y="transport_cost_usd",
                 hue="transport_mode", alpha=0.55, s=35, palette="Set2")
plt.title("Transport Cost vs. Distance")
plt.xlabel("Distance (km)")
plt.ylabel("Transport Cost (USD)")
plt.legend(title="Mode")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_cost_vs_distance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------
numeric_cols = ["distance_km", "shipment_volume_kg", "delivery_time_hours",
                 "transport_cost_usd", "cost_per_kg", "customer_satisfaction"]
corr = df[numeric_cols].corr()
plt.figure(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Key Logistics Metrics")
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Monthly trend: shipment volume (count) & on-time rate
# ---------------------------------------------------------------
monthly = df.groupby("month_dt").agg(
    shipments=("shipment_id", "count"),
    on_time_rate=("on_time", "mean"),
).reset_index()

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.bar(monthly["month_dt"], monthly["shipments"], width=20, color="#8fb4d9",
        label="Shipment Count")
ax1.set_xlabel("Month")
ax1.set_ylabel("Shipment Count", color="#3b6ea5")
ax1.tick_params(axis="y", labelcolor="#3b6ea5")

ax2 = ax1.twinx()
ax2.plot(monthly["month_dt"], monthly["on_time_rate"] * 100, color="crimson",
         marker="o", linewidth=2, label="On-Time Rate (%)")
ax2.set_ylabel("On-Time Delivery Rate (%)", color="crimson")
ax2.tick_params(axis="y", labelcolor="crimson")
ax2.set_ylim(0, 100)

plt.title("Monthly Shipment Volume and On-Time Delivery Rate (2025)")
fig.autofmt_xdate()
fig.tight_layout()
plt.savefig(f"{OUT}/fig5_monthly_trend.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Bar chart: on-time delivery rate by region and mode
# ---------------------------------------------------------------
pivot = df.groupby(["region", "transport_mode"])["on_time"].mean().unstack() * 100
plt.figure(figsize=(9, 5.5))
pivot.plot(kind="bar", ax=plt.gca(), colormap="Set2")
plt.axhline(df["on_time"].mean() * 100, color="black", linestyle="--", linewidth=1,
            label=f"Overall Avg = {df['on_time'].mean()*100:.1f}%")
plt.title("On-Time Delivery Rate by Region and Transport Mode")
plt.xlabel("Region")
plt.ylabel("On-Time Delivery Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_ontime_region_mode.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Cost per kg by transport mode (bar, log scale)
# ---------------------------------------------------------------
plt.figure(figsize=(7.5, 5))
mode_cost = df.groupby("transport_mode")["cost_per_kg"].mean().sort_values(ascending=False)
bars = plt.bar(mode_cost.index, mode_cost.values, color=sns.color_palette("Set2"))
plt.yscale("log")
plt.title("Average Cost per Kilogram by Transport Mode (log scale)")
plt.xlabel("Transport Mode")
plt.ylabel("Cost per kg (USD, log scale)")
for b in bars:
    plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"${b.get_height():.2f}",
              ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_cost_per_kg_mode.png", dpi=150)
plt.close()

print("All 7 figures generated.")
