"""
Create Volume Accuracy Report
------------------------------
Recalculates SKU-level accuracy using Volume Error % instead of MAPE
(More meaningful for intermittent demand patterns)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("VOLUME ACCURACY REPORT - TOP 244 SKUs")
print("=" * 80)

# Load results
results_df = pd.read_csv("top244_skus_forecast_summary.csv")

# Calculate volume accuracy metrics
results_df["Volume_Error_%"] = abs(
    (results_df["Actual_Total"] - results_df["Forecast_Total"])
    / results_df["Actual_Total"]
    * 100
)
results_df["Volume_Accuracy_%"] = 100 - results_df["Volume_Error_%"]

# Calculate bias (over vs under forecasting)
results_df["Bias"] = results_df["Forecast_Total"] - results_df["Actual_Total"]
results_df["Bias_%"] = (results_df["Bias"] / results_df["Actual_Total"]) * 100

# Sort by volume accuracy
results_df = results_df.sort_values("Volume_Error_%")

print("\n" + "=" * 80)
print("OVERALL PERFORMANCE SUMMARY")
print("=" * 80)

# Performance tiers based on volume accuracy
excellent = len(results_df[results_df["Volume_Error_%"] < 20])
good = len(
    results_df[
        (results_df["Volume_Error_%"] >= 20) & (results_df["Volume_Error_%"] < 40)
    ]
)
fair = len(
    results_df[
        (results_df["Volume_Error_%"] >= 40) & (results_df["Volume_Error_%"] < 60)
    ]
)
poor = len(results_df[results_df["Volume_Error_%"] >= 60])

print(f"\nTotal SKUs Forecasted: {len(results_df)}")
print(f"Total Actual Sales: {results_df['Actual_Total'].sum():,.0f} units")
print(f"Total Forecast Sales: {results_df['Forecast_Total'].sum():,.0f} units")
print(
    f"Overall Volume Accuracy: {100 - abs((results_df['Actual_Total'].sum() - results_df['Forecast_Total'].sum()) / results_df['Actual_Total'].sum() * 100):.2f}%"
)
print(f"Average MAE: {results_df['MAE'].mean():.2f} units per day")

print(f"\n📊 Performance Distribution (by Volume Accuracy):")
print(
    f"  ✓ Excellent (error < 20%):   {excellent:3d} SKUs ({excellent/len(results_df)*100:5.1f}%)"
)
print(
    f"  ✓ Good (error 20-40%):       {good:3d} SKUs ({good/len(results_df)*100:5.1f}%)"
)
print(
    f"  ⚠ Fair (error 40-60%):       {fair:3d} SKUs ({fair/len(results_df)*100:5.1f}%)"
)
print(
    f"  ✗ Poor (error > 60%):        {poor:3d} SKUs ({poor/len(results_df)*100:5.1f}%)"
)

print(f"\n📈 Forecast Bias:")
over_forecast = len(results_df[results_df["Bias"] > 0])
under_forecast = len(results_df[results_df["Bias"] < 0])
print(f"  Over-forecasted:  {over_forecast} SKUs")
print(f"  Under-forecasted: {under_forecast} SKUs")
print(f"  Average Bias: {results_df['Bias_%'].mean():.1f}%")

# Top performers
print("\n" + "=" * 80)
print("TOP 20 SKUs BY VOLUME ACCURACY (Best Forecasts)")
print("=" * 80)

top_20 = results_df.head(20)
for idx, row in top_20.iterrows():
    status = (
        "✓"
        if row["Volume_Error_%"] < 20
        else "⚠" if row["Volume_Error_%"] < 40 else "✗"
    )
    bias_indicator = "↑" if row["Bias"] > 0 else "↓"
    print(
        f"{status} SKU {row['SKU_Code']:8d} | {row['Product_Name'][:35]:35s} | "
        f"Actual: {row['Actual_Total']:4.0f} | Forecast: {row['Forecast_Total']:4.0f} | "
        f"Error: {row['Volume_Error_%']:5.1f}% {bias_indicator}"
    )

# Bottom performers
print("\n" + "=" * 80)
print("BOTTOM 20 SKUs BY VOLUME ACCURACY (Challenging Forecasts)")
print("=" * 80)

bottom_20 = results_df.tail(20).iloc[::-1]  # Reverse to show worst first
for idx, row in bottom_20.iterrows():
    bias_indicator = "↑" if row["Bias"] > 0 else "↓"
    print(
        f"✗ SKU {row['SKU_Code']:8d} | {row['Product_Name'][:35]:35s} | "
        f"Actual: {row['Actual_Total']:4.0f} | Forecast: {row['Forecast_Total']:4.0f} | "
        f"Error: {row['Volume_Error_%']:5.1f}% {bias_indicator}"
    )

# Brand-level analysis
print("\n" + "=" * 80)
print("BRAND-LEVEL VOLUME ACCURACY")
print("=" * 80)

brand_perf = (
    results_df.groupby("Brand")
    .agg(
        {
            "Actual_Total": "sum",
            "Forecast_Total": "sum",
            "SKU_Code": "count",
            "Volume_Error_%": "mean",
        }
    )
    .rename(columns={"SKU_Code": "Num_SKUs"})
)

brand_perf["Volume_Error_%"] = abs(
    (brand_perf["Actual_Total"] - brand_perf["Forecast_Total"])
    / brand_perf["Actual_Total"]
    * 100
)
brand_perf = brand_perf.sort_values("Actual_Total", ascending=False)

print(
    f"\n{'Brand':<15} | {'SKUs':<5} | {'Actual':>8} | {'Forecast':>8} | {'Vol Error':>10}"
)
print("-" * 80)
for brand, row in brand_perf.head(15).iterrows():
    status = "✓" if row["Volume_Error_%"] < 20 else "⚠"
    print(
        f"{status} {brand:<13} | {row['Num_SKUs']:>4.0f}  | {row['Actual_Total']:>8.0f} | "
        f"{row['Forecast_Total']:>8.0f} | {row['Volume_Error_%']:>9.1f}%"
    )

# Save enhanced results
results_df.to_csv("top244_skus_volume_accuracy_report.csv", index=False)
print("\n" + "=" * 80)
print("✅ Saved: top244_skus_volume_accuracy_report.csv")
print("=" * 80)

# Create visualizations
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Volume Error Distribution
ax = axes[0, 0]
bins = [0, 20, 40, 60, 80, 100, max(results_df["Volume_Error_%"]) + 1]
hist_vals, bin_edges, patches = ax.hist(
    results_df["Volume_Error_%"], bins=bins, edgecolor="black", alpha=0.7
)
# Color the bars
colors = ["green", "lightgreen", "yellow", "orange", "red"]
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)

ax.set_xlabel("Volume Error (%)", fontsize=11)
ax.set_ylabel("Number of SKUs", fontsize=11)
ax.set_title("Distribution of Volume Error %", fontsize=13, fontweight="bold")
ax.axvline(x=20, color="darkgreen", linestyle="--", linewidth=2, label="20% threshold")
ax.axvline(x=40, color="darkorange", linestyle="--", linewidth=2, label="40% threshold")
ax.legend()
ax.grid(True, alpha=0.3)

# Add count labels on bars
for i, count in enumerate(hist_vals):
    if count > 0:
        ax.text(
            (bins[i] + bins[i + 1]) / 2,
            count,
            str(int(count)),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

# 2. Actual vs Forecast Scatter
ax = axes[0, 1]
scatter = ax.scatter(
    results_df["Actual_Total"],
    results_df["Forecast_Total"],
    c=results_df["Volume_Error_%"],
    cmap="RdYlGn_r",
    alpha=0.6,
    s=50,
)
ax.plot(
    [0, results_df["Actual_Total"].max()],
    [0, results_df["Actual_Total"].max()],
    "r--",
    linewidth=2,
    label="Perfect Forecast",
)
ax.set_xlabel("Actual Sales", fontsize=11)
ax.set_ylabel("Forecast Sales", fontsize=11)
ax.set_title(
    "Actual vs Forecast (colored by Volume Error %)", fontsize=13, fontweight="bold"
)
plt.colorbar(scatter, ax=ax, label="Volume Error %")
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Top 15 Best SKUs
ax = axes[1, 0]
top_15 = results_df.head(15)
y_pos = range(len(top_15))
colors_bar = ["green" if x < 20 else "lightgreen" for x in top_15["Volume_Error_%"]]
bars = ax.barh(y_pos, top_15["Volume_Error_%"], color=colors_bar)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"SKU {sku}" for sku in top_15["SKU_Code"]], fontsize=9)
ax.set_xlabel("Volume Error (%)", fontsize=11)
ax.set_title("Top 15 Most Accurate SKU Forecasts", fontsize=13, fontweight="bold")
ax.invert_yaxis()
ax.axvline(x=20, color="orange", linestyle="--", linewidth=2, alpha=0.5)
# Add value labels
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax.text(
        row["Volume_Error_%"],
        i,
        f" {row['Volume_Error_%']:.1f}%",
        va="center",
        fontsize=8,
    )

# 4. Summary Stats
ax = axes[1, 1]
ax.axis("off")

summary_text = f"""
VOLUME ACCURACY SUMMARY

Total SKUs: {len(results_df)}
Test Period: October 2025 (30 days)

AGGREGATE PERFORMANCE:
• Total Actual:    {results_df['Actual_Total'].sum():>7,.0f} units
• Total Forecast:  {results_df['Forecast_Total'].sum():>7,.0f} units
• Volume Accuracy: {100 - abs((results_df['Actual_Total'].sum() - results_df['Forecast_Total'].sum()) / results_df['Actual_Total'].sum() * 100):>7.2f}%

SKU-LEVEL ACCURACY:
• Excellent (<20% error):  {excellent:>4d} SKUs ({excellent/len(results_df)*100:.1f}%)
• Good (20-40% error):     {good:>4d} SKUs ({good/len(results_df)*100:.1f}%)
• Fair (40-60% error):     {fair:>4d} SKUs ({fair/len(results_df)*100:.1f}%)
• Poor (>60% error):       {poor:>4d} SKUs ({poor/len(results_df)*100:.1f}%)

FORECAST BIAS:
• Over-forecasted:  {over_forecast:>4d} SKUs
• Under-forecasted: {under_forecast:>4d} SKUs
• Avg Daily MAE:    {results_df['MAE'].mean():>7.2f} units

BEST PERFORMERS:
1. {results_df.iloc[0]['Product_Name'][:30]}
   Error: {results_df.iloc[0]['Volume_Error_%']:.1f}%
   
2. {results_df.iloc[1]['Product_Name'][:30]}
   Error: {results_df.iloc[1]['Volume_Error_%']:.1f}%
   
3. {results_df.iloc[2]['Product_Name'][:30]}
   Error: {results_df.iloc[2]['Volume_Error_%']:.1f}%
"""

ax.text(
    0.1,
    0.5,
    summary_text,
    fontsize=10,
    verticalalignment="center",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
)
ax.set_title("Performance Summary", fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("volume_accuracy_analysis.png", dpi=300, bbox_inches="tight")
print("   ✓ volume_accuracy_analysis.png")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
print(
    """
KEY INSIGHT:
📊 Using Volume Accuracy (total error %) instead of daily MAPE gives a much clearer
   picture of forecast performance for intermittent demand patterns.
   
   {0} SKUs ({1:.1f}%) have excellent volume accuracy (<20% error)
   This is much more actionable than the MAPE metric!
""".format(
        excellent, excellent / len(results_df) * 100
    )
)
