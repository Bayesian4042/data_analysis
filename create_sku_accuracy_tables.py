"""
Create SKU-Level Accuracy Tables
---------------------------------
Generates detailed accuracy tables with actual vs forecasted values for each SKU
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("CREATING SKU ACCURACY TABLES")
print("=" * 80)

# Load the forecast results
summary_df = pd.read_csv("top244_skus_forecast_summary.csv")
details_df = pd.read_csv("top244_skus_forecast_details.csv")

print(f"\n✓ Loaded {len(summary_df)} SKUs")
print(f"✓ Loaded {len(details_df)} daily forecast records")

# ============================================================================
# TABLE 1: SKU-Level Summary Table
# ============================================================================
print("\n" + "=" * 80)
print("TABLE 1: SKU-LEVEL ACCURACY SUMMARY")
print("=" * 80)

# Calculate additional metrics
summary_df["Volume_Error_%"] = abs(
    (summary_df["Actual_Total"] - summary_df["Forecast_Total"])
    / summary_df["Actual_Total"]
    * 100
).round(2)

summary_df["Volume_Accuracy_%"] = (100 - summary_df["Volume_Error_%"]).round(2)

summary_df["Bias"] = summary_df["Forecast_Total"] - summary_df["Actual_Total"]
summary_df["Bias_%"] = (summary_df["Bias"] / summary_df["Actual_Total"] * 100).round(2)

summary_df["Performance_Category"] = pd.cut(
    summary_df["Volume_Error_%"],
    bins=[0, 20, 40, 60, float("inf")],
    labels=["Excellent", "Good", "Fair", "Poor"],
)

# Create clean accuracy table
accuracy_table = summary_df[
    [
        "SKU_Code",
        "Product_Name",
        "Brand",
        "Category",
        "Actual_Total",
        "Forecast_Total",
        "Volume_Error_%",
        "Volume_Accuracy_%",
        "MAE",
        "Bias",
        "Bias_%",
        "Train_Days",
        "Test_Days",
        "Performance_Category",
    ]
].copy()

# Sort by volume accuracy (best first)
accuracy_table = accuracy_table.sort_values("Volume_Error_%")

# Save to CSV
accuracy_table.to_csv("sku_accuracy_summary_table.csv", index=False)
print("\n✓ Saved: sku_accuracy_summary_table.csv")

# Display sample
print("\nSample (Top 10 Most Accurate SKUs):")
print("-" * 80)
print(accuracy_table.head(10).to_string(index=False))

# ============================================================================
# TABLE 2: Daily Breakdown for Each SKU
# ============================================================================
print("\n\n" + "=" * 80)
print("TABLE 2: DAILY ACCURACY BREAKDOWN BY SKU")
print("=" * 80)

# Calculate daily metrics
details_df["Date"] = pd.to_datetime(details_df["Date"])
details_df["Error"] = abs(details_df["Actual"] - details_df["Forecast"])
details_df["Error_%"] = (
    details_df["Error"] / details_df["Actual"].replace(0, 0.01) * 100
).round(2)
details_df["Bias"] = details_df["Forecast"] - details_df["Actual"]

# Add day of week
details_df["Day_of_Week"] = details_df["Date"].dt.day_name()

# Create clean daily table
daily_accuracy_table = details_df[
    [
        "SKU_Code",
        "Product_Name",
        "Brand",
        "Date",
        "Day_of_Week",
        "Actual",
        "Forecast",
        "Lower_Bound",
        "Upper_Bound",
        "Error",
        "Error_%",
        "Bias",
    ]
].copy()

# Sort by SKU and Date
daily_accuracy_table = daily_accuracy_table.sort_values(["SKU_Code", "Date"])

# Save to CSV
daily_accuracy_table.to_csv("sku_daily_accuracy_table.csv", index=False)
print("\n✓ Saved: sku_daily_accuracy_table.csv")

# Display sample for one SKU
sample_sku = summary_df.iloc[0]["SKU_Code"]
print(f"\nSample Daily Data for SKU {sample_sku}:")
print("-" * 80)
sample_daily = daily_accuracy_table[
    daily_accuracy_table["SKU_Code"] == sample_sku
].head(15)
print(
    sample_daily[
        ["Date", "Day_of_Week", "Actual", "Forecast", "Error", "Bias"]
    ].to_string(index=False)
)

# ============================================================================
# TABLE 3: SKU Performance by Category/Brand
# ============================================================================
print("\n\n" + "=" * 80)
print("TABLE 3: ACCURACY BY BRAND AND CATEGORY")
print("=" * 80)

# Brand-level aggregation
brand_accuracy = (
    summary_df.groupby("Brand")
    .agg(
        {
            "SKU_Code": "count",
            "Actual_Total": "sum",
            "Forecast_Total": "sum",
            "MAE": "mean",
            "Volume_Error_%": "mean",
        }
    )
    .rename(columns={"SKU_Code": "Num_SKUs"})
)

brand_accuracy["Volume_Error_%_Aggregate"] = abs(
    (brand_accuracy["Actual_Total"] - brand_accuracy["Forecast_Total"])
    / brand_accuracy["Actual_Total"]
    * 100
).round(2)

brand_accuracy["Avg_SKU_Error_%"] = brand_accuracy["Volume_Error_%"].round(2)
brand_accuracy = brand_accuracy.sort_values("Actual_Total", ascending=False)

# Save brand accuracy
brand_accuracy.to_csv("brand_accuracy_table.csv")
print("\n✓ Saved: brand_accuracy_table.csv")

print("\nBrand-Level Accuracy:")
print("-" * 80)
print(
    brand_accuracy[
        [
            "Num_SKUs",
            "Actual_Total",
            "Forecast_Total",
            "Volume_Error_%_Aggregate",
            "Avg_SKU_Error_%",
            "MAE",
        ]
    ].to_string()
)

# Category-level aggregation
category_accuracy = (
    summary_df.groupby("Category")
    .agg(
        {
            "SKU_Code": "count",
            "Actual_Total": "sum",
            "Forecast_Total": "sum",
            "MAE": "mean",
            "Volume_Error_%": "mean",
        }
    )
    .rename(columns={"SKU_Code": "Num_SKUs"})
)

category_accuracy["Volume_Error_%_Aggregate"] = abs(
    (category_accuracy["Actual_Total"] - category_accuracy["Forecast_Total"])
    / category_accuracy["Actual_Total"]
    * 100
).round(2)

category_accuracy["Avg_SKU_Error_%"] = category_accuracy["Volume_Error_%"].round(2)
category_accuracy = category_accuracy.sort_values("Actual_Total", ascending=False)

# Save category accuracy
category_accuracy.to_csv("category_accuracy_table.csv")
print("\n✓ Saved: category_accuracy_table.csv")

print("\nCategory-Level Accuracy:")
print("-" * 80)
print(
    category_accuracy[
        [
            "Num_SKUs",
            "Actual_Total",
            "Forecast_Total",
            "Volume_Error_%_Aggregate",
            "Avg_SKU_Error_%",
            "MAE",
        ]
    ].to_string()
)

# ============================================================================
# TABLE 4: Weekly Aggregation for Each SKU
# ============================================================================
print("\n\n" + "=" * 80)
print("TABLE 4: WEEKLY ACCURACY BY SKU")
print("=" * 80)

# Add week number
details_df["Week"] = details_df["Date"].dt.isocalendar().week

# Weekly aggregation
weekly_accuracy = (
    details_df.groupby(["SKU_Code", "Product_Name", "Brand", "Week"])
    .agg({"Actual": "sum", "Forecast": "sum", "Date": ["min", "max"]})
    .reset_index()
)

weekly_accuracy.columns = [
    "SKU_Code",
    "Product_Name",
    "Brand",
    "Week",
    "Actual_Weekly",
    "Forecast_Weekly",
    "Week_Start",
    "Week_End",
]

weekly_accuracy["Weekly_Error"] = abs(
    weekly_accuracy["Actual_Weekly"] - weekly_accuracy["Forecast_Weekly"]
)
weekly_accuracy["Weekly_Error_%"] = (
    weekly_accuracy["Weekly_Error"]
    / weekly_accuracy["Actual_Weekly"].replace(0, 0.01)
    * 100
).round(2)
weekly_accuracy["Weekly_Bias"] = (
    weekly_accuracy["Forecast_Weekly"] - weekly_accuracy["Actual_Weekly"]
)

# Save weekly accuracy
weekly_accuracy.to_csv("sku_weekly_accuracy_table.csv", index=False)
print("\n✓ Saved: sku_weekly_accuracy_table.csv")

# Display sample
print(f"\nSample Weekly Data for SKU {sample_sku}:")
print("-" * 80)
sample_weekly = weekly_accuracy[weekly_accuracy["SKU_Code"] == sample_sku]
print(
    sample_weekly[
        [
            "Week",
            "Week_Start",
            "Week_End",
            "Actual_Weekly",
            "Forecast_Weekly",
            "Weekly_Error",
            "Weekly_Error_%",
        ]
    ].to_string(index=False)
)

# ============================================================================
# TABLE 5: Top/Bottom Performers
# ============================================================================
print("\n\n" + "=" * 80)
print("TABLE 5: TOP & BOTTOM PERFORMERS")
print("=" * 80)

# Top 20 performers
top_performers = accuracy_table.head(20)[
    [
        "SKU_Code",
        "Product_Name",
        "Brand",
        "Actual_Total",
        "Forecast_Total",
        "Volume_Error_%",
        "Performance_Category",
    ]
]
top_performers.to_csv("top_20_performers.csv", index=False)
print("\n✓ Saved: top_20_performers.csv")

print("\nTop 20 Most Accurate SKUs:")
print("-" * 80)
for idx, row in top_performers.iterrows():
    print(
        f"SKU {row['SKU_Code']:8d} | {row['Product_Name'][:35]:35s} | "
        f"Actual: {row['Actual_Total']:4.0f} | Forecast: {row['Forecast_Total']:4.0f} | "
        f"Error: {row['Volume_Error_%']:5.1f}%"
    )

# Bottom 20 performers
bottom_performers = accuracy_table.tail(20)[
    [
        "SKU_Code",
        "Product_Name",
        "Brand",
        "Actual_Total",
        "Forecast_Total",
        "Volume_Error_%",
        "Performance_Category",
    ]
]
bottom_performers.to_csv("bottom_20_performers.csv", index=False)
print("\n✓ Saved: bottom_20_performers.csv")

print("\nBottom 20 (Most Challenging) SKUs:")
print("-" * 80)
for idx, row in bottom_performers.iloc[::-1].iterrows():
    print(
        f"SKU {row['SKU_Code']:8d} | {row['Product_Name'][:35]:35s} | "
        f"Actual: {row['Actual_Total']:4.0f} | Forecast: {row['Forecast_Total']:4.0f} | "
        f"Error: {row['Volume_Error_%']:5.1f}%"
    )

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

summary_stats = {
    "Total_SKUs": len(summary_df),
    "Total_Actual_Sales": summary_df["Actual_Total"].sum(),
    "Total_Forecast_Sales": summary_df["Forecast_Total"].sum(),
    "Overall_Volume_Accuracy_%": 100
    - abs(
        (summary_df["Actual_Total"].sum() - summary_df["Forecast_Total"].sum())
        / summary_df["Actual_Total"].sum()
        * 100
    ),
    "Avg_Volume_Error_%": summary_df["Volume_Error_%"].mean(),
    "Median_Volume_Error_%": summary_df["Volume_Error_%"].median(),
    "Avg_MAE": summary_df["MAE"].mean(),
    "Excellent_SKUs": len(summary_df[summary_df["Volume_Error_%"] < 20]),
    "Good_SKUs": len(
        summary_df[
            (summary_df["Volume_Error_%"] >= 20) & (summary_df["Volume_Error_%"] < 40)
        ]
    ),
    "Fair_SKUs": len(
        summary_df[
            (summary_df["Volume_Error_%"] >= 40) & (summary_df["Volume_Error_%"] < 60)
        ]
    ),
    "Poor_SKUs": len(summary_df[summary_df["Volume_Error_%"] >= 60]),
    "Over_Forecast_SKUs": len(summary_df[summary_df["Bias"] > 0]),
    "Under_Forecast_SKUs": len(summary_df[summary_df["Bias"] < 0]),
}

summary_stats_df = pd.DataFrame([summary_stats]).T
summary_stats_df.columns = ["Value"]
summary_stats_df.to_csv("accuracy_summary_statistics.csv")

print(summary_stats_df.to_string())
print("\n✓ Saved: accuracy_summary_statistics.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("✅ ACCURACY TABLES CREATED SUCCESSFULLY!")
print("=" * 80)

print(
    """
Generated Files:
================

1. sku_accuracy_summary_table.csv
   → SKU-level summary with actual, forecast, accuracy metrics
   
2. sku_daily_accuracy_table.csv
   → Daily breakdown for each SKU (30 days × 243 SKUs = 7,290 rows)
   
3. sku_weekly_accuracy_table.csv
   → Weekly aggregation for easier analysis
   
4. brand_accuracy_table.csv
   → Accuracy metrics aggregated by brand
   
5. category_accuracy_table.csv
   → Accuracy metrics aggregated by category
   
6. top_20_performers.csv
   → Best forecasted SKUs
   
7. bottom_20_performers.csv
   → Most challenging SKUs
   
8. accuracy_summary_statistics.csv
   → Overall summary statistics

Use Case:
=========
- sku_accuracy_summary_table.csv → For executive summary reports
- sku_daily_accuracy_table.csv → For detailed drill-down analysis
- sku_weekly_accuracy_table.csv → For weekly planning meetings
- brand/category tables → For brand manager reviews
- top/bottom performers → For focus areas and best practices
"""
)

print("\n" + "=" * 80)
print("DONE!")
print("=" * 80)
