"""
Top 244 SKUs Prophet Forecasting Script
---------------------------------------
Trains Prophet models for the top 244 SKUs that contribute to 80% of sales.
Uses Pareto analysis results to focus on high-impact SKUs.

Usage:
    python forecast_top244_skus.py
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")


def load_data(filepath, test_start, test_end):
    """Load and split data into train/test"""
    df = pd.read_csv(filepath, low_memory=False)
    df["Bill Date"] = pd.to_datetime(df["Bill Date"])

    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    train_df = df[df["Bill Date"] < test_start].copy()
    test_df = df[(df["Bill Date"] >= test_start) & (df["Bill Date"] <= test_end)].copy()

    return train_df, test_df


def get_top_244_skus(pareto_file):
    """
    Load top 244 SKUs from Pareto analysis results

    Parameters:
    -----------
    pareto_file : str
        Path to the pareto analysis results CSV

    Returns:
    --------
    list : Top 244 SKU codes
    """
    if not os.path.exists(pareto_file):
        raise FileNotFoundError(
            f"Pareto analysis file not found: {pareto_file}\n"
            "Please run sku_pareto_analysis.py first."
        )

    pareto_df = pd.read_csv(pareto_file)

    # Get SKUs that contribute to 80% of sales
    top_skus = pareto_df[pareto_df["Cumulative_Percentage"] <= 80]["SKU Code"].tolist()

    print(f"  Loaded {len(top_skus)} SKUs from Pareto analysis")
    print(f"  These SKUs contribute to 80% of total sales")

    return top_skus


def train_sku_model(sku_data):
    """
    Train Prophet model for a single SKU

    Parameters:
    -----------
    sku_data : DataFrame
        Daily aggregated data with columns ['ds', 'y']
    """
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode="multiplicative",
    )

    model.fit(sku_data)
    return model


def forecast_sku(model, periods=30):
    """Generate forecast for specified periods"""
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast


def evaluate_forecast(actual, forecast):
    """Calculate forecast metrics"""
    # Handle zero values for MAPE calculation
    actual_safe = actual.replace(0, 0.01)
    forecast_safe = forecast.replace(0, 0.01)

    mae = mean_absolute_error(actual, forecast)

    try:
        mape = mean_absolute_percentage_error(actual_safe, forecast_safe) * 100
    except:
        mape = (
            abs((actual.sum() - forecast.sum()) / actual.sum() * 100)
            if actual.sum() > 0
            else 0
        )

    return mae, mape


def forecast_all_skus(train_df, test_df, sku_list, test_start):
    """
    Train models and generate forecasts for all SKUs in the list

    Returns:
    --------
    results_df : DataFrame
        Summary of results for each SKU
    forecasts_df : DataFrame
        Detailed daily forecasts for each SKU
    """
    results = []
    forecasts = []

    total_skus = len(sku_list)
    skipped = 0
    errors = 0

    print(f"\n  Starting forecast for {total_skus} SKUs...")
    print("-" * 80)

    for idx, sku_code in enumerate(sku_list, 1):
        try:
            # Get SKU data
            sku_train = train_df[train_df["SKU Code"] == sku_code].copy()
            sku_test = test_df[test_df["SKU Code"] == sku_code].copy()

            # Get product info
            product_name = (
                sku_train["Product Name"].iloc[0] if len(sku_train) > 0 else "Unknown"
            )
            brand = sku_train["Brand"].iloc[0] if len(sku_train) > 0 else "Unknown"
            category = (
                sku_train["Category"].iloc[0] if len(sku_train) > 0 else "Unknown"
            )

            # Aggregate daily sales
            sku_train_daily = sku_train.groupby("Bill Date")["Qty"].sum().reset_index()
            sku_train_daily.columns = ["ds", "y"]

            sku_test_daily = sku_test.groupby("Bill Date")["Qty"].sum().reset_index()
            sku_test_daily.columns = ["ds", "y"]

            # Skip if insufficient data
            if len(sku_train_daily) < 15:
                print(
                    f"  {idx:3d}/{total_skus} SKU {sku_code:8d} - Skipped (insufficient data: {len(sku_train_daily)} days)"
                )
                skipped += 1
                continue

            # Train model
            model = train_sku_model(sku_train_daily)

            # Generate forecast
            forecast = forecast_sku(model, periods=30)
            test_forecast = forecast[forecast["ds"] >= test_start].head(30)

            # Merge with actual
            comparison = sku_test_daily.merge(
                test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                on="ds",
                how="outer",
            )
            comparison["Actual"] = comparison["y"].fillna(0)
            comparison["Forecast"] = (
                comparison["yhat"].clip(lower=0).fillna(0)
            )  # Clip negative forecasts

            # Calculate metrics
            actual_total = comparison["Actual"].sum()
            forecast_total = comparison["Forecast"].sum()

            mae, mape = evaluate_forecast(comparison["Actual"], comparison["Forecast"])

            # Store results
            results.append(
                {
                    "SKU_Code": sku_code,
                    "Product_Name": product_name,
                    "Brand": brand,
                    "Category": category,
                    "Train_Days": len(sku_train_daily),
                    "Test_Days": len(sku_test_daily),
                    "Actual_Total": actual_total,
                    "Forecast_Total": forecast_total,
                    "MAE": mae,
                    "MAPE": mape,
                    "Accuracy": 100 - mape if mape < 100 else 0,
                }
            )

            # Store forecast details
            for _, row in comparison.iterrows():
                forecasts.append(
                    {
                        "SKU_Code": sku_code,
                        "Product_Name": product_name,
                        "Brand": brand,
                        "Date": row["ds"],
                        "Actual": row["Actual"],
                        "Forecast": row["Forecast"],
                        "Lower_Bound": row.get("yhat_lower", 0),
                        "Upper_Bound": row.get("yhat_upper", 0),
                    }
                )

            # Progress indicator
            status = "✓" if mape < 50 else "⚠" if mape < 100 else "✗"
            print(
                f"  {status} {idx:3d}/{total_skus} SKU {sku_code:8d} | {product_name[:30]:30s} | MAPE: {mape:6.1f}% | Actual: {actual_total:5.0f}"
            )

        except Exception as e:
            print(f"  ✗ {idx:3d}/{total_skus} SKU {sku_code:8d} - Error: {str(e)}")
            errors += 1
            continue

    print("-" * 80)
    print(f"  Completed: {len(results)} SKUs | Skipped: {skipped} | Errors: {errors}")

    results_df = pd.DataFrame(results)
    forecasts_df = pd.DataFrame(forecasts)

    return results_df, forecasts_df


def print_summary(results_df):
    """Print summary of forecast results"""
    print("\n" + "=" * 80)
    print("FORECAST SUMMARY - TOP 244 SKUs (80% of Sales)")
    print("=" * 80)

    print(f"\nTotal SKUs Forecasted: {len(results_df)}")
    print(f"Total Actual Sales: {results_df['Actual_Total'].sum():,.0f} units")
    print(f"Total Forecast Sales: {results_df['Forecast_Total'].sum():,.0f} units")

    overall_error = (
        abs(results_df["Actual_Total"].sum() - results_df["Forecast_Total"].sum())
        / results_df["Actual_Total"].sum()
        * 100
    )
    print(f"Overall Accuracy: {100 - overall_error:.2f}%")
    print(f"Average MAPE: {results_df['MAPE'].mean():.2f}%")
    print(f"Median MAPE: {results_df['MAPE'].median():.2f}%")

    # Performance tiers
    excellent = len(results_df[results_df["MAPE"] < 30])
    good = len(results_df[(results_df["MAPE"] >= 30) & (results_df["MAPE"] < 50)])
    fair = len(results_df[(results_df["MAPE"] >= 50) & (results_df["MAPE"] < 100)])
    poor = len(results_df[results_df["MAPE"] >= 100])

    print(f"\nPerformance Distribution:")
    print(
        f"  Excellent (MAPE < 30%):  {excellent:3d} SKUs ({excellent/len(results_df)*100:5.1f}%)"
    )
    print(
        f"  Good (30-50%):           {good:3d} SKUs ({good/len(results_df)*100:5.1f}%)"
    )
    print(
        f"  Fair (50-100%):          {fair:3d} SKUs ({fair/len(results_df)*100:5.1f}%)"
    )
    print(
        f"  Poor (> 100%):           {poor:3d} SKUs ({poor/len(results_df)*100:5.1f}%)"
    )

    # Brand level performance
    if "Brand" in results_df.columns:
        print(f"\nTop Brands by Volume:")
        brand_perf = (
            results_df.groupby("Brand")
            .agg({"Actual_Total": "sum", "Forecast_Total": "sum", "SKU_Code": "count"})
            .rename(columns={"SKU_Code": "Num_SKUs"})
        )
        brand_perf["Error_%"] = abs(
            (brand_perf["Actual_Total"] - brand_perf["Forecast_Total"])
            / brand_perf["Actual_Total"]
            * 100
        ).round(2)
        brand_perf = brand_perf.sort_values("Actual_Total", ascending=False).head(10)

        for brand, row in brand_perf.iterrows():
            print(
                f"  {brand:15s} | {row['Num_SKUs']:3.0f} SKUs | Actual: {row['Actual_Total']:6.0f} | Forecast: {row['Forecast_Total']:6.0f} | Error: {row['Error_%']:5.2f}%"
            )

    # Top 10 best and worst forecasts
    print(f"\nTop 10 Best Forecasts (Lowest MAPE):")
    best = results_df.nsmallest(10, "MAPE")
    for idx, row in best.iterrows():
        print(
            f"  SKU {row['SKU_Code']:8d} | {row['Product_Name'][:35]:35s} | MAPE: {row['MAPE']:5.1f}%"
        )

    print(f"\nTop 10 Worst Forecasts (Highest MAPE):")
    worst = results_df.nlargest(10, "MAPE")
    for idx, row in worst.iterrows():
        print(
            f"  SKU {row['SKU_Code']:8d} | {row['Product_Name'][:35]:35s} | MAPE: {row['MAPE']:5.1f}%"
        )


def main():
    """Main execution function"""

    # Configuration
    DATA_FILE = "/Users/abhilasha/Documents/client-projects/data_analysis/aggregated_offtake_data.csv"
    PARETO_FILE = "/Users/abhilasha/Documents/client-projects/data_analysis/sku_pareto_analysis_results.csv"
    TEST_START = "2025-09-21"  # Test period: 21st to 20th cycle
    TEST_END = "2025-10-20"

    print("=" * 80)
    print("PROPHET FORECASTING - TOP 244 SKUs (80% of Sales)")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n1. Loading and splitting data...")
    train_df, test_df = load_data(DATA_FILE, TEST_START, TEST_END)
    print(
        f"   Training: {len(train_df):,} records ({train_df['Bill Date'].min()} to {train_df['Bill Date'].max()})"
    )
    print(f"   Testing:  {len(test_df):,} records ({TEST_START} to {TEST_END})")

    # Get top 244 SKUs from Pareto analysis
    print(f"\n2. Loading top 244 SKUs from Pareto analysis...")
    sku_list = get_top_244_skus(PARETO_FILE)

    # Forecast all SKUs
    print(f"\n3. Training Prophet models and generating forecasts...")
    print(f"   This may take 10-15 minutes for 244 SKUs...")

    results_df, forecasts_df = forecast_all_skus(
        train_df, test_df, sku_list, pd.Timestamp(TEST_START)
    )

    # Print summary
    print_summary(results_df)

    # Save results
    print("\n4. Saving results...")
    results_file = "top244_skus_forecast_summary.csv"
    forecasts_file = "top244_skus_forecast_details.csv"

    results_df.to_csv(results_file, index=False)
    forecasts_df.to_csv(forecasts_file, index=False)

    print(f"   ✓ {results_file}")
    print(f"   ✓ {forecasts_file}")

    print("\n" + "=" * 80)
    print(f"✅ FORECASTING COMPLETED!")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results_df, forecasts_df


if __name__ == "__main__":
    results_df, forecasts_df = main()
