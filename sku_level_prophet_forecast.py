"""
SKU-Level Prophet Forecasting Script
-------------------------------------
Trains Prophet models for individual SKUs and evaluates performance.
Handles sparse/intermittent demand patterns.

Usage:
    python sku_level_prophet_forecast.py
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings

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


def get_top_skus(train_df, n=50, min_days=30):
    """
    Get top N SKUs by sales volume with minimum number of days

    Parameters:
    -----------
    train_df : DataFrame
        Training data
    n : int
        Number of top SKUs to return
    min_days : int
        Minimum number of days with sales required
    """
    sku_stats = train_df.groupby("SKU Code").agg({"Bill Date": "count", "Qty": "sum"})
    sku_stats.columns = ["Days", "Total_Qty"]

    # Filter by minimum days and get top N
    qualified_skus = sku_stats[sku_stats["Days"] >= min_days]
    top_skus = qualified_skus.nlargest(n, "Total_Qty")

    return top_skus.index.tolist()


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
            if len(sku_train_daily) < 20:
                print(
                    f"  {idx:2d}/{total_skus} SKU {sku_code} - Skipped (insufficient data)"
                )
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
            comparison["Forecast"] = comparison["yhat"].fillna(0)

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
                        "Date": row["ds"],
                        "Actual": row["Actual"],
                        "Forecast": row["Forecast"],
                        "Lower_Bound": row.get("yhat_lower", 0),
                        "Upper_Bound": row.get("yhat_upper", 0),
                    }
                )

            print(
                f"  {idx:2d}/{total_skus} SKU {sku_code} - {product_name[:35]:35s} | MAPE: {mape:6.1f}% | Actual: {actual_total:4.0f}"
            )

        except Exception as e:
            print(f"  {idx:2d}/{total_skus} SKU {sku_code} - Error: {str(e)}")
            continue

    results_df = pd.DataFrame(results)
    forecasts_df = pd.DataFrame(forecasts)

    return results_df, forecasts_df


def print_summary(results_df):
    """Print summary of forecast results"""
    print("\n" + "=" * 80)
    print("FORECAST SUMMARY")
    print("=" * 80)

    print(f"\nTotal SKUs Forecasted: {len(results_df)}")
    print(f"Total Actual Sales: {results_df['Actual_Total'].sum():,.0f} units")
    print(f"Total Forecast Sales: {results_df['Forecast_Total'].sum():,.0f} units")
    print(
        f"Overall Accuracy: {100 - abs(results_df['Actual_Total'].sum() - results_df['Forecast_Total'].sum())/results_df['Actual_Total'].sum()*100:.2f}%"
    )

    # Performance tiers
    excellent = len(results_df[results_df["MAPE"] < 30])
    good = len(results_df[(results_df["MAPE"] >= 30) & (results_df["MAPE"] < 50)])
    fair = len(results_df[(results_df["MAPE"] >= 50) & (results_df["MAPE"] < 100)])
    poor = len(results_df[results_df["MAPE"] >= 100])

    print(f"\nPerformance Distribution:")
    print(f"  Excellent (MAPE < 30%):  {excellent} SKUs")
    print(f"  Good (30-50%):           {good} SKUs")
    print(f"  Fair (50-100%):          {fair} SKUs")
    print(f"  Poor (> 100%):           {poor} SKUs")

    # Category level
    print(f"\nCategory-Level Performance:")
    category_perf = results_df.groupby("Category").agg(
        {"Actual_Total": "sum", "Forecast_Total": "sum"}
    )
    category_perf["Error_%"] = abs(
        (category_perf["Actual_Total"] - category_perf["Forecast_Total"])
        / category_perf["Actual_Total"]
        * 100
    ).round(2)

    for cat, row in category_perf.iterrows():
        print(
            f"  {cat:12s} - Actual: {row['Actual_Total']:5.0f}, Forecast: {row['Forecast_Total']:6.0f}, Error: {row['Error_%']:5.2f}%"
        )


def main():
    """Main execution function"""

    # Configuration
    DATA_FILE = "/Users/abhilasha/Documents/client-projects/data_analysis/aggregated_offtake_data.csv"
    TEST_START = "2025-09-21"
    TEST_END = "2025-10-20"
    TOP_N_SKUS = 20
    MIN_DAYS = 30

    print("=" * 80)
    print("SKU-LEVEL PROPHET FORECASTING")
    print("=" * 80)

    # Load data
    print("\n1. Loading and splitting data...")
    train_df, test_df = load_data(DATA_FILE, TEST_START, TEST_END)
    print(f" Training: {len(train_df):,} records")
    print(f" Testing: {len(test_df):,} records")

    # Get top SKUs
    print(f"\n2. Selecting top {TOP_N_SKUS} SKUs...")
    sku_list = get_top_skus(train_df, n=TOP_N_SKUS, min_days=MIN_DAYS)
    print(f"  Selected {len(sku_list)} SKUs")

    # Forecast all SKUs
    print(f"\n3. Training models and generating forecasts...")
    print("-" * 80)
    results_df, forecasts_df = forecast_all_skus(
        train_df, test_df, sku_list, pd.Timestamp(TEST_START)
    )

    # Print summary
    print_summary(results_df)

    # Save results
    print("\n4. Saving results...")
    results_df.to_csv("sku_level_forecast_summary.csv", index=False)
    forecasts_df.to_csv("sku_level_forecast_details.csv", index=False)
    print("   ✓ sku_level_forecast_summary.csv")
    print("   ✓ sku_level_forecast_details.csv")

    print("\n" + "=" * 80)
    print("✅ SKU-LEVEL FORECASTING COMPLETED!")
    print("=" * 80)

    return results_df, forecasts_df


if __name__ == "__main__":
    results_df, forecasts_df = main()
