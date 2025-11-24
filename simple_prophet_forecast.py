"""
Simple Prophet Forecasting Model for Offtake Data
--------------------------------------------------
Custom month cycle: 21st to 20th
Test period: September 21 - October 20, 2025
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(filepath):
    """Load and prepare aggregated offtake data"""
    df = pd.read_csv(filepath)
    df["Bill Date"] = pd.to_datetime(df["Bill Date"])
    return df


def split_data_custom_cycle(df, test_start, test_end):
    """
    Split data based on custom month cycle (21st-20th)

    Parameters:
    -----------
    df : DataFrame
        Aggregated offtake data
    test_start : str or Timestamp
        Start of test period (e.g., '2025-09-21')
    test_end : str or Timestamp
        End of test period (e.g., '2025-10-20')
    """
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    train_df = df[df["Bill Date"] < test_start].copy()
    test_df = df[(df["Bill Date"] >= test_start) & (df["Bill Date"] <= test_end)].copy()

    return train_df, test_df


def aggregate_daily_sales(df):
    """Aggregate total daily quantity across all stores and SKUs"""
    daily = df.groupby("Bill Date")["Qty"].sum().reset_index()
    daily.columns = ["ds", "y"]
    return daily


def train_prophet_model(train_data):
    """
    Train Prophet model on training data

    Parameters:
    -----------
    train_data : DataFrame
        Daily aggregated data with columns ['ds', 'y']
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )

    model.fit(train_data)
    return model


def generate_forecast(model, periods=30):
    """Generate forecast for specified number of days"""
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast


def evaluate_forecast(test_actual, test_forecast):
    """
    Evaluate forecast performance

    Returns metrics and comparison dataframe
    """
    comparison = test_actual.merge(
        test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds", how="left"
    )
    comparison.columns = ["Date", "Actual", "Forecast", "Lower_Bound", "Upper_Bound"]

    # Calculate metrics
    mae = mean_absolute_error(comparison["Actual"], comparison["Forecast"])
    rmse = np.sqrt(mean_squared_error(comparison["Actual"], comparison["Forecast"]))
    mape = (
        mean_absolute_percentage_error(comparison["Actual"], comparison["Forecast"])
        * 100
    )

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Accuracy": 100 - mape,
        "Actual_Total": comparison["Actual"].sum(),
        "Forecast_Total": comparison["Forecast"].sum(),
    }

    return comparison, metrics


def print_results(metrics, comparison):
    """Print formatted results"""
    print("=" * 80)
    print("PROPHET FORECASTING RESULTS")
    print("=" * 80)

    print(f"\nModel Performance:")
    print(f"  MAE:        {metrics['MAE']:.2f} units/day")
    print(f"  RMSE:       {metrics['RMSE']:.2f} units/day")
    print(f"  MAPE:       {metrics['MAPE']:.2f}%")
    print(f"  Accuracy:   {metrics['Accuracy']:.2f}%")

    print(f"\nTotal Sales:")
    print(f"  Actual:     {metrics['Actual_Total']:,.0f} units")
    print(f"  Forecast:   {metrics['Forecast_Total']:,.0f} units")
    print(
        f"  Difference: {metrics['Actual_Total'] - metrics['Forecast_Total']:,.0f} units"
    )

    print(f"\nForecast vs Actual (first 10 days):")
    display = comparison.head(10)[["Date", "Actual", "Forecast"]].copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["Forecast"] = display["Forecast"].round(2)
    print(display.to_string(index=False))


def main():
    """Main execution function"""

    # Configuration
    DATA_FILE = "/Users/abhilasha/Documents/client-projects/data_analysis/aggregated_offtake_data.csv"
    TEST_START = "2025-09-21"
    TEST_END = "2025-10-20"

    print("=" * 80)
    print("SIMPLE PROPHET FORECASTING")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    df = load_and_prepare_data(DATA_FILE)
    print(f"   ✓ Loaded {len(df):,} records")

    # Split data
    print("\n2. Splitting data...")
    train_df, test_df = split_data_custom_cycle(df, TEST_START, TEST_END)
    print(f"   ✓ Training: {len(train_df):,} records")
    print(f"   ✓ Testing: {len(test_df):,} records")

    # Aggregate daily
    print("\n3. Aggregating daily sales...")
    train_daily = aggregate_daily_sales(train_df)
    test_daily = aggregate_daily_sales(test_df)
    print(f"   ✓ Training days: {len(train_daily)}")
    print(f"   ✓ Testing days: {len(test_daily)}")

    # Train model
    print("\n4. Training Prophet model...")
    model = train_prophet_model(train_daily)
    print("   ✓ Model trained successfully")

    # Generate forecast
    print("\n5. Generating forecast...")
    forecast = generate_forecast(model, periods=30)
    test_forecast = forecast[forecast["ds"] >= TEST_START].head(30)
    print("   ✓ Forecast generated")

    # Evaluate
    print("\n6. Evaluating performance...")
    comparison, metrics = evaluate_forecast(test_daily, test_forecast)

    # Print results
    print_results(metrics, comparison)

    # Save results
    print("\n7. Saving results...")
    comparison.to_csv("prophet_forecast_results.csv", index=False)
    forecast.to_csv("prophet_full_forecast.csv", index=False)
    print("   ✓ Results saved to:")
    print("     - prophet_forecast_results.csv")
    print("     - prophet_full_forecast.csv")

    print("\n" + "=" * 80)
    print("✅ FORECASTING COMPLETED!")
    print("=" * 80)

    return model, comparison, metrics


if __name__ == "__main__":
    model, comparison, metrics = main()
