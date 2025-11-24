"""
Simple Approach to SKU-Level Forecasting
-----------------------------------------
Strategy:
1. Train ONE Prophet model on total sales (all SKUs combined)
2. Forecast total sales for next month
3. Distribute to individual SKUs based on their historical proportions
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load aggregated data"""
    df = pd.read_csv(filepath)
    df['Bill Date'] = pd.to_datetime(df['Bill Date'])
    return df


def get_top_skus(df, top_n=20):
    """Get top N SKUs by total quantity"""
    top_skus = df.groupby('SKU Code')['Qty'].sum().nlargest(top_n).index.tolist()
    return top_skus


def calculate_sku_proportions(df, sku_list):
    """
    Calculate historical proportion of each SKU in total sales
    
    Returns: Dictionary with SKU Code -> proportion
    """
    sku_data = df[df['SKU Code'].isin(sku_list)].copy()
    
    # Total sales per SKU
    sku_totals = sku_data.groupby('SKU Code')['Qty'].sum()
    
    # Calculate proportions
    total_sales = sku_totals.sum()
    proportions = (sku_totals / total_sales).to_dict()
    
    # Get product names for reference
    sku_names = sku_data.groupby('SKU Code')['Product Name'].first().to_dict()
    
    return proportions, sku_names, sku_totals


def train_aggregate_model(df, sku_list, train_end_date):
    """
    Train ONE model on aggregate sales of top SKUs
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    sku_list : list
        List of SKU codes to include
    train_end_date : str or Timestamp
        End date for training data
    """
    # Filter for top SKUs and training period
    train_data = df[(df['SKU Code'].isin(sku_list)) & 
                    (df['Bill Date'] <= train_end_date)].copy()
    
    # Aggregate by date (sum all SKUs)
    daily_data = train_data.groupby('Bill Date')['Qty'].sum().reset_index()
    daily_data.columns = ['ds', 'y']
    
    print(f"\nTraining on aggregate data:")
    print(f"  Training days: {len(daily_data)}")
    print(f"  Total quantity: {daily_data['y'].sum():,.0f} units")
    print(f"  Daily average: {daily_data['y'].mean():.1f} units")
    
    # Train Prophet model with tuned parameters
    model = Prophet(
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=1.0,
        seasonality_mode='multiplicative',
        changepoint_range=0.8,
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=False
    )
    
    print("\nTraining Prophet model...")
    model.fit(daily_data)
    print("✓ Model trained successfully")
    
    return model, daily_data


def forecast_and_distribute(model, forecast_start, forecast_end, proportions, sku_names, last_train_date=None):
    """
    Generate aggregate forecast and distribute to individual SKUs
    
    Returns:
    --------
    DataFrame with daily SKU-level forecasts
    """
    # Calculate number of days to forecast
    forecast_start = pd.Timestamp(forecast_start)
    forecast_end = pd.Timestamp(forecast_end)
    
    # If forecasting into future, calculate periods from last training date
    if last_train_date:
        last_train_date = pd.Timestamp(last_train_date)
        num_periods = (forecast_end - last_train_date).days
    else:
        num_periods = (forecast_end - forecast_start).days + 10
    
    print(f"\nGenerating forecast (periods: {num_periods})...")
    
    # Generate aggregate forecast
    future = model.make_future_dataframe(periods=max(num_periods, 60), freq='D')
    forecast = model.predict(future)
    
    # Filter to forecast period
    forecast_period = forecast[(forecast['ds'] >= forecast_start) & 
                                (forecast['ds'] <= forecast_end)].copy()
    
    print(f"✓ Aggregate forecast generated: {forecast_period['yhat'].sum():,.0f} total units")
    
    # Distribute to individual SKUs
    sku_forecasts = []
    
    for sku_code, proportion in proportions.items():
        sku_forecast = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        sku_forecast['SKU_Code'] = sku_code
        sku_forecast['Product_Name'] = sku_names[sku_code]
        sku_forecast['Historical_Proportion'] = proportion
        
        # Apply proportion to get SKU-level forecast
        sku_forecast['Forecast'] = sku_forecast['yhat'] * proportion
        sku_forecast['Forecast_Lower'] = sku_forecast['yhat_lower'] * proportion
        sku_forecast['Forecast_Upper'] = sku_forecast['yhat_upper'] * proportion
        
        sku_forecasts.append(sku_forecast)
    
    # Combine all SKU forecasts
    all_forecasts = pd.concat(sku_forecasts, ignore_index=True)
    
    return all_forecasts, forecast_period


def evaluate_on_test_period(df, model, proportions, sku_names, test_start, test_end):
    """
    Evaluate the simple approach on test period
    """
    print(f"\n{'='*80}")
    print("EVALUATING ON TEST PERIOD")
    print("="*80)
    
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    
    # Get actual data for test period
    test_data = df[(df['Bill Date'] >= test_start) & (df['Bill Date'] <= test_end)].copy()
    test_data = test_data[test_data['SKU Code'].isin(proportions.keys())]
    
    # Get aggregate forecast
    forecast_df, aggregate_forecast = forecast_and_distribute(
        model, test_start, test_end, proportions, sku_names, last_train_date=None
    )
    
    # Compare aggregate level
    actual_daily = test_data.groupby('Bill Date')['Qty'].sum().reset_index()
    actual_daily.columns = ['ds', 'y']
    
    comparison = actual_daily.merge(aggregate_forecast[['ds', 'yhat']], on='ds', how='inner')
    
    agg_mae = mean_absolute_error(comparison['y'], comparison['yhat'])
    agg_mape = mean_absolute_percentage_error(comparison['y'], comparison['yhat']) * 100
    
    print(f"\nAggregate Level Performance:")
    print(f"  MAPE: {agg_mape:.2f}%")
    print(f"  MAE:  {agg_mae:.2f} units/day")
    print(f"  Actual Total: {comparison['y'].sum():,.0f} units")
    print(f"  Forecast Total: {comparison['yhat'].sum():,.0f} units")
    
    # Compare SKU level
    print(f"\n{'='*80}")
    print("SKU-LEVEL PERFORMANCE")
    print("="*80)
    
    sku_results = []
    
    for sku_code in proportions.keys():
        # Actual SKU data
        sku_actual = test_data[test_data['SKU Code'] == sku_code].groupby('Bill Date')['Qty'].sum().reset_index()
        sku_actual.columns = ['ds', 'y']
        
        # Forecast SKU data
        sku_forecast = forecast_df[forecast_df['SKU_Code'] == sku_code][['ds', 'Forecast']].copy()
        sku_forecast.columns = ['ds', 'yhat']
        
        # Merge
        sku_comp = sku_actual.merge(sku_forecast, on='ds', how='inner')
        
        if len(sku_comp) == 0:
            continue
        
        sku_mae = mean_absolute_error(sku_comp['y'], sku_comp['yhat'])
        sku_mape = mean_absolute_percentage_error(sku_comp['y'], sku_comp['yhat']) * 100
        
        sku_results.append({
            'SKU_Code': sku_code,
            'Product_Name': sku_names[sku_code],
            'Proportion': proportions[sku_code] * 100,
            'Actual_Total': sku_comp['y'].sum(),
            'Forecast_Total': sku_comp['yhat'].sum(),
            'MAPE': sku_mape,
            'MAE': sku_mae
        })
    
    results_df = pd.DataFrame(sku_results).sort_values('Actual_Total', ascending=False)
    
    print(f"\n{'SKU Code':<12} {'Proportion':<12} {'Actual':<10} {'Forecast':<10} {'MAPE':<10}")
    print("-"*60)
    for _, row in results_df.head(10).iterrows():
        print(f"{int(row['SKU_Code']):<12} {row['Proportion']:<11.1f}% {row['Actual_Total']:<9.0f} {row['Forecast_Total']:<9.0f} {row['MAPE']:<9.1f}%")
    
    print(f"\n... (showing top 10 out of {len(results_df)})")
    
    print(f"\nOverall SKU-Level Performance:")
    print(f"  Mean MAPE: {results_df['MAPE'].mean():.2f}%")
    print(f"  Median MAPE: {results_df['MAPE'].median():.2f}%")
    print(f"  Std MAPE: {results_df['MAPE'].std():.2f}%")
    
    return results_df, agg_mape, agg_mae


def generate_next_month_forecast(model, proportions, sku_names, sku_totals, 
                                  forecast_start, forecast_end, last_train_date):
    """
    Generate SKU-level forecast for next month
    """
    print(f"\n{'='*80}")
    print(f"FORECASTING NEXT MONTH: {forecast_start} to {forecast_end}")
    print("="*80)
    
    # Generate forecast
    forecast_df, aggregate_forecast = forecast_and_distribute(
        model, forecast_start, forecast_end, proportions, sku_names, last_train_date
    )
    
    # Create summary by SKU
    summary = forecast_df.groupby(['SKU_Code', 'Product_Name', 'Historical_Proportion']).agg({
        'Forecast': 'sum',
        'Forecast_Lower': 'sum',
        'Forecast_Upper': 'sum'
    }).reset_index()
    
    summary.columns = ['SKU_Code', 'Product_Name', 'Historical_Proportion', 
                       'Forecast_Quantity', 'Forecast_Lower', 'Forecast_Upper']
    
    # Add historical total for reference
    summary['Historical_Total'] = summary['SKU_Code'].map(sku_totals)
    
    # Sort by forecast
    summary = summary.sort_values('Forecast_Quantity', ascending=False)
    
    # Display
    print(f"\n{'Rank':<6} {'SKU Code':<12} {'Forecast':<12} {'Range':<25} {'Share':<8}")
    print("-"*70)
    for i, (_, row) in enumerate(summary.head(20).iterrows(), 1):
        range_str = f"{row['Forecast_Lower']:.0f} - {row['Forecast_Upper']:.0f}"
        print(f"{i:<6} {int(row['SKU_Code']):<12} {row['Forecast_Quantity']:<11.0f} {range_str:<25} {row['Historical_Proportion']*100:<7.1f}%")
    
    print(f"\n✓ Total Forecast: {summary['Forecast_Quantity'].sum():,.0f} units")
    
    return summary, forecast_df


def main():
    """Main execution"""
    
    print("="*80)
    print("SIMPLE APPROACH: SKU-LEVEL FORECASTING")
    print("="*80)
    print("\nStrategy: Train one model on total → Split by historical proportions")
    
    # Configuration
    DATA_FILE = 'aggregated_offtake_data.csv'
    TOP_N = 20
    TRAIN_END = '2025-09-20'  # Train on all data up to Sept 20
    TEST_START = '2025-09-21'
    TEST_END = '2025-10-20'
    FORECAST_START = '2025-11-21'
    FORECAST_END = '2025-12-20'
    
    # Load data
    print("\n1. Loading data...")
    df = load_data(DATA_FILE)
    print(f"   ✓ Loaded {len(df):,} records")
    
    # Get top SKUs
    print(f"\n2. Identifying top {TOP_N} SKUs...")
    top_skus = get_top_skus(df, TOP_N)
    print(f"   ✓ Top {TOP_N} SKUs identified")
    
    # Calculate proportions from ALL historical data
    print(f"\n3. Calculating historical proportions...")
    proportions, sku_names, sku_totals = calculate_sku_proportions(df, top_skus)
    
    print(f"\n   Top 5 SKUs by proportion:")
    sorted_props = sorted(proportions.items(), key=lambda x: x[1], reverse=True)
    for sku, prop in sorted_props[:5]:
        print(f"   • {sku}: {prop*100:.1f}% - {sku_names[sku][:50]}")
    
    # Train aggregate model
    print(f"\n4. Training aggregate model (up to {TRAIN_END})...")
    model, train_daily = train_aggregate_model(df, top_skus, TRAIN_END)
    
    # Evaluate on test period (October)
    print(f"\n5. Evaluating on October test period...")
    test_results, agg_mape, agg_mae = evaluate_on_test_period(
        df, model, proportions, sku_names, TEST_START, TEST_END
    )
    
    # Generate next month forecast
    print(f"\n6. Generating next month forecast...")
    summary, detailed_forecast = generate_next_month_forecast(
        model, proportions, sku_names, sku_totals,
        FORECAST_START, FORECAST_END, TRAIN_END
    )
    
    # Save results
    print(f"\n7. Saving results...")
    test_results.to_csv('simple_approach_test_results.csv', index=False)
    summary.to_csv('simple_approach_next_month_summary.csv', index=False)
    detailed_forecast.to_csv('simple_approach_next_month_details.csv', index=False)
    
    print("   ✓ Files saved:")
    print("     - simple_approach_test_results.csv")
    print("     - simple_approach_next_month_summary.csv")
    print("     - simple_approach_next_month_details.csv")
    
    print("\n" + "="*80)
    print("✅ SIMPLE APPROACH COMPLETED!")
    print("="*80)
    
    print(f"\nKey Results:")
    print(f"  • Test Period MAPE (aggregate): {agg_mape:.2f}%")
    print(f"  • Test Period MAPE (SKU avg): {test_results['MAPE'].mean():.2f}%")
    print(f"  • Next Month Total Forecast: {summary['Forecast_Quantity'].sum():,.0f} units")
    print(f"  • Approach: ONE model + proportional distribution")
    
    return model, test_results, summary


if __name__ == "__main__":
    model, test_results, summary = main()

