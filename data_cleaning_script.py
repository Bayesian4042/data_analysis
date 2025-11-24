"""
Data Cleaning Script for H&G X HBDH Offtake Trend Data
-------------------------------------------------------
This script handles:
1. Mixed Bill Date formats (text dates and Excel serial numbers)
2. Data type conversions
3. Data quality checks
4. Missing value handling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


def convert_excel_date(date_value):
    """
    Convert Excel serial date to datetime or parse text date

    Excel dates are stored as numbers representing days since 1899-12-30
    Text dates are in format like '1-Jul-25'
    """
    try:
        # Check if it's a numeric value (Excel date)
        numeric_val = pd.to_numeric(date_value, errors="coerce")
        if pd.notna(numeric_val):
            # Excel dates start from 1899-12-30
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=numeric_val)
        else:
            # Try parsing as string date
            return pd.to_datetime(date_value, format="%d-%b-%y", errors="coerce")
    except:
        return pd.NaT


def clean_offtake_data(input_file, output_file=None):
    """
    Clean and standardize offtake data

    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str, optional
        Path to save cleaned data. If None, returns DataFrame

    Returns:
    --------
    pd.DataFrame : Cleaned data
    """

    print("=" * 80)
    print("STARTING DATA CLEANING PROCESS")
    print("=" * 80)

    # Read the data
    print("\n1. Reading data...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"   ✓ Loaded {len(df):,} records with {len(df.columns)} columns")

    # Clean column names
    print("\n2. Cleaning column names...")
    df.columns = df.columns.str.strip()
    print("   ✓ Removed trailing spaces from column names")

    # Convert Bill Date
    print("\n3. Converting Bill Date column...")
    print("   This may take a moment...")
    df["Bill Date"] = df["Bill Date"].apply(convert_excel_date)
    success_count = df["Bill Date"].notna().sum()
    failed_count = df["Bill Date"].isna().sum()
    print(f"   ✓ Successfully converted: {success_count:,}")
    if failed_count > 0:
        print(f"   ⚠ Failed conversions: {failed_count}")

    # Convert Sell Value to numeric
    print("\n4. Converting Sell Value to numeric...")
    df["Sell Value"] = pd.to_numeric(df["Sell Value"], errors="coerce")
    print("   ✓ Sell Value converted to numeric")

    # Data quality checks
    print("\n5. Running data quality checks...")
    quality_issues = []

    # Check for missing dates
    missing_dates = df["Bill Date"].isna().sum()
    if missing_dates > 0:
        quality_issues.append(f"   ⚠ {missing_dates} records with missing Bill Date")

    # Check for negative values
    neg_sales = (df["Sell Value"] < 0).sum()
    if neg_sales > 0:
        quality_issues.append(f"   ⚠ {neg_sales} records with negative Sell Value")

    zero_qty = (df["Qty"] <= 0).sum()
    if zero_qty > 0:
        quality_issues.append(f"   ⚠ {zero_qty} records with zero or negative Quantity")

    # Check for high missing values
    missing_mbrand = df["M- Brand"].isna().sum()
    if missing_mbrand > len(df) * 0.1:  # More than 10%
        quality_issues.append(
            f"   ⚠ {missing_mbrand} records missing M-Brand ({missing_mbrand/len(df)*100:.1f}%)"
        )

    missing_segment = df["Segment"].isna().sum()
    if missing_segment > len(df) * 0.1:  # More than 10%
        quality_issues.append(
            f"   ⚠ {missing_segment} records missing Segment ({missing_segment/len(df)*100:.1f}%)"
        )

    if quality_issues:
        for issue in quality_issues:
            print(issue)
    else:
        print("   ✓ No major data quality issues detected")

    # Save cleaned data
    if output_file:
        print(f"\n6. Saving cleaned data...")
        df.to_csv(output_file, index=False)
        print(f"   ✓ Cleaned data saved to: {output_file}")

    print("\n" + "=" * 80)
    print("DATA CLEANING COMPLETE")
    print("=" * 80)

    return df


def generate_summary_report(df):
    """Generate a summary report of the cleaned data"""

    print("\n" + "=" * 80)
    print("DATA SUMMARY REPORT")
    print("=" * 80)

    print("\nDATA OVERVIEW:")
    print(f"  Total Records: {len(df):,}")
    print(f"  Date Range: {df['Bill Date'].min()} to {df['Bill Date'].max()}")
    print(f"  Duration: {(df['Bill Date'].max() - df['Bill Date'].min()).days} days")

    print("\nKEY METRICS:")
    print(f"  Unique Stores: {df['Store Code'].nunique()}")
    print(f"  Unique Brands: {df['Brand'].nunique()}")
    print(f"  Unique Products: {df['Product Name'].nunique()}")
    print(f"  Unique Categories: {df['Category'].nunique()}")

    print("\nSALES SUMMARY:")
    print(f"  Total Quantity: {df['Qty'].sum():,} units")
    print(f"  Total Sales Value: ₹{df['Sell Value'].sum():,.2f}")
    print(f"  Average Transaction: ₹{df['Sell Value'].mean():,.2f}")

    print("\nTOP 5 CATEGORIES:")
    top_cats = (
        df.groupby("Category")["Sell Value"].sum().sort_values(ascending=False).head(5)
    )
    for i, (cat, val) in enumerate(top_cats.items(), 1):
        print(f"  {i}. {cat:<30} ₹{val:>12,.2f}")

    print("\nTOP 5 STORES:")
    top_stores = (
        df.groupby(["Store Code", "Store Name"])["Sell Value"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )
    for i, ((code, name), val) in enumerate(top_stores.items(), 1):
        print(f"  {i}. Store {code} - {name:<35} ₹{val:>12,.2f}")


if __name__ == "__main__":
    # Example usage
    input_file = (
        "/Users/abhilasha/Downloads/H&G X HBDH - Offtake Trend.xlsx - Base (2).csv"
    )
    output_file = "/Users/abhilasha/Documents/client-projects/data_analysis/cleaned_offtake_data.csv"

    # Clean the data
    df = clean_offtake_data(input_file, output_file)

    # Generate summary report
    generate_summary_report(df)
