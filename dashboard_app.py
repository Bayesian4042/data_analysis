import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Cache data loading for performance
@st.cache_data
def load_data():
    """Load the aggregated offtake data"""
    df = pd.read_csv(
        os.path.join(BASE_DIR, "aggregated_offtake_data.csv"),
        low_memory=False,
    )
    df["Bill Date"] = pd.to_datetime(df["Bill Date"])

    # Add derived columns
    df["Is_Weekend"] = df["Bill Date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["Day_Name"] = df["Bill Date"].dt.day_name()
    df["Month_Name"] = df["Bill Date"].dt.strftime("%B %Y")
    df["Week"] = df["Bill Date"].dt.isocalendar().week

    return df


@st.cache_data
def load_forecast_data():
    """Load Prophet forecast results"""
    try:
        # Load SKU-level forecast summary (Sept 21 - Oct 20, 2025)
        test_summary = pd.read_csv(
            os.path.join(BASE_DIR, "top244_skus_forecast_summary.csv")
        )

        # Load SKU-level forecast details
        forecast_details = pd.read_csv(
            os.path.join(BASE_DIR, "top244_skus_forecast_details.csv")
        )
        forecast_details["Date"] = pd.to_datetime(forecast_details["Date"])

        # Load accuracy tables
        accuracy_summary = pd.read_csv(
            os.path.join(BASE_DIR, "sku_accuracy_summary_table.csv")
        )

        return test_summary, forecast_details, accuracy_summary
    except Exception as e:
        st.error(f"Error loading forecast data: {str(e)}")
        return None, None, None


@st.cache_data
def load_sku_analysis_data():
    """Load SKU analysis data (daily, weekly, monthly)"""
    try:
        # Load daily data with category info
        df_daily = pd.read_csv(
            os.path.join(BASE_DIR, "sku_analysis/df_with_category.csv")
        )
        df_daily["Bill Date"] = pd.to_datetime(df_daily["Bill Date"])

        # Load weekly data
        df_weekly = pd.read_csv(os.path.join(BASE_DIR, "sku_analysis/df_weekly.csv"))

        # Load monthly data
        df_monthly = pd.read_csv(os.path.join(BASE_DIR, "sku_analysis/df_monthly.csv"))

        return df_daily, df_weekly, df_monthly
    except Exception as e:
        st.error(f"Error loading SKU analysis data: {str(e)}")
        return None, None, None


# Load data
with st.spinner("Loading data..."):
    df = load_data()
    test_summary, forecast_details, accuracy_summary = load_forecast_data()
    df_daily, df_weekly, df_monthly = load_sku_analysis_data()

# Sidebar filters
st.sidebar.markdown("## Sales Forecasting Dashboard")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

# Search by SKU Code
sku_search = st.sidebar.text_input(
    "Search by SKU Code",
    placeholder="Enter SKU code (e.g., 548919)",
    help="Enter specific SKU code to filter",
)

# Search by Store Code
store_search = st.sidebar.text_input(
    "Search by Store Code",
    placeholder="Enter Store code (e.g., 16)",
    help="Enter specific Store code to filter",
)

# Date range filter
min_date = df["Bill Date"].min().date()
max_date = df["Bill Date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Apply filters
df_filtered = df.copy()

# Apply SKU filter
if sku_search:
    try:
        sku_code = int(sku_search)
        df_filtered = df_filtered[df_filtered["SKU Code"] == sku_code]
        if len(df_filtered) == 0:
            st.sidebar.warning(f"No data found for SKU Code: {sku_code}")
    except ValueError:
        st.sidebar.error("Please enter a valid numeric SKU code")

# Apply Store filter
if store_search:
    try:
        store_code = int(store_search)
        df_filtered = df_filtered[df_filtered["Store Code"] == store_code]
        if len(df_filtered) == 0:
            st.sidebar.warning(f"No data found for Store Code: {store_code}")
    except ValueError:
        st.sidebar.error("Please enter a valid numeric Store code")

# Apply date range filter
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_filtered[
        (df_filtered["Bill Date"].dt.date >= start_date)
        & (df_filtered["Bill Date"].dt.date <= end_date)
    ]

# Category filter (only if no SKU search)
if not sku_search:
    all_categories = ["All Categories"] + sorted(
        df_filtered["Category"].unique().tolist()
    )
    selected_categories = st.sidebar.multiselect(
        "Select Categories", options=all_categories, default=["All Categories"]
    )

    if "All Categories" not in selected_categories and len(selected_categories) > 0:
        df_filtered = df_filtered[df_filtered["Category"].isin(selected_categories)]

# Brand filter (only if no SKU search)
if not sku_search:
    all_brands = ["All Brands"] + sorted(
        df_filtered["Brand"].dropna().unique().tolist()
    )
    selected_brands = st.sidebar.multiselect(
        "Select Brands", options=all_brands, default=["All Brands"]
    )

    if "All Brands" not in selected_brands and len(selected_brands) > 0:
        df_filtered = df_filtered[df_filtered["Brand"].isin(selected_brands)]

# Day type filter
day_type = st.sidebar.radio(
    "Day Type", options=["All Days", "Weekdays Only", "Weekends Only"]
)

if day_type == "Weekdays Only":
    df_filtered = df_filtered[df_filtered["Is_Weekend"] == 0]
elif day_type == "Weekends Only":
    df_filtered = df_filtered[df_filtered["Is_Weekend"] == 1]

# Main title
st.markdown(
    '<h1 class="main-header">Sales Forecasting Dashboard</h1>',
    unsafe_allow_html=True,
)

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Overview",
        "Product Analysis",
        "Store Performance",
        "Time Series",
        "Forecasts",
        "SKU/Basepack Analysis",
    ]
)

# TAB 1: OVERVIEW
with tab1:
    st.header("Executive Summary")

    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_qty = df_filtered["Qty"].sum()
        st.metric(
            "Total Quantity Sold",
            f"{total_qty:,.0f}",
            help="Total units sold in selected period",
        )

    with col2:
        total_skus = df_filtered["SKU Code"].nunique()
        st.metric("Total SKUs", f"{total_skus}", help="Number of unique SKU codes")

    with col3:
        avg_daily_sales = df_filtered.groupby("Bill Date")["Qty"].sum().mean()
        st.metric(
            "Avg Daily Sales",
            f"{avg_daily_sales:,.0f}",
            help="Average units sold per day",
        )

    with col4:
        total_stores = df_filtered["Store Code"].nunique()
        st.metric(
            "Active Stores", f"{total_stores}", help="Number of stores with sales"
        )

    with col5:
        total_products = df_filtered["Basepack Code"].nunique()
        st.metric("Products", f"{total_products}", help="Number of unique basepacks")

    st.markdown("---")

    # Two column layout for charts
    col1, col2 = st.columns(2)

    with col1:
        # Daily sales trend
        daily_sales = df_filtered.groupby("Bill Date")["Qty"].sum().reset_index()
        fig = px.line(
            daily_sales,
            x="Bill Date",
            y="Qty",
            title="Daily Sales Trend",
            labels={"Qty": "Quantity", "Bill Date": "Date"},
        )
        fig.update_traces(line_color="#1f77b4", line_width=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top 10 products
        product_sales = df_filtered.groupby("Product Name")["Qty"].sum().reset_index()
        product_sales = product_sales.nlargest(10, "Qty")
        fig = px.bar(
            product_sales,
            y="Product Name",
            x="Qty",
            title="Top 10 Products",
            labels={"Qty": "Total Quantity", "Product Name": "Product"},
            orientation="h",
        )
        fig.update_traces(marker_color="#2ecc71")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Weekend vs Weekday and Category analysis
    col1, col2 = st.columns(2)

    with col1:
        weekend_sales = df_filtered.groupby("Is_Weekend")["Qty"].sum().reset_index()
        weekend_sales["Day_Type"] = weekend_sales["Is_Weekend"].map(
            {0: "Weekday", 1: "Weekend"}
        )
        fig = px.pie(
            weekend_sales,
            values="Qty",
            names="Day_Type",
            title="Sales: Weekday vs Weekend",
            color_discrete_sequence=["#3498db", "#e74c3c"],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Category sales
        category_sales = df_filtered.groupby("Category")["Qty"].sum().reset_index()
        fig = px.pie(
            category_sales,
            values="Qty",
            names="Category",
            title="Sales by Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Weekly pattern
    st.subheader("Weekly Sales Pattern")
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    dow_sales = df_filtered.groupby("Day_Name")["Qty"].sum().reset_index()
    dow_sales["Day_Name"] = pd.Categorical(
        dow_sales["Day_Name"], categories=day_order, ordered=True
    )
    dow_sales = dow_sales.sort_values("Day_Name")

    fig = px.bar(
        dow_sales,
        x="Day_Name",
        y="Qty",
        title="Average Sales by Day of Week",
        labels={"Qty": "Total Quantity", "Day_Name": "Day"},
        color="Qty",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Monthly trend
    st.subheader("Monthly Sales Trend")
    df_filtered_monthly = df_filtered.copy()
    df_filtered_monthly["Month_Year"] = (
        df_filtered_monthly["Bill Date"].dt.to_period("M").apply(lambda r: r.start_time)
    )
    monthly_sales = df_filtered_monthly.groupby("Month_Year")["Qty"].sum().reset_index()
    monthly_sales["Month_Name"] = monthly_sales["Month_Year"].dt.strftime("%B %Y")

    fig = px.line(
        monthly_sales,
        x="Month_Year",
        y="Qty",
        title="Monthly Sales Trend",
        labels={"Qty": "Total Quantity", "Month_Year": "Month"},
        markers=True,
    )
    fig.update_traces(line_color="#9b59b6", line_width=3, marker_size=8)
    fig.update_layout(
        height=400, hovermode="x unified", xaxis_title="Month", yaxis_title="Quantity"
    )
    st.plotly_chart(fig, use_container_width=True)


# TAB 2: PRODUCT ANALYSIS
with tab2:
    st.header("Product Performance Analysis")

    # Top products selector
    top_n = st.slider("Select number of top products to display", 5, 30, 20)

    top_products = (
        df_filtered.groupby("Product Name")["Qty"].sum().nlargest(top_n).index.tolist()
    )

    selected_product = st.selectbox(
        "Select a Product for Detailed Analysis",
        options=top_products,
    )

    product_data = df_filtered[df_filtered["Product Name"] == selected_product]

    # Product KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        prod_qty = product_data["Qty"].sum()
        st.metric("Total Quantity", f"{prod_qty:,.0f}")

    with col2:
        prod_days = product_data["Bill Date"].nunique()
        st.metric("Days with Sales", f"{prod_days}")

    with col3:
        prod_stores = product_data["Store Code"].nunique()
        st.metric("Stores Selling", f"{prod_stores}")

    with col4:
        avg_daily = prod_qty / prod_days if prod_days > 0 else 0
        st.metric("Avg Daily Sales", f"{avg_daily:.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Daily sales for selected product
        prod_daily = product_data.groupby("Bill Date")["Qty"].sum().reset_index()
        fig = px.area(
            prod_daily,
            x="Bill Date",
            y="Qty",
            title=f"Daily Sales: {selected_product[:50]}",
            labels={"Qty": "Quantity", "Bill Date": "Date"},
        )
        fig.update_traces(
            fill="tozeroy", line_color="#e74c3c", fillcolor="rgba(231, 76, 60, 0.3)"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top stores for this product
        prod_stores = product_data.groupby("Store Name")["Qty"].sum().reset_index()
        prod_stores = prod_stores.nlargest(10, "Qty")
        fig = px.bar(
            prod_stores,
            x="Qty",
            y="Store Name",
            title=f"Top 10 Stores",
            orientation="h",
        )
        fig.update_traces(marker_color="#f39c12")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Product comparison
    st.subheader("Compare All Products")

    product_metrics = (
        df_filtered.groupby(["Product Name", "Brand", "Category"])
        .agg(
            {
                "Qty": "sum",
                "Bill Date": "nunique",
                "Store Code": "nunique",
            }
        )
        .reset_index()
    )
    product_metrics.columns = [
        "Product",
        "Brand",
        "Category",
        "Total Qty",
        "Days Active",
        "Store Count",
    ]
    product_metrics["Avg Daily Sales"] = (
        product_metrics["Total Qty"] / product_metrics["Days Active"]
    )
    product_metrics = product_metrics.sort_values("Total Qty", ascending=False)

    st.dataframe(
        product_metrics.style.format(
            {
                "Total Qty": "{:,.0f}",
                "Days Active": "{:,.0f}",
                "Store Count": "{:,.0f}",
                "Avg Daily Sales": "{:.1f}",
            }
        ).background_gradient(subset=["Total Qty"], cmap="Blues"),
        use_container_width=True,
        height=400,
    )


# TAB 3: STORE PERFORMANCE
with tab3:
    st.header("Store Performance Analysis")

    # Top stores
    col1, col2 = st.columns([2, 1])

    with col1:
        store_perf = (
            df_filtered.groupby("Store Name")
            .agg({"Qty": "sum", "Bill Date": "nunique", "SKU Code": "nunique"})
            .reset_index()
        )
        store_perf.columns = [
            "Store",
            "Total Qty",
            "Days Active",
            "SKU Count",
        ]
        store_perf = store_perf.sort_values("Total Qty", ascending=False).head(15)

        fig = px.bar(
            store_perf,
            x="Total Qty",
            y="Store",
            title="Top 15 Stores by Quantity Sold",
            orientation="h",
            color="SKU Count",
            color_continuous_scale="viridis",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Store Metrics")
        top_store = store_perf.iloc[0]
        st.metric("Top Store", top_store["Store"])
        st.metric("Qty Sold", f"{top_store['Total Qty']:,.0f}")
        st.metric("Days Active", f"{top_store['Days Active']:,.0f}")
        st.metric("SKU Count", f"{top_store['SKU Count']:,.0f}")

    # Store comparison table
    st.subheader("Store Performance Table")

    store_full = (
        df_filtered.groupby("Store Name")
        .agg(
            {
                "Qty": "sum",
                "Bill Date": lambda x: x.nunique(),
                "SKU Code": "nunique",
            }
        )
        .reset_index()
    )
    store_full.columns = [
        "Store Name",
        "Total Qty",
        "Days Active",
        "SKUs Sold",
    ]
    store_full["Avg Daily Sales"] = store_full["Total Qty"] / store_full["Days Active"]
    store_full = store_full.sort_values("Total Qty", ascending=False)

    st.dataframe(
        store_full.style.format(
            {
                "Total Qty": "{:,.0f}",
                "Days Active": "{:,.0f}",
                "SKUs Sold": "{:,.0f}",
                "Avg Daily Sales": "{:.1f}",
            }
        ).background_gradient(subset=["Total Qty"], cmap="Greens"),
        use_container_width=True,
        height=400,
    )


# TAB 4: TIME SERIES
with tab4:
    st.header("Time Series Analysis")

    # Granularity selector
    granularity = st.radio(
        "Select Time Granularity",
        options=["Daily", "Weekly", "Monthly"],
        horizontal=True,
    )

    # Prepare time series data
    if granularity == "Daily":
        ts_data = df_filtered.groupby("Bill Date")["Qty"].sum().reset_index()
        ts_data.columns = ["Date", "Quantity"]
        ts_data["7-Day MA"] = ts_data["Quantity"].rolling(window=7).mean()
        ts_data["30-Day MA"] = ts_data["Quantity"].rolling(window=30).mean()
    elif granularity == "Weekly":
        df_filtered["Week"] = (
            df_filtered["Bill Date"].dt.to_period("W").apply(lambda r: r.start_time)
        )
        ts_data = df_filtered.groupby("Week")["Qty"].sum().reset_index()
        ts_data.columns = ["Date", "Quantity"]
    else:  # Monthly
        df_filtered["Month"] = (
            df_filtered["Bill Date"].dt.to_period("M").apply(lambda r: r.start_time)
        )
        ts_data = df_filtered.groupby("Month")["Qty"].sum().reset_index()
        ts_data.columns = ["Date", "Quantity"]

    # Main time series plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts_data["Date"],
            y=ts_data["Quantity"],
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color="#3498db", width=2),
            marker=dict(size=4),
        )
    )

    if granularity == "Daily":
        fig.add_trace(
            go.Scatter(
                x=ts_data["Date"],
                y=ts_data["7-Day MA"],
                mode="lines",
                name="7-Day Moving Avg",
                line=dict(color="#e74c3c", width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts_data["Date"],
                y=ts_data["30-Day MA"],
                mode="lines",
                name="30-Day Moving Avg",
                line=dict(color="#2ecc71", width=2),
            )
        )

    fig.update_layout(
        title=f"{granularity} Sales Trend with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Quantity",
        height=500,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Category-wise time series
    st.subheader("Category-wise Time Series")

    category_ts = (
        df_filtered.groupby(["Bill Date", "Category"])["Qty"].sum().reset_index()
    )

    fig = px.line(
        category_ts,
        x="Bill Date",
        y="Qty",
        color="Category",
        title="Daily Sales by Category",
        labels={"Qty": "Quantity"},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# TAB 5: FORECASTS
with tab5:
    st.header("Prophet Forecasting Results")

    if test_summary is not None and len(test_summary) > 0:
        # Testing results
        st.subheader("Model Validation (Sept 21 - Oct 20, 2025)")

        col1, col2, col3, col4, col5 = st.columns(5)

        # Calculate volume accuracy
        if accuracy_summary is not None and len(accuracy_summary) > 0:
            excellent_skus = len(
                accuracy_summary[accuracy_summary["Volume_Error_%"] < 20]
            )
            good_skus = len(
                accuracy_summary[
                    (accuracy_summary["Volume_Error_%"] >= 20)
                    & (accuracy_summary["Volume_Error_%"] < 40)
                ]
            )
        else:
            excellent_skus = 0
            good_skus = 0

        with col1:
            total_actual = test_summary["Actual_Total"].sum()
            st.metric("Total Actual", f"{total_actual:,.0f} units")

        with col2:
            total_forecast = test_summary["Forecast_Total"].sum()
            st.metric("Total Forecast", f"{total_forecast:,.0f} units")

        with col3:
            accuracy = 100 - abs(total_actual - total_forecast) / total_actual * 100
            st.metric("Overall Accuracy", f"{accuracy:.2f}%")

        with col4:
            st.metric("Excellent SKUs", f"{excellent_skus}", help="Volume error < 20%")

        with col5:
            st.metric("Good SKUs", f"{good_skus}", help="Volume error 20-40%")

        st.markdown("---")

        # Category performance
        st.subheader("Category-Level Performance")

        cat_performance = (
            test_summary.groupby("Category")
            .agg({"Actual_Total": "sum", "Forecast_Total": "sum", "SKU_Code": "count"})
            .reset_index()
        )
        cat_performance.columns = [
            "Category",
            "Actual_Total",
            "Forecast_Total",
            "SKU_Count",
        ]
        cat_performance["Error_%"] = (
            abs(cat_performance["Actual_Total"] - cat_performance["Forecast_Total"])
            / cat_performance["Actual_Total"]
            * 100
        ).round(2)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Actual",
                x=cat_performance["Category"],
                y=cat_performance["Actual_Total"],
                marker_color="#3498db",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Forecast",
                x=cat_performance["Category"],
                y=cat_performance["Forecast_Total"],
                marker_color="#2ecc71",
            )
        )
        fig.update_layout(
            title="Actual vs Forecast by Category (Sept 21 - Oct 20, 2025)",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show category table
        st.dataframe(
            cat_performance.style.format(
                {
                    "Actual_Total": "{:,.0f}",
                    "Forecast_Total": "{:,.0f}",
                    "SKU_Count": "{:,.0f}",
                    "Error_%": "{:.2f}%",
                }
            ).background_gradient(subset=["Error_%"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        # Brand performance
        st.subheader("Brand-Level Performance")

        brand_performance = (
            test_summary.groupby("Brand")
            .agg({"Actual_Total": "sum", "Forecast_Total": "sum", "SKU_Code": "count"})
            .reset_index()
        )
        brand_performance.columns = [
            "Brand",
            "Actual_Total",
            "Forecast_Total",
            "SKU_Count",
        ]
        brand_performance["Error_%"] = (
            abs(brand_performance["Actual_Total"] - brand_performance["Forecast_Total"])
            / brand_performance["Actual_Total"]
            * 100
        ).round(2)
        brand_performance = brand_performance.sort_values(
            "Actual_Total", ascending=False
        ).head(10)

        st.dataframe(
            brand_performance.style.format(
                {
                    "Actual_Total": "{:,.0f}",
                    "Forecast_Total": "{:,.0f}",
                    "SKU_Count": "{:,.0f}",
                    "Error_%": "{:.2f}%",
                }
            ).background_gradient(subset=["Error_%"], cmap="RdYlGn_r"),
            use_container_width=True,
        )

        # Top and Bottom Performers
        st.markdown("---")
        st.subheader("SKU Performance Analysis")

        if accuracy_summary is not None and len(accuracy_summary) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Top 10 Best Performers")
                top_10 = accuracy_summary.nsmallest(10, "Volume_Error_%")[
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

                st.dataframe(
                    top_10.style.format(
                        {
                            "SKU_Code": "{:,.0f}",
                            "Actual_Total": "{:,.0f}",
                            "Forecast_Total": "{:,.0f}",
                            "Volume_Error_%": "{:.2f}%",
                        }
                    ).background_gradient(subset=["Volume_Error_%"], cmap="RdYlGn_r"),
                    use_container_width=True,
                    height=400,
                )

            with col2:
                st.markdown("### Bottom 10 (Need Attention)")
                bottom_10 = accuracy_summary.nlargest(10, "Volume_Error_%")[
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

                st.dataframe(
                    bottom_10.style.format(
                        {
                            "SKU_Code": "{:,.0f}",
                            "Actual_Total": "{:,.0f}",
                            "Forecast_Total": "{:,.0f}",
                            "Volume_Error_%": "{:.2f}%",
                        }
                    ).background_gradient(subset=["Volume_Error_%"], cmap="RdYlGn_r"),
                    use_container_width=True,
                    height=400,
                )

        # Daily forecast for top SKU
        if forecast_details is not None and len(forecast_details) > 0:
            st.markdown("---")
            st.subheader("Daily Forecast Visualization")

            # Get top SKU by volume
            top_sku = test_summary.nlargest(1, "Actual_Total").iloc[0]
            top_sku_code = top_sku["SKU_Code"]

            daily_forecast = forecast_details[
                forecast_details["SKU_Code"] == top_sku_code
            ].copy()

            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(
                    x=daily_forecast["Date"],
                    y=daily_forecast["Actual"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="#e74c3c", width=2),
                    marker=dict(size=6),
                )
            )

            # Add forecast
            fig.add_trace(
                go.Scatter(
                    x=daily_forecast["Date"],
                    y=daily_forecast["Forecast"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#3498db", width=2),
                    marker=dict(size=6),
                )
            )

            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=daily_forecast["Date"],
                    y=daily_forecast["Upper_Bound"],
                    mode="lines",
                    name="Upper Bound",
                    line=dict(color="lightgray", width=1, dash="dash"),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=daily_forecast["Date"],
                    y=daily_forecast["Lower_Bound"],
                    mode="lines",
                    name="Confidence Interval",
                    line=dict(color="lightgray", width=1, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(68, 68, 68, 0.1)",
                )
            )

            fig.update_layout(
                title=f"Daily Actual vs Forecast: {top_sku['Product_Name'][:60]}<br>Volume Error: {accuracy_summary[accuracy_summary['SKU_Code']==top_sku_code]['Volume_Error_%'].iloc[0]:.2f}%",
                xaxis_title="Date",
                yaxis_title="Quantity",
                height=500,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # SKU selector for custom visualization
            st.markdown("---")
            st.subheader("  Explore Individual SKU Forecasts")

            sku_options = test_summary.nlargest(50, "Actual_Total")[
                ["SKU_Code", "Product_Name"]
            ]
            sku_options["Display"] = (
                sku_options["SKU_Code"].astype(str)
                + " - "
                + sku_options["Product_Name"].str[:60]
            )

            selected_sku_display = st.selectbox(
                "Select SKU to visualize:",
                options=sku_options["Display"].tolist(),
            )

            selected_sku_code = int(selected_sku_display.split(" - ")[0])
            selected_sku_data = forecast_details[
                forecast_details["SKU_Code"] == selected_sku_code
            ].copy()
            selected_sku_info = test_summary[
                test_summary["SKU_Code"] == selected_sku_code
            ].iloc[0]

            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Actual Total", f"{selected_sku_info['Actual_Total']:,.0f}")
            with col2:
                st.metric(
                    "Forecast Total", f"{selected_sku_info['Forecast_Total']:,.0f}"
                )
            with col3:
                vol_error = accuracy_summary[
                    accuracy_summary["SKU_Code"] == selected_sku_code
                ]["Volume_Error_%"].iloc[0]
                st.metric("Volume Error", f"{vol_error:.2f}%")
            with col4:
                st.metric("MAE", f"{selected_sku_info['MAE']:.2f}")

            # Plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=selected_sku_data["Date"],
                    y=selected_sku_data["Actual"],
                    mode="lines+markers",
                    name="Actual",
                    line=dict(color="#e74c3c", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=selected_sku_data["Date"],
                    y=selected_sku_data["Forecast"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#3498db", width=2),
                )
            )
            fig.update_layout(
                title=f"Daily Forecast: {selected_sku_info['Product_Name']}",
                xaxis_title="Date (Sept 21 - Oct 20, 2025)",
                yaxis_title="Quantity",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "Forecast data not available. Please ensure forecast files are generated."
        )


# TAB 6: SKU/BASEPACK ANALYSIS
with tab6:
    st.header("SKU/Basepack Analysis")

    if df_daily is not None and df_weekly is not None and df_monthly is not None:

        # Add high-level stability analysis section
        st.subheader("SKU Stability Overview")

        # Get unique SKU-level data with stability metrics
        sku_stability = df_daily[
            [
                "SKU Code",
                "Product Name",
                "Brand",
                "Category",
                "months_with_sales",
                "category",
            ]
        ].drop_duplicates(subset=["SKU Code"])

        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_skus = sku_stability["SKU Code"].nunique()
            st.metric("Total SKUs", f"{total_skus:,}")

        with col2:
            avg_months = sku_stability["months_with_sales"].mean()
            st.metric("Avg Months Active", f"{avg_months:.1f}")

        with col3:
            consistent_skus = len(
                sku_stability[sku_stability["months_with_sales"] >= 6]
            )
            st.metric("Consistent SKUs (≥6 months)", f"{consistent_skus:,}")

        with col4:
            pct_consistent = (
                (consistent_skus / total_skus * 100) if total_skus > 0 else 0
            )
            st.metric("% Consistent", f"{pct_consistent:.1f}%")

        st.markdown("---")

        # Two column layout for visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Distribution by stability category
            category_counts = sku_stability["category"].value_counts().reset_index()
            category_counts.columns = ["Stability Category", "SKU Count"]

            fig = px.bar(
                category_counts,
                x="SKU Count",
                y="Stability Category",
                orientation="h",
                title="SKU Distribution by Stability Category",
                color="SKU Count",
                color_continuous_scale="Blues",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Distribution by months with sales
            months_counts = (
                sku_stability["months_with_sales"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            months_counts.columns = ["Months Active", "SKU Count"]

            fig = px.bar(
                months_counts,
                x="Months Active",
                y="SKU Count",
                title="SKU Distribution by Months Active",
                color="SKU Count",
                color_continuous_scale="Greens",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed breakdown section
        st.subheader("🔍 Detailed SKU Breakdown by Stability")

        # Filter by stability category
        selected_stability = st.selectbox(
            "Select Stability Category:",
            options=["All Categories"]
            + sorted(sku_stability["category"].unique().tolist()),
        )

        if selected_stability == "All Categories":
            filtered_stability = sku_stability.copy()
        else:
            filtered_stability = sku_stability[
                sku_stability["category"] == selected_stability
            ].copy()

        # Additional filters
        col1, col2 = st.columns(2)

        with col1:
            min_months = st.slider(
                "Minimum Months Active:",
                min_value=int(sku_stability["months_with_sales"].min()),
                max_value=int(sku_stability["months_with_sales"].max()),
                value=int(sku_stability["months_with_sales"].min()),
            )

        with col2:
            selected_category_filter = st.multiselect(
                "Filter by Product Category:",
                options=sorted(sku_stability["Category"].unique().tolist()),
                default=[],
            )

        # Apply filters
        filtered_stability = filtered_stability[
            filtered_stability["months_with_sales"] >= min_months
        ]

        if len(selected_category_filter) > 0:
            filtered_stability = filtered_stability[
                filtered_stability["Category"].isin(selected_category_filter)
            ]

        # Show metrics for filtered data
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Filtered SKU Count", f"{len(filtered_stability):,}")

        with col2:
            avg_months_filtered = (
                filtered_stability["months_with_sales"].mean()
                if len(filtered_stability) > 0
                else 0
            )
            st.metric("Avg Months Active", f"{avg_months_filtered:.1f}")

        with col3:
            if len(filtered_stability) > 0:
                most_common_cat = (
                    filtered_stability["category"].mode()[0]
                    if len(filtered_stability["category"].mode()) > 0
                    else "N/A"
                )
                st.metric("Most Common Category", most_common_cat)

        # Display the filtered SKU table
        st.subheader("SKU Details")

        display_columns = [
            "SKU Code",
            "Product Name",
            "Brand",
            "Category",
            "months_with_sales",
            "category",
        ]
        display_df = filtered_stability[display_columns].copy()
        display_df.columns = [
            "SKU Code",
            "Product Name",
            "Brand",
            "Product Category",
            "Months Active",
            "Stability Category",
        ]
        display_df = display_df.sort_values("Months Active", ascending=False)

        st.dataframe(
            display_df.style.format(
                {"SKU Code": "{:,.0f}", "Months Active": "{:,.0f}"}
            ).background_gradient(subset=["Months Active"], cmap="RdYlGn"),
            use_container_width=True,
            height=400,
        )

        # Download button for filtered data
        csv = display_df.to_csv(index=False)
        # st.download_button(
        #     label="📥 Download Filtered SKU List (CSV)",
        #     data=csv,
        #     file_name=f"sku_stability_{selected_stability.replace(' ', '_')}.csv",
        #     mime="text/csv",
        # )

        st.markdown("---")
        st.markdown("---")

        # Original individual SKU analysis section
        st.header("Individual SKU/Basepack Analysis")
        # Create basepack to SKU mapping
        basepack_sku_mapping = df_daily[
            [
                "Basepack Code",
                "Basepack Desc",
                "SKU Code",
                "Product Name",
                "Brand",
                "Category",
            ]
        ].drop_duplicates()

        # Selection method
        col1, col2 = st.columns([1, 3])

        with col1:
            selection_method = st.radio(
                "Select by:",
                options=["Basepack Code", "SKU Code"],
                help="Choose whether to search by Basepack Code or SKU Code",
            )

        with col2:
            if selection_method == "Basepack Code":
                # Get unique basepacks
                basepack_options = basepack_sku_mapping[
                    ["Basepack Code", "Basepack Desc"]
                ].drop_duplicates()
                basepack_options["Basepack Code"] = pd.to_numeric(
                    basepack_options["Basepack Code"], errors="coerce"
                )
                basepack_options = basepack_options.dropna(subset=["Basepack Code"])
                basepack_options["Basepack Code"] = basepack_options[
                    "Basepack Code"
                ].astype(int)
                basepack_options["Display"] = (
                    basepack_options["Basepack Code"].astype(str)
                    + " - "
                    + basepack_options["Basepack Desc"].str[:60]
                )
                basepack_options = basepack_options.sort_values("Basepack Code")

                selected_display = st.selectbox(
                    "Select Basepack Code:",
                    options=basepack_options["Display"].tolist(),
                )

                selected_code = int(selected_display.split(" - ")[0])
                selected_sku = basepack_sku_mapping[
                    basepack_sku_mapping["Basepack Code"] == selected_code
                ]["SKU Code"].iloc[0]
                selected_info = basepack_sku_mapping[
                    basepack_sku_mapping["Basepack Code"] == selected_code
                ].iloc[0]

            else:  # SKU Code
                # Get unique SKUs
                sku_options = basepack_sku_mapping[
                    ["SKU Code", "Product Name", "Basepack Code"]
                ].drop_duplicates()
                sku_options["SKU Code"] = pd.to_numeric(
                    sku_options["SKU Code"], errors="coerce"
                )
                sku_options = sku_options.dropna(subset=["SKU Code"])
                sku_options["SKU Code"] = sku_options["SKU Code"].astype(int)
                sku_options["Display"] = (
                    sku_options["SKU Code"].astype(str)
                    + " - "
                    + sku_options["Product Name"].str[:60]
                )
                sku_options = sku_options.sort_values("SKU Code")

                selected_display = st.selectbox(
                    "Select SKU Code:",
                    options=sku_options["Display"].tolist(),
                )

                selected_sku = int(selected_display.split(" - ")[0])
                selected_info = basepack_sku_mapping[
                    basepack_sku_mapping["SKU Code"] == selected_sku
                ].iloc[0]
                selected_code = selected_info["Basepack Code"]

        # Display product information
        st.markdown("---")
        st.subheader("Product Information")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("SKU Code", f"{selected_sku}")
        with col2:
            st.metric("Basepack Code", f"{selected_code}")
        with col3:
            st.metric("Brand", f"{selected_info['Brand']}")
        with col4:
            st.metric("Category", f"{selected_info['Category']}")

        st.markdown(f"**Product Name:** {selected_info['Product Name']}")
        st.markdown(f"**Basepack Description:** {selected_info['Basepack Desc']}")

        st.markdown("---")

        # Time granularity selection
        analysis_type = st.radio(
            "Select Time Granularity:",
            options=["All", "Daily", "Weekly", "Monthly"],
            horizontal=True,
        )

        # Filter data for selected SKU
        daily_data = df_daily[df_daily["SKU Code"] == selected_sku].copy()
        weekly_data = df_weekly[df_weekly["SKU Code"] == selected_sku].copy()
        monthly_data = df_monthly[df_monthly["SKU Code"] == selected_sku].copy()

        if analysis_type == "All":
            # Show all three views
            st.subheader("Daily Analysis")
            if len(daily_data) > 0:
                daily_agg = daily_data.groupby("Bill Date")["Qty"].sum().reset_index()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Qty (Daily)", f"{daily_agg['Qty'].sum():,.0f}")
                with col2:
                    st.metric("Avg Daily Qty", f"{daily_agg['Qty'].mean():.2f}")
                with col3:
                    st.metric("Days with Sales", f"{len(daily_agg)}")

                fig = px.line(
                    daily_agg,
                    x="Bill Date",
                    y="Qty",
                    title=f'Daily Quantity Trend - {selected_info["Product Name"][:50]}',
                    labels={"Qty": "Quantity", "Bill Date": "Date"},
                    markers=True,
                )
                fig.update_traces(line_color="#3498db", line_width=2, marker_size=5)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily data available for this SKU")

            st.markdown("---")
            st.subheader("Weekly Analysis")
            if len(weekly_data) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Qty (Weekly)", f"{weekly_data['Qty'].sum():,.0f}")
                with col2:
                    st.metric("Avg Weekly Qty", f"{weekly_data['Qty'].mean():.2f}")
                with col3:
                    st.metric("Weeks with Sales", f"{len(weekly_data)}")

                fig = px.line(
                    weekly_data,
                    x="Week",
                    y="Qty",
                    title=f'Weekly Quantity Trend - {selected_info["Product Name"][:50]}',
                    labels={"Qty": "Quantity", "Week": "Week Number"},
                    markers=True,
                )
                fig.update_traces(line_color="#2ecc71", line_width=2, marker_size=6)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No weekly data available for this SKU")

            st.markdown("---")
            st.subheader("Monthly Analysis")
            if len(monthly_data) > 0:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Qty (Monthly)", f"{monthly_data['Qty'].sum():,.0f}"
                    )
                with col2:
                    st.metric("Avg Monthly Qty", f"{monthly_data['Qty'].mean():.2f}")
                with col3:
                    st.metric("Months with Sales", f"{len(monthly_data)}")

                # Add month names for better readability
                monthly_data["Month_Name"] = pd.to_datetime(
                    monthly_data["Month"], format="%m"
                ).dt.strftime("%B")

                fig = px.line(
                    monthly_data,
                    x="Month_Name",
                    y="Qty",
                    title=f'Monthly Quantity Trend - {selected_info["Product Name"][:50]}',
                    labels={"Qty": "Quantity", "Month_Name": "Month"},
                    markers=True,
                )
                fig.update_traces(line_color="#e74c3c", line_width=2, marker_size=6)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No monthly data available for this SKU")

        elif analysis_type == "Daily":
            st.subheader("Daily Analysis")
            if len(daily_data) > 0:
                daily_agg = daily_data.groupby("Bill Date")["Qty"].sum().reset_index()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Qty", f"{daily_agg['Qty'].sum():,.0f}")
                with col2:
                    st.metric("Avg Daily Qty", f"{daily_agg['Qty'].mean():.2f}")
                with col3:
                    st.metric("Max Daily Qty", f"{daily_agg['Qty'].max():,.0f}")
                with col4:
                    st.metric("Days with Sales", f"{len(daily_agg)}")

                fig = px.line(
                    daily_agg,
                    x="Bill Date",
                    y="Qty",
                    title=f'Daily Quantity Trend - {selected_info["Product Name"][:50]}',
                    labels={"Qty": "Quantity", "Bill Date": "Date"},
                    markers=True,
                )
                fig.update_traces(line_color="#3498db", line_width=2, marker_size=6)
                fig.update_layout(height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # Show store-wise daily analysis
                st.subheader("Store-wise Daily Sales")
                store_daily = (
                    daily_data.groupby(["Store Name", "Bill Date"])["Qty"]
                    .sum()
                    .reset_index()
                )
                top_stores = (
                    daily_data.groupby("Store Name")["Qty"]
                    .sum()
                    .nlargest(5)
                    .index.tolist()
                )
                store_daily_top = store_daily[
                    store_daily["Store Name"].isin(top_stores)
                ]

                fig = px.line(
                    store_daily_top,
                    x="Bill Date",
                    y="Qty",
                    color="Store Name",
                    title="Daily Sales by Top 5 Stores",
                    labels={"Qty": "Quantity", "Bill Date": "Date"},
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily data available for this SKU")

        elif analysis_type == "Weekly":
            st.subheader("Weekly Analysis")
            if len(weekly_data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Qty", f"{weekly_data['Qty'].sum():,.0f}")
                with col2:
                    st.metric("Avg Weekly Qty", f"{weekly_data['Qty'].mean():.2f}")
                with col3:
                    st.metric("Max Weekly Qty", f"{weekly_data['Qty'].max():,.0f}")
                with col4:
                    st.metric("Weeks with Sales", f"{len(weekly_data)}")

                fig = px.line(
                    weekly_data,
                    x="Week",
                    y="Qty",
                    title=f'Weekly Quantity Trend - {selected_info["Product Name"][:50]}',
                    labels={"Qty": "Quantity", "Week": "Week Number"},
                    markers=True,
                )
                fig.update_traces(line_color="#2ecc71", line_width=2, marker_size=6)
                fig.update_layout(height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # Statistical summary
                st.subheader("Weekly Statistics")
                stats_df = pd.DataFrame(
                    {
                        "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Total"],
                        "Value": [
                            f"{weekly_data['Qty'].mean():.2f}",
                            f"{weekly_data['Qty'].median():.2f}",
                            f"{weekly_data['Qty'].std():.2f}",
                            f"{weekly_data['Qty'].min():.0f}",
                            f"{weekly_data['Qty'].max():.0f}",
                            f"{weekly_data['Qty'].sum():.0f}",
                        ],
                    }
                )
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.info("No weekly data available for this SKU")

        elif analysis_type == "Monthly":
            st.subheader("Monthly Analysis")
            if len(monthly_data) > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Qty", f"{monthly_data['Qty'].sum():,.0f}")
                with col2:
                    st.metric("Avg Monthly Qty", f"{monthly_data['Qty'].mean():.2f}")
                with col3:
                    st.metric("Max Monthly Qty", f"{monthly_data['Qty'].max():,.0f}")
                with col4:
                    st.metric("Months with Sales", f"{len(monthly_data)}")

                # Add month names for better readability
                monthly_data["Month_Name"] = pd.to_datetime(
                    monthly_data["Month"], format="%m"
                ).dt.strftime("%B")

                fig = px.line(
                    monthly_data,
                    x="Month_Name",
                    y="Qty",
                    title=f'Monthly Quantity Trend - {selected_info["Product Name"][:50]}',
                    labels={"Qty": "Quantity", "Month_Name": "Month"},
                    markers=True,
                )
                fig.update_traces(line_color="#e74c3c", line_width=2, marker_size=6)
                fig.update_layout(height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                # Show data table
                st.subheader("Monthly Data Table")
                monthly_display = monthly_data[["Month_Name", "Qty"]].copy()
                monthly_display.columns = ["Month", "Quantity"]
                st.dataframe(
                    monthly_display.style.format(
                        {"Quantity": "{:,.0f}"}
                    ).background_gradient(subset=["Quantity"], cmap="Blues"),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No monthly data available for this SKU")

    else:
        st.warning(
            "SKU analysis data not available. Please ensure the data files are in the sku_analysis folder."
        )


# Footer
st.markdown("---")
st.markdown(
    f"""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><strong>Sales Forecasting Dashboard</strong> | Built with Streamlit & Plotly</p>
    <p>Data Period: {df['Bill Date'].min().strftime('%B %Y')} - {df['Bill Date'].max().strftime('%B %Y')} | 
    {len(df):,} records | {df['SKU Code'].nunique()} SKUs</p>
</div>
""",
    unsafe_allow_html=True,
)
