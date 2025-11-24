#!/bin/bash
# Sales Analytics Dashboard Launcher

echo "Launching Sales Analytics Dashboard..."
echo "=================================="
echo ""
echo "Dashboard Features:"
echo "  - Interactive filters (date, product, store)"
echo "  - 6 analysis tabs (Overview, Products, Stores, Time Series, Forecasts, Discounts)"
echo "  - Real-time data exploration"
echo "  - Prophet forecast results"
echo ""
echo "The dashboard will open in your default browser"
echo "Default URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "=================================="
echo ""

# Activate virtual environment if it exists
if [ -d "data_analysis" ]; then
    source data_analysis/bin/activate
fi

# Launch Streamlit
streamlit run dashboard_app.py

