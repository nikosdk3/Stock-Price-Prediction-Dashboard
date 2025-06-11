import streamlit as st
from data_loader import DataLoader
from visualize import Visualizer
from datetime import datetime, timedelta


def main():
    st.set_page_config(
        page_title="Stock Price Prediction Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📈 Stock Price Prediction Dashboard")
    st.markdown("---")

    data_loader = DataLoader()
    visualizer = Visualizer()

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)",
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", value=datetime.now() - timedelta(days=365 * 2)
            )
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())

        model_type = st.selectbox("Model Type", ["LSTM", "ARIMA", "Both"])

        forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30)

        if model_type in ["LSTM", "ARIMA"]:
            st.subheader("LSTM Parameters")
            lookback_period = st.slider("Lookback Period", min_value=30, max_value=120, value=60)
            epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=50)

        load_data = st.button("Load Data", type="primary")

if __name__ == "__main__":
    main()
