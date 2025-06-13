import streamlit as st
from data_loader import DataLoader
from visualize import Visualizer
from datetime import datetime, timedelta
from models.LSTM import LSTMModel
from models.utils import calculate_metrics


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

        load_data = st.button("Load Data", type="primary")

        if model_type in ["LSTM", "ARIMA"]:
            st.subheader("Model Parameters")
            lookback_period = st.slider(
                "Lookback Period", min_value=30, max_value=120, value=60
            )
            epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=50)
            hidden_size = st.slider(
                "Hidden Size", min_value=32, max_value=128, value=50
            )
            num_layers = st.slider(
                "Number of LSTM Layers", min_value=1, max_value=4, value=3
            )
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001,
            )

    if load_data or "stock_data" in st.session_state:
        if load_data:
            if "ticker" in st.session_state and st.session_state["ticker"] != ticker:
                st.cache_data.clear()

            with st.spinner("Loading stock data..."):
                data = data_loader.load_stock_data(ticker, start_date, end_date)
                if data is not None:
                    st.session_state["stock_data"] = data
                    st.session_state["ticker"] = ticker
                    st.session_state["data_loader"] = data_loader
                else:
                    st.error(
                        "Failed to load data. Please check the ticker symbol and try again."
                    )
                    return
        data = st.session_state["stock_data"]

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📊 Data Overview",
                "🤖 Model Training",
                "🔮 Forecast Results",
                "📈 Metrics & Backtesting",
            ]
        )

        with tab1:
            st.header("Data Overview")

            if "data_loader" in st.session_state:
                data_loader = st.session_state["data_loader"]

            stock_info = data_loader.get_stock_info()

            if stock_info:
                st.metric("Company:", stock_info["name"])
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sector", stock_info["sector"])
                with col2:
                    st.metric("Industry", stock_info["industry"])
                with col3:
                    market_cap = stock_info["market_cap"]
                    if isinstance(market_cap, (int, float)):
                        market_cap = f"{market_cap/1e9:.1f}B"
                    st.metric("Market Cap", market_cap)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
            with col2:
                price_change = data["Close"].iloc[-1] - data["Close"].iloc[-2]
                percentage_change = (price_change / data["Close"].iloc[-2]) * 100
                st.metric(
                    "Daily Change",
                    f"{price_change:.2f}",
                    delta=f"{percentage_change:.2f}%",
                )
            with col3:
                st.metric("52W High", f"{data['High'].max():.2f}")
            with col4:
                st.metric("52W Low", f"{data['Low'].min():.2f}")

            fig = visualizer.plot_stock_data(data, f"{ticker} Stock Price History")
            st.plotly_chart(fig)

            st.subheader("Recent Data")
            st.dataframe(
                data.tail(10)[["Date", "Open", "High", "Low", "Close", "Volume"]]
            )

        with tab2:
            st.header("Model Training")
            if st.button("Train Models", type="primary"):
                results = {}
                if model_type in ["LSTM", "Both"]:
                    try:
                        lstm_model = LSTMModel(
                            lookback_period=lookback_period,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                        )
                        history, test_data = lstm_model.train(
                            data, epochs=epochs, learning_rate=learning_rate
                        )
                        y_test_actual, y_test_pred = lstm_model.eval_model(test_data)
                        lstm_metrics = calculate_metrics(y_test_actual, y_test_pred)

                        results["LSTM"] = {
                            "model": lstm_model,
                            "metrics": lstm_metrics,
                            "status": "success",
                        }

                    except Exception as e:
                        st.error(f"LSTM training failed: {str(e)}")
                        results["LSTM"] = {"status": "failed", "error": str(e)}

if __name__ == "__main__":
    main()
