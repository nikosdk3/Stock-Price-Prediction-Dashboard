# Stock Price Prediction Dashboard

A Streamlit web application for visualizing and forecasting stock prices using LSTM and ARIMA models.

## Features

- **Data Loading:** Fetch historical stock data using Yahoo Finance.
- **Visualization:** Interactive charts for stock price history and model predictions.
- **Model Training:** Train and evaluate LSTM (deep learning) and ARIMA (statistical) models.
- **Backtesting:** Visualize and compare model predictions with actual prices for a selected period.
- **Metrics:** View MAE, RMSE, and MAPE for model performance.
- **User Controls:** Select ticker, date range, forecast days, model type, and model hyperparameters.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Stock-Price-Prediction-Dashboard.git
   cd Stock-Price-Prediction-Dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the provided local URL.

## File Structure

- `app.py` — Main Streamlit app.
- `data_loader.py` — Loads stock data from Yahoo Finance.
- `models/`
  - `LSTM.py` — LSTM model class and training logic.
  - `ARIMA.py` — ARIMA model class and training logic.
  - `utils.py` — Utility functions (e.g., metrics).
- `visualize.py` — Plotting and visualization functions.
- `requirements.txt` — Python dependencies.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Troubleshooting

- If you see warnings from statsmodels (ARIMA), they are common and usually do not stop the app.
- For best results, ensure your data is stationary for ARIMA and experiment with LSTM hyperparameters.

## License

MIT License
