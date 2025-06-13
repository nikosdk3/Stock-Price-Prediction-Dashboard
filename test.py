import pandas as pd

last_date = pd.Timestamp("2025-06-13")
forecast_days = 5

forecast_dates = pd.date_range(
    start=last_date + pd.offsets.Day(1),
    periods=forecast_days,
    freq='D'
)

print(forecast_dates)
