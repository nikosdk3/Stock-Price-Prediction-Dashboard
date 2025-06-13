import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None

    def check_stationarity(self, timeseries):
        result = adfuller(timeseries)
        return result[1] <= 0.05

    def find_best_order(self, data, max_p=5, max_d=2, max_q=5):
        best_aic = np.inf
        best_order = None

        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue

        return best_order if best_order else (1, 1, 1)

    def train(self, data, target_column='Close', order=None):
        timeseries = data[target_column].values

        if order is None:
            order = self.find_best_order(timeseries)

        try:
            self.model = ARIMA(timeseries, order=order)
            self.fitted_model = self.model.fit()
        except Exception as e:
            raise ValueError(f"Failed to train ARIMA model: ")
