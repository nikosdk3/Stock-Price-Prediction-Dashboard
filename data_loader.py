import pandas as pd
import yfinance as yf
import streamlit as st


class DataLoader:
    def __init__(self):
        self.ticker = None
        self.data = None

    @st.cache_data
    def load_stock_data(self, ticker, start_date, end_date):
        """Load stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")

            self.data = data
            self.ticker = ticker

            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def get_stock_info(self):
        if self.ticker:
            try:
                stock = yf.Ticker(self.ticker)
                info = stock.info
                return {
                    "name": info.get("longName", self.ticker),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "market_cap": info.get("marketCap", "N/A"),
                }
            except Exception as e:
                return {
                    "name": self.ticker,
                    "sector": "N/A",
                    "industry": "N/A",
                    "market_cap": "N/A",
                }
