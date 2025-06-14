from tempfile import template
from turtle import title
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


class Visualizer:
    def __init__(self):
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff9800",
            "info": "#17a2b8",
        }

    def plot_stock_data(self, data, title="Stock Price History"):
        """Plot Historical Data"""
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Price", "Volume"),
            vertical_spacing=0.3,
            row_heights=[0.7, 0.3],
        )

        # Price Plot
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color=self.colors["primary"], width=2),
            ),
            row=1,
            col=1,
        )

        # Volume plot
        fig.add_trace(
            go.Bar(
                x=data["Date"],
                y=data["Volume"],
                name="Volume",
                marker_color=self.colors["secondary"],
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=title,
            template="plotly_white",
            height=600,
            showlegend=False,
        )

        return fig

    def plot_predictions(
        self, historical_data, predictions, forecast_dates, model_type, true_values=None
    ):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=historical_data["Date"],
                y=historical_data["Close"],
                mode="lines",
                name="Historical",
                line=dict(color=self.colors["primary"], width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=predictions,
                mode="lines+markers",
                name=f"{model_type} Forecast",
                line=dict(color=self.colors["danger"], width=2, dash="dash"),
                marker=dict(size=6),
            )
        )

        fig.update_layout(
            title=f"Stock Price Forecast - {model_type} Model",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
        )

        return fig

    def plot_backtesting(self, actual, predicted, dates, model_type):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=actual,
                mode="lines",
                name="Actual",
                line=dict(color=self.colors["primary"], width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predicted,
                mode="lines",
                name="Predicted",
                line=dict(color=self.colors["danger"], width=2, dash="dash"),
            )
        )

        fig.update_layout(
            title=f"Backtesting Results - {model_type} Model",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
        )

        return fig

    def plot_lstm_accuracy(self, actual, predicted):
        dates = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=len(actual)),
            periods=len(actual),
        )
        return self.plot_backtesting(actual, predicted, dates, model_type="LSTM")

    def plot_metrics_comparison(self, metrics_dict):
        models = list(metrics_dict.keys())
        metrics = ["MAE", "RMSE", "MAPE"]

        fig = make_subplots(
            rows=1,
            cols=len(metrics),
            subplot_titles=metrics,
            specs=[[{"type": "bar"}] * len(metrics)],
        )

        colors = [self.colors["primary"], self.colors["secondary"]]

        for i, metric in enumerate(metrics):
            values = [metrics_dict[model][metric] for model in models]

            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                ),
                row=1,
                col=i + 1,
            )

        fig.update_layout(title="Model Performance Comparison", template="plotly_white")

        return fig
