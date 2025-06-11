import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
            vertical_spacing=0.1,
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
            xaxis_title="Date",
            template="plotly_white",
            height=600,
            showlegend=False,
        )

        return fig
