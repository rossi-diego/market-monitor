import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------------------------------------
# Helpers internos (não alteram a API pública)
# ------------------------------------------------------------
def _empty_fig_plotly(msg="Poucos pontos para plotar", height=420):
    fig = go.Figure()
    fig.update_layout(title=msg, template="plotly_dark", height=height)
    return fig


# ------------------------------------------------------------
# Relação + Rolling STD (Matplotlib)
# ------------------------------------------------------------
def plot_ratio_std_plt(x, y, title="", ylabel="", rolling_window=90, label_series="Série"):
    # --- estatísticas e janelas móveis ---
    y_max, y_min = y.max(), y.min()
    y_rolling_mean = y.rolling(window=rolling_window, min_periods=1).mean()
    y_rolling_std  = y.rolling(window=rolling_window, min_periods=1).std()
    idx_max, idx_min = y.idxmax(), y.idxmin()
    y_last, x_last = y.iloc[-1], x.iloc[-1]

    # --- figura ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(20, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    # Subplot 1: série + média + anotações
    ax1.plot(x, y, marker="D", linewidth=0.8, markersize=0.1, label=label_series)
    ax1.plot(x, y_rolling_mean, linewidth=1.2, label=f"Média Móvel ({rolling_window} dias)")
    ax1.axhline(y=y_max, linestyle="--", linewidth=1.2, label=f"Máx: {y_max:.2f}")
    ax1.axhline(y=y_min, linestyle="--", linewidth=1.2, label=f"Mín: {y_min:.2f}")

    ann_box = dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7)
    ax1.annotate(f"{y_max:.2f}", (x.loc[idx_max], y_max), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=9, bbox=ann_box)
    ax1.annotate(f"{y_min:.2f}", (x.loc[idx_min], y_min), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=9, bbox=ann_box)
    ax1.annotate(f"{y_last:.2f}", (x_last, y_last), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=9, bbox=ann_box)

    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # Subplot 2: Rolling STD
    ax2.plot(x, y_rolling_std, linewidth=1.2, label=f"Desvio Padrão ({rolling_window} dias)")
    ax2.set_ylabel("Rolling STD", fontsize=12)
    ax2.set_xlabel("Data", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left")

    fig.autofmt_xdate()
    return fig


# ------------------------------------------------------------
# Relação + Rolling STD (Plotly)
# ------------------------------------------------------------
def plot_ratio_std_plotly(
    x: pd.Series,
    y: pd.Series,
    title: str = "",
    ylabel: str = "",
    rolling_window: int = 90,
    label_series: str = "Série",
):
    # --- prepara dados ---
    x = pd.to_datetime(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    df = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if df.shape[0] < 2:
        return _empty_fig_plotly()

    # --- estatísticas e janelas móveis ---
    minp = max(2, rolling_window // 4)
    y_roll_mean = df["y"].rolling(rolling_window, min_periods=minp).mean()
    y_roll_std  = df["y"].rolling(rolling_window, min_periods=minp).std()

    idx_max = df["y"].idxmax(); idx_min = df["y"].idxmin()
    x_max, y_max = df.loc[idx_max, "x"], df.loc[idx_max, "y"]
    x_min, y_min = df.loc[idx_min, "x"], df.loc[idx_min, "y"]
    x_last, y_last = df["x"].iloc[-1], df["y"].iloc[-1]

    # --- figura com 2 subplots ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.08
    )

    # Série principal + média móvel
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=df["y"], mode="lines", name=label_series,
            hovertemplate="%{x|%d/%m/%Y}<br>Valor: %{y:.2f}<extra></extra>"
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=y_roll_mean, mode="lines",
            name=f"Média Móvel ({rolling_window}d)", line=dict(dash="dash"),
            hovertemplate="%{x|%d/%m/%Y}<br>Média: %{y:.2f}<extra></extra>"
        ), row=1, col=1
    )

    # Máx/Mín + marcadores
    fig.add_hline(y=y_max, line_dash="dot", line_width=1.5,
                  annotation_text=f"Máx {y_max:.2f}", annotation_position="top left",
                  row=1, col=1)
    fig.add_hline(y=y_min, line_dash="dot", line_width=1.5,
                  annotation_text=f"Mín {y_min:.2f}", annotation_position="bottom left",
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[x_max], y=[y_max], mode="markers+text",
                             text=[f"{y_max:.2f}"], textposition="top center",
                             name="Máx", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x_min], y=[y_min], mode="markers+text",
                             text=[f"{y_min:.2f}"], textposition="bottom center",
                             name="Mín", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x_last], y=[y_last], mode="markers+text",
                             text=[f"{y_last:.2f}"], textposition="top center",
                             name="Último", showlegend=False), row=1, col=1)

    # Subplot 2: Rolling STD
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=y_roll_std, mode="lines",
            name=f"Desvio Padrão ({rolling_window}d)",
            hovertemplate="%{x|%d/%m/%Y}<br>STD: %{y:.2f}<extra></extra>"
        ), row=2, col=1
    )

    # Layout / eixos
    fig.update_layout(
        template="plotly_dark",
        height=560,
        margin=dict(l=40, r=20, t=60, b=30),
        title=title or "Relação",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text=ylabel or "Valor", row=1, col=1)
    fig.update_yaxes(title_text="Rolling STD", row=2, col=1)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        ),
        rangeslider_visible=False,
        showspikes=True, spikemode="across", spikesnap="cursor",
        row=2, col=1
    )
    return fig


# ------------------------------------------------------------
# Preço + RSI (Matplotlib)
# ------------------------------------------------------------
def plot_price_rsi_plt(
    df,
    title="BOc1",
    date_col="date",
    close_col="boc1",
    rsi_col="RSI",          # se não existir, passe rsi_fn
    rsi_fn=None,            # ex.: sua src.utils.rsi
    rsi_len=14,
    ma_window=90,
    # ---- estilos
    price_color="#4A77FF",
    ma_color="#1f7a1f",
    rsi_color="#546E7A",
    # ---- bollinger
    show_bollinger=False,
    bands_window=20,
    bands_sigma=2.0,
    bands_color="#4A77FF",
    theme="transparent",
):
    """
    Figura com 2 subplots:
      (1) Preço + MM + Máx/Mín/Último + (opcional) Bollinger
      (2) RSI com faixas 30/50/70 e áreas
    """
    # Prep
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    x = data[date_col]
    y = pd.to_numeric(data[close_col], errors="coerce")
    ma = y.rolling(ma_window, min_periods=1).mean()

    # RSI
    if rsi_col in data.columns:
        rsi_series = pd.to_numeric(data[rsi_col], errors="coerce")
    elif callable(rsi_fn):
        rsi_df = rsi_fn(data[[date_col, close_col]].copy(),
                        ticker_col=close_col, date_col=date_col, window=rsi_len)
        rsi_series = pd.to_numeric(rsi_df["RSI"], errors="coerce")
    else:
        rsi_series = pd.Series(index=data.index, dtype=float)

    y_max, y_min = float(np.nanmax(y)), float(np.nanmin(y))
    idx_max, idx_min = int(np.nanargmax(y)), int(np.nanargmin(y))
    y_last, x_last = float(y.iloc[-1]), x.iloc[-1]

    # Figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    # (1) Preço
    ax1.plot(x, y, linewidth=1.05, alpha=0.95, color=price_color, label="Preço (close)")
    ax1.plot(x, ma, linewidth=1.6, color=ma_color, label=f"Média Móvel ({ma_window} dias)")

    if show_bollinger:
        m = y.rolling(bands_window, min_periods=bands_window).mean()
        s = y.rolling(bands_window, min_periods=bands_window).std()
        upper = m + bands_sigma * s
        lower = m - bands_sigma * s
        mask = (~upper.isna()) & (~lower.isna())
        ax1.plot(x, upper, color=bands_color, linewidth=0.9, alpha=0.8)
        ax1.plot(x, lower, color=bands_color, linewidth=0.9, alpha=0.8)
        ax1.fill_between(x, lower, upper, where=mask, color=bands_color, alpha=0.10,
                         interpolate=True, label=f"Bollinger ({bands_window}, {bands_sigma}σ)")

    ax1.axhline(y=y_max, linestyle="--", linewidth=1.2, color="#FF8C00", label=f"Máximo: {y_max:.2f}")
    ax1.axhline(y=y_min, linestyle="--", linewidth=1.2, color="#FF8C00", label=f"Mínima: {y_min:.2f}")
    ax1.axhline(y=y_last, linestyle=":", linewidth=1.0, color=price_color, alpha=0.35)

    ann_box = dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85)
    ax1.annotate(f"{y_max:.2f}", (x.iloc[idx_max], y_max), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=9, bbox=ann_box)
    ax1.annotate(f"{y_min:.2f}", (x.iloc[idx_min], y_min), xytext=(0, -16),
                 textcoords="offset points", ha="center", fontsize=9, bbox=ann_box)
    ax1.annotate(f"{y_last:.2f}", (x_last, y_last), xytext=(6, 0),
                 textcoords="offset points", va="center", fontsize=9, bbox=ann_box)

    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel("Preço", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.28)
    ax1.legend(loc="upper left")
    ax1.margins(x=0.01)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (2) RSI
    ax2.axhspan(70, 100, alpha=0.08, color="red")
    ax2.axhspan(0, 30,   alpha=0.08, color="green")
    ax2.axhline(70, linestyle="--", linewidth=1.0, color="#E57373", alpha=0.85)
    ax2.axhline(50, linestyle="--", linewidth=1.0, color="#9E9E9E", alpha=0.7)
    ax2.axhline(30, linestyle="--", linewidth=1.0, color="#81C784", alpha=0.85)

    ax2.plot(x, rsi_series, linewidth=1.6, color=rsi_color, label="RSI", zorder=3)
    ax2.fill_between(x, rsi_series, 50, where=~np.isnan(rsi_series),
                     color=rsi_color, alpha=0.07, zorder=2)

    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", fontsize=12)
    ax2.set_xlabel("Data", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.28)
    ax2.legend(loc="upper left")
    ax2.margins(x=0.01)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    fig.update_layout(
    title=dict(text=CLOSE.upper(), x=0.0, xanchor="left", y=0.98, yanchor="top", pad=dict(b=12)),
    margin=dict(t=80)
    )

    return fig


# ------------------------------------------------------------
# Preço + RSI (Plotly)
# ------------------------------------------------------------
def plot_price_rsi_plotly(
    df: pd.DataFrame,
    title: str = "BOC1",
    date_col: str = "date",
    close_col: str = "boc1",
    rsi_col: str = "RSI",       # se não existir, passe rsi_fn
    rsi_fn=None,                # ex.: sua utils.rsi(df, ticker_col, date_col, window)
    rsi_len: int = 14,
    ma_window: int = 90,
    # estilos
    price_color: str = "#4A77FF",
    ma_color: str = "#1f7a1f",
    rsi_color: str = "#546E7A",
    # bollinger
    show_bollinger: bool = False,
    bands_window: int = 20,
    bands_sigma: float = 2.0,
):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.sort_values(date_col)

    # força numérico
    y = pd.to_numeric(data[close_col], errors="coerce")
    x = data[date_col]
    base = pd.DataFrame({"x": x, "y": y}).dropna()
    if base.shape[0] < 2:
        return _empty_fig_plotly(height=460)

    # média móvel
    minp = max(2, ma_window // 4)
    ma = base["y"].rolling(ma_window, min_periods=minp).mean()

    # RSI
    if rsi_col in data.columns:
        rsi_series = pd.to_numeric(data[rsi_col], errors="coerce")
        rsi_series = pd.Series(rsi_series.values, index=data.index)
    elif callable(rsi_fn):
        rsi_df = rsi_fn(
            data[[date_col, close_col]].copy(),
            ticker_col=close_col, date_col=date_col, window=rsi_len
        )
        rsi_series = pd.to_numeric(rsi_df["RSI"], errors="coerce")
        rsi_series = pd.Series(rsi_series.values, index=rsi_df.index)
        rsi_series = pd.Series(rsi_series.values, index=data.index)  # realinha
    else:
        rsi_series = pd.Series(index=data.index, dtype=float)

    # estatísticas
    idx_max = base["y"].idxmax(); idx_min = base["y"].idxmin()
    x_max, y_max = base.loc[idx_max, "x"], float(base.loc[idx_max, "y"])
    x_min, y_min = base.loc[idx_min, "x"], float(base.loc[idx_min, "y"])
    x_last, y_last = base["x"].iloc[-1], float(base["y"].iloc[-1])

    # figura
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.08
    )

    # (1) Preço + MM
    fig.add_trace(
        go.Scatter(
            x=base["x"], y=base["y"], mode="lines",
            name="Preço (close)", line=dict(width=1.1, color=price_color),
            hovertemplate="%{x|%d/%m/%Y}<br>Preço: %{y:.2f}<extra></extra>"
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=base["x"], y=ma, mode="lines",
            name=f"Média Móvel ({ma_window}d)", line=dict(width=1.4, dash="dash", color=ma_color),
            hovertemplate="%{x|%d/%m/%Y}<br>MM: %{y:.2f}<extra></extra>"
        ), row=1, col=1
    )

    if show_bollinger:
        m = base["y"].rolling(bands_window, min_periods=bands_window).mean()
        s = base["y"].rolling(bands_window, min_periods=bands_window).std()
        upper = m + bands_sigma * s
        lower = m - bands_sigma * s
        fig.add_trace(
            go.Scatter(x=base["x"], y=upper, mode="lines",
                       line=dict(width=0.9, color=price_color, dash="dot"),
                       name=f"Bollinger ({bands_window}, {bands_sigma}σ)"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=base["x"], y=lower, mode="lines",
                       line=dict(width=0.9, color=price_color, dash="dot"),
                       showlegend=False, fill="tonexty", opacity=0.12),
            row=1, col=1
        )

    # Máx/Mín/Último
    fig.add_hline(y=y_max, line_dash="dot", line_width=1.2,
                  annotation_text=f"Máx {y_max:.2f}", annotation_position="top left",
                  row=1, col=1)
    fig.add_hline(y=y_min, line_dash="dot", line_width=1.2,
                  annotation_text=f"Mín {y_min:.2f}", annotation_position="bottom left",
                  row=1, col=1)
    fig.add_trace(
        go.Scatter(x=[x_max], y=[y_max], mode="markers+text",
                   text=[f"{y_max:.2f}"], textposition="top center",
                   marker=dict(size=8, color=price_color), showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[x_min], y=[y_min], mode="markers+text",
                   text=[f"{y_min:.2f}"], textposition="bottom center",
                   marker=dict(size=8, color=price_color), showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[x_last], y=[y_last], mode="markers+text",
                   text=[f"{y_last:.2f}"], textposition="middle right",
                   marker=dict(size=7, color=price_color), showlegend=False),
        row=1, col=1
    )

    # (2) RSI
    fig.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)
    fig.add_shape(
        type="rect", xref="x2 domain", yref="y2",
        x0=0, x1=1, y0=70, y1=100,
        fillcolor="rgba(229,115,115,0.20)", line_width=0, layer="below"
    )
    fig.add_shape(
        type="rect", xref="x2 domain", yref="y2",
        x0=0, x1=1, y0=0, y1=30,
        fillcolor="rgba(129,199,132,0.20)", line_width=0, layer="below"
    )
    fig.add_hline(y=70, line_dash="dash", line_width=1.0, line_color="#E57373", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_width=1.0, line_color="#9E9E9E", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_width=1.0, line_color="#81C784", row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=data[date_col], y=rsi_series, mode="lines",
            name="RSI", line=dict(width=1.6, color=rsi_color),
            fill="tonexty", fillcolor="rgba(84,110,122,0.06)",
            hovertemplate="%{x|%d/%m/%Y}<br>RSI: %{y:.1f}<extra></extra>"
        ), row=2, col=1
    )

    # Layout / interações
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=40, r=20, t=60, b=30),
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        ),
        rangeslider=dict(visible=False),
        showspikes=True, spikemode="across", spikesnap="cursor",
        row=2, col=1
    )
    return fig
