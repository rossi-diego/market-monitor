import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def rsi(df, ticker_col, date_col='date', window=14):
    df = df.copy()
    df.sort_values(by=date_col, inplace=True)

    df['delta'] = df[ticker_col].diff()

    # Separar ganhos e perdas
    df['gain'] = df['delta'].where(df['delta'] > 0, 0)
    df['loss'] = -df['delta'].where(df['delta'] < 0, 0)

    # Média dos ganhos e perdas
    df['avg_gain'] = df['gain'].rolling(window=window).mean()
    df['avg_loss'] = df['loss'].rolling(window=window).mean()

    # RS e RSI
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['RSI'] = 100 - (100 / (1 + df['rs']))

    # Retornar apenas as colunas úteis
    return df[[date_col, ticker_col, 'RSI']]

def plot_ratio_std(x, y, title="", ylabel="", rolling_window=90, label_series="Série"):
    # Estatísticas
    y_max, y_min = y.max(), y.min()
    y_rolling_mean = y.rolling(window=rolling_window, min_periods=1).mean()
    y_rolling_std  = y.rolling(window=rolling_window, min_periods=1).std()

    # Índices para máximo e mínimo
    idx_max, idx_min = y.idxmax(), y.idxmin()

    # Figura
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        constrained_layout=True
    )

    # Subplot 1
    ax1.plot(x, y, marker='D', linewidth=0.8, markersize=0.1, label=label_series)
    ax1.plot(x, y_rolling_mean, linewidth=1.2, label=f'Média Móvel ({rolling_window} dias)')
    ax1.axhline(y=y_max, linestyle='--', linewidth=1.2, label=f'Máx: {y_max:.2f}')
    ax1.axhline(y=y_min, linestyle='--', linewidth=1.2, label=f'Mín: {y_min:.2f}')
    ax1.annotate(f"{y_max:.2f}", (x.loc[idx_max], y_max), xytext=(0, 10), textcoords="offset points",
                 ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    ax1.annotate(f"{y_min:.2f}", (x.loc[idx_min], y_min), xytext=(0, 10), textcoords="offset points",
                 ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    y_last, x_last = y.iloc[-1], x.iloc[-1]
    ax1.annotate(f"{y_last:.2f}", (x_last, y_last), xytext=(0, 10), textcoords="offset points",
                 ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')

    # Subplot 2
    ax2.plot(x, y_rolling_std, linewidth=1.2, label=f'Desvio Padrão ({rolling_window} dias)')
    ax2.set_ylabel('Rolling STD', fontsize=12)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left')

    fig.autofmt_xdate()
    return fig

def plot_price_rsi(
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
    rsi_color="#546E7A",     # <- RSI mais suave (azul-acinzentado)
    # ---- bollinger
    show_bollinger=False,
    bands_window=20,
    bands_sigma=2.0,
    bands_color="#4A77FF",
    theme="transparent",
):
    """
    Figura com 2 subplots:
      (1) Preço (close) + MM(ma_window) + Máx/Mín/Último + (opcional) Bollinger (bands_window, bands_sigma)
      (2) RSI com faixas 30/50/70, zonas coloridas (verde/vermelho) e linha discreta.

    Retorna fig (Matplotlib).
    """
    # --------------------------
    # Prep
    # --------------------------
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)

    x = data[date_col]
    y = pd.to_numeric(data[close_col], errors="coerce")
    ma = y.rolling(ma_window, min_periods=1).mean()

    # --- RSI: usa coluna existente; se não houver e rsi_fn for dado, calcula
    if rsi_col in data.columns:
        rsi_series = pd.to_numeric(data[rsi_col], errors="coerce")
    elif callable(rsi_fn):
        # sua rsi(df, ticker_col, date_col='date', window=14) retorna um DF com coluna 'RSI'
        rsi_df = rsi_fn(data[[date_col, close_col]].copy(),
                        ticker_col=close_col, date_col=date_col, window=rsi_len)
        rsi_series = pd.to_numeric(rsi_df["RSI"], errors="coerce")
    else:
        rsi_series = pd.Series(index=data.index, dtype=float)

    y_max, y_min = float(np.nanmax(y)), float(np.nanmin(y))
    idx_max, idx_min = int(np.nanargmax(y)), int(np.nanargmin(y))
    y_last, x_last = float(y.iloc[-1]), x.iloc[-1]

    # --------------------------
    # Figure
    # --------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    # --------------------------
    # (1) Preço + MM + (opcional) Bollinger
    # --------------------------
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

    # Máx / Mín / Último
    ax1.axhline(y=y_max, linestyle="--", linewidth=1.2, color="#FF8C00", label=f"Máximo: {y_max:.2f}")
    ax1.axhline(y=y_min, linestyle="--", linewidth=1.2, color="#FF8C00", label=f"Mínima: {y_min:.2f}")
    ax1.axhline(y=y_last, linestyle=":", linewidth=1.0, color=price_color, alpha=0.35)

    ax1.annotate(f"{y_max:.2f}", (x.iloc[idx_max], y_max), xytext=(0, 10), textcoords="offset points",
                 ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    ax1.annotate(f"{y_min:.2f}", (x.iloc[idx_min], y_min), xytext=(0, -16), textcoords="offset points",
                 ha="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    ax1.annotate(f"{y_last:.2f}", (x_last, y_last), xytext=(6, 0), textcoords="offset points",
                 va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel("Preço", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.28)
    ax1.legend(loc="upper left")
    ax1.margins(x=0.01)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # --------------------------
    # (2) RSI: faixas verde/vermelha + linha mais discreta
    # --------------------------
    ax2.axhspan(70, 100, alpha=0.08, color="red")     # zona sobrecomprado
    ax2.axhspan(0, 30, alpha=0.08, color="green")     # zona sobrevendido
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
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # Datas compactas
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    return fig