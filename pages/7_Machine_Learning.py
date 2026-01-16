"""
Machine Learning Forecast Page
------------------------------

This page allows the user to:

1. Select an asset (target variable).
2. Choose which dataset columns will be used as FEATURES.
3. Choose number of LAGS (0 to 10).
4. Select ML model (Ridge, Random Forest, XGBoost).
5. View model performance vs actual (historical backtest).
6. Produce OUT-OF-SAMPLE FORECAST for up to 45 future days.

The page also includes explanations for:
- What are lags?
- How the model learns temporal structure.
- What is multi-step forecasting?
- What MAE, RMSE and R² represent.
- What normalization/standardization is and why we use it.
"""

# ============================================================
# Imports & Setup
# ============================================================
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.data_pipeline import df
from src.utils import apply_theme, date_range_picker, section
import plotly.graph_objects as go

# Theme
apply_theme()

# Page header
st.markdown("# 🔮 Machine Learning - Previsão de Preços")
st.markdown("Utilize algoritmos de Machine Learning para prever preços futuros de commodities com base em dados históricos")
st.divider()

# ============================================================
# Base Data
# ============================================================
BASE = df.copy()
BASE["date"] = pd.to_datetime(BASE["date"], errors="coerce")
BASE = BASE.sort_values("date")

# ============================================================
# Explanation Expander
# ============================================================
with st.expander("📘 Como funciona o modelo de Machine Learning?", expanded=False):
    st.markdown("""
    ### 🧠 Conceitos Fundamentais

    **1) Lags (Valores Históricos)**
    Lags são valores passados da própria série temporal que estamos prevendo.
    - **Lag 1** = preço de ontem (t-1)
    - **Lag 2** = preço de anteontem (t-2)
    - **Lag 5** = preço de 5 dias atrás (t-5)

    Os lags ajudam o modelo a capturar **tendência, momentum e autocorrelação** da série.

    **2) Features Externas (Variáveis Explicativas)**
    São outras variáveis que podem influenciar o preço do ativo alvo:
    - Dólar, commodities relacionadas, prêmios, etc.
    - Permitem ao modelo aprender relações causais ou correlações fortes

    **3) Multi-Step Forecasting (Previsão Iterativa)**
    Para prever vários dias à frente, o modelo:
    1. Prevê o próximo dia usando lags reais
    2. Usa essa previsão como novo lag para prever o dia seguinte
    3. Repete o processo até completar o horizonte de previsão

    ⚠️ **Importante**: Quanto mais dias à frente, maior a incerteza acumulada!

    ### 📊 Métricas de Avaliação

    | Métrica | Significado | Interpretação |
    |---------|-------------|---------------|
    | **MAE** | Erro médio absoluto | Erro típico em unidades do preço |
    | **RMSE** | Raiz do erro quadrático médio | Penaliza erros grandes, mesma unidade do preço |
    | **R²** | Coeficiente de determinação | % da variância explicada (0-1). Negativo = pior que média |
    | **MAPE** | Erro percentual médio absoluto | Erro em % - mais fácil de interpretar |

    ### 🎯 Escolhendo o Modelo

    - **Ridge**: Rápido, simples, bom para relações lineares
    - **Random Forest**: Robusto, captura não-linearidades, menos overfitting
    - **XGBoost**: Poderoso, melhor performance, requer mais dados
    """)

st.divider()

# ============================================================
# Configuration Section
# ============================================================
st.markdown("## ⚙️ Configuração do Modelo")

# Container 1: Target and Features
with st.container(border=True):
    st.markdown("### 🎯 Dados de Entrada")

    valid_cols = [c for c in BASE.columns if c not in ["date"] and BASE[c].dtype != "object"]

    col1, col2 = st.columns([1, 2])

    with col1:
        target_col = st.selectbox(
            "Ativo a prever (Target)",
            valid_cols,
            index=0,
            help="Variável que o modelo irá prever"
        )

    with col2:
        feature_cols = st.multiselect(
            "Features (variáveis explicativas)",
            options=valid_cols,
            default=[c for c in valid_cols if c != target_col][:3],
            help="Variáveis que o modelo usará para fazer previsões"
        )

    # At least 1 feature or lag must exist
    if len(feature_cols) == 0:
        st.warning("⚠️ Selecione ao menos uma feature ou configure lags abaixo.")

# Container 2: Model Configuration
with st.container(border=True):
    st.markdown("### 🤖 Configuração do Algoritmo")

    col_model, col_lags, col_horizon = st.columns(3)

    with col_model:
        models_dict = {
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor(n_estimators=500, random_state=42),
        }
        if HAS_XGB:
            models_dict["XGBoost"] = XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )

        model_label = st.selectbox(
            "Algoritmo",
            list(models_dict.keys()),
            help="Ridge: linear rápido | RF: não-linear robusto | XGB: máxima performance"
        )
        model = models_dict[model_label]

        # Model description
        model_descriptions = {
            "Ridge Regression": "✅ Rápido e interpretável\n✅ Bom para relações lineares\n⚠️ Pode não capturar não-linearidades",
            "Random Forest": "✅ Robusto a outliers\n✅ Captura não-linearidades\n✅ Menos propenso a overfitting",
            "XGBoost": "✅ Melhor performance geral\n✅ Captura padrões complexos\n⚠️ Requer mais dados para treinar"
        }
        st.caption(model_descriptions.get(model_label, ""))

    with col_lags:
        num_lags = st.slider(
            "Número de lags",
            min_value=0,
            max_value=10,
            value=5,
            step=1,
            help="Valores históricos do target: lag1=ontem, lag2=anteontem, etc."
        )
        if num_lags > 0:
            st.caption(f"✓ Usando {num_lags} valores históricos")
        else:
            st.caption("⚠️ Sem lags - apenas features externas")

    with col_horizon:
        horizon = st.slider(
            "Dias à frente",
            min_value=1,
            max_value=45,
            value=30,
            help="Quantidade de dias futuros a prever"
        )
        st.caption(f"🔮 Prevendo {horizon} dias")

# Container 3: Advanced Settings
with st.container(border=True):
    st.markdown("### ⚙️ Configurações Avançadas")

    col_norm, col_period = st.columns([1, 2])

    with col_norm:
        normalize = st.checkbox(
            "Normalizar dados (StandardScaler)",
            value=True,
            help="Recomendado: padroniza features para média=0 e std=1"
        )
        if normalize:
            st.caption("✓ Normalização ativada")
        else:
            st.caption("⚠️ Usando escala original")

    with col_period:
        start_model_date, end_model_date = date_range_picker(
            BASE["date"],
            state_key="ml_train_range",
            default_days=365 * 3,
        )

    mask_model = (BASE["date"].dt.date >= start_model_date) & (
        BASE["date"].dt.date <= end_model_date
    )
    BASE_RANGE = BASE.loc[mask_model].copy()

    if BASE_RANGE.empty:
        st.error("❌ Sem dados no período selecionado para treinar o modelo.")
        st.stop()

    st.caption(f"📊 Usando {len(BASE_RANGE)} dias de dados históricos ({start_model_date} a {end_model_date})")

# Check if we have enough features
if len(feature_cols) == 0 and num_lags == 0:
    st.error("❌ Configure ao menos uma feature ou um lag para treinar o modelo.")
    st.stop()

st.divider()


# ============================================================
# Prepare dataset (lags + features)
# ============================================================
df_ml = BASE_RANGE[["date", target_col] + feature_cols].copy()

# Generate lag columns (on target)
for lag in range(1, num_lags + 1):
    df_ml[f"{target_col}_lag{lag}"] = df_ml[target_col].shift(lag)

# Drop rows with NaN caused by lags or missing features
df_ml = df_ml.dropna().reset_index(drop=True)

if df_ml.empty:
    st.error("Dados insuficientes após aplicar lags e filtrar NaNs.")
    st.stop()

# Build X, y
X = df_ml.drop(columns=["date", target_col])
y = df_ml[target_col]

feature_names = X.columns.tolist()

# Train-test split (80/20, temporal)
split = int(len(df_ml) * 0.80)
X_train_raw, X_test_raw = X.iloc[:split].copy(), X.iloc[split:].copy()
y_train_raw, y_test_raw = y.iloc[:split].copy(), y.iloc[split:].copy()
dates_test = df_ml["date"].iloc[split:]

# Scalers (only if normalize=True)
x_scaler = None
y_scaler = None

if normalize:
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(
        y_train_raw.values.reshape(-1, 1)
    ).ravel()

    # Fit on normalized data
    model.fit(X_train, y_train_scaled)

    # Predict on normalized space, then invert back to original scale
    pred_test_scaled = model.predict(X_test)
    pred_test = y_scaler.inverse_transform(
        pred_test_scaled.reshape(-1, 1)
    ).ravel()

    y_test_true = y_test_raw.copy()

else:
    # No normalization: use raw values directly
    X_train = X_train_raw
    X_test = X_test_raw
    y_train = y_train_raw

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    y_test_true = y_test_raw.copy()

# ============================================================
# Show metrics with professional cards
# ============================================================
st.markdown("## 📊 Performance do Modelo")

with st.container(border=True):
    st.markdown("### 🎯 Métricas de Erro (Conjunto de Teste)")

    # Calculate all metrics
    mae = mean_absolute_error(y_test_true, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test_true, pred_test))
    r2 = r2_score(y_test_true, pred_test)

    # MAPE calculation (avoiding division by zero)
    mape = np.mean(np.abs((y_test_true - pred_test) / y_test_true)) * 100 if (y_test_true != 0).all() else None

    # Mean of actual values for context
    y_mean = y_test_true.mean()

    # Display metrics in columns
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        mae_pct = (mae / y_mean * 100) if y_mean != 0 else 0
        mae_color = "🟢" if mae_pct < 3 else "🟡" if mae_pct < 7 else "🔴"
        st.metric(
            "MAE (Erro Médio Absoluto)",
            f"{mae:.2f}",
            f"{mae_color} {mae_pct:.1f}% da média",
            help="Erro médio em unidades do preço. Quanto menor, melhor."
        )

    with col_m2:
        rmse_pct = (rmse / y_mean * 100) if y_mean != 0 else 0
        rmse_color = "🟢" if rmse_pct < 5 else "🟡" if rmse_pct < 10 else "🔴"
        st.metric(
            "RMSE",
            f"{rmse:.2f}",
            f"{rmse_color} {rmse_pct:.1f}% da média",
            help="Raiz do erro quadrático médio. Penaliza erros grandes."
        )

    with col_m3:
        r2_color = "🟢" if r2 > 0.7 else "🟡" if r2 > 0.3 else "🔴"
        r2_pct = r2 * 100 if r2 > 0 else 0
        st.metric(
            "R² (Coef. Determinação)",
            f"{r2:.3f}",
            f"{r2_color} {r2_pct:.1f}% explicado",
            help="% da variância explicada. 1.0 = perfeito, negativo = pior que média"
        )

    with col_m4:
        if mape is not None:
            mape_color = "🟢" if mape < 5 else "🟡" if mape < 10 else "🔴"
            st.metric(
                "MAPE (Erro %)",
                f"{mape:.1f}%",
                f"{mape_color}",
                help="Erro percentual médio absoluto. Mais interpretável."
            )
        else:
            st.metric(
                "MAPE",
                "N/A",
                help="Não calculável (divisão por zero)"
            )

    # Interpretation guide
    with st.expander("📖 Como interpretar as métricas"):
        st.markdown(f"""
        ### Contexto
        - **Média do target**: {y_mean:.2f}
        - **Desvio padrão**: {y_test_true.std():.2f}
        - **Samples no teste**: {len(y_test_true)}

        ### Interpretação por Métrica

        **MAE (Mean Absolute Error)**
        - Erro médio: {mae:.2f} unidades ({mae_pct:.1f}% da média)
        - {"✅ Excelente!" if mae_pct < 3 else "⚠️ Moderado" if mae_pct < 7 else "❌ Alto"}
        - Significa que, em média, o modelo erra {mae:.2f} unidades

        **RMSE (Root Mean Squared Error)**
        - RMSE: {rmse:.2f} ({rmse_pct:.1f}% da média)
        - Penaliza outliers e erros grandes
        - {"✅ Bom desempenho" if rmse_pct < 5 else "⚠️ Aceitável" if rmse_pct < 10 else "❌ Revisar modelo"}

        **R² (Coefficient of Determination)**
        - R²: {r2:.3f} ({r2_pct:.1f}% da variância explicada)
        - {"✅ Modelo forte" if r2 > 0.7 else "⚠️ Modelo moderado" if r2 > 0.3 else "❌ Modelo fraco" if r2 > 0 else "❌ Pior que baseline (média)"}
        - {f"Explica {r2_pct:.0f}% da variabilidade dos dados" if r2 > 0 else "Não consegue melhorar a baseline"}

        **MAPE (Mean Absolute Percentage Error)**
        - {f"MAPE: {mape:.1f}%" if mape is not None else "N/A"}
        - {f"✅ Excelente precisão" if mape and mape < 5 else f"⚠️ Precisão moderada" if mape and mape < 10 else f"❌ Baixa precisão" if mape else "N/A"}
        - {"Erro típico em % do valor real" if mape else ""}
        """)

st.divider()


# ============================================================
# Feature Importance (visual + table)
# ============================================================
st.markdown("### 🧠 Importância das Features")

importance_values = None

# Modelos tipo árvore (Random Forest, XGBoost)
if hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_
    importance_type = "Importância (Gini/Gain)"

# Modelos lineares (Ridge) – usamos o valor absoluto dos coeficientes
elif hasattr(model, "coef_"):
    coef = model.coef_
    importance_values = np.abs(np.ravel(coef))
    importance_type = "Importância (|Coeficiente|)"

if importance_values is None:
    st.info("ℹ️ O modelo selecionado não expõe importância de features de forma direta.")
else:
    fi_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_values,
        }
    ).sort_values("Importance", ascending=False)

    # Normalize importance to percentage
    fi_df["Importance_Pct"] = (fi_df["Importance"] / fi_df["Importance"].sum()) * 100

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        # Bar chart of feature importance
        fig_importance = go.Figure()

        fig_importance.add_trace(
            go.Bar(
                x=fi_df["Importance_Pct"],
                y=fi_df["Feature"],
                orientation='h',
                marker=dict(
                    color=fi_df["Importance_Pct"],
                    colorscale='Blues',
                    showscale=False
                ),
                text=fi_df["Importance_Pct"].round(1).astype(str) + '%',
                textposition='outside',
            )
        )

        fig_importance.update_layout(
            title=f"{importance_type}",
            xaxis_title="Importância Relativa (%)",
            yaxis_title="Features",
            height=max(300, len(fi_df) * 30),
            showlegend=False,
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig_importance, use_container_width=True)

    with col_table:
        st.markdown("**Top Features**")
        st.dataframe(
            fi_df[["Feature", "Importance_Pct"]].head(10).round(2),
            use_container_width=True,
            height=350,
            hide_index=True
        )

    st.caption(
        f"💡 **{importance_type}**: Mostra quais features têm maior impacto nas previsões do modelo. "
        f"Features com maior importância são mais relevantes para prever o target."
    )

st.divider()



# ============================================================
# Plot historical performance (real vs predicted)
# ============================================================
st.markdown("### 📈 Desempenho Histórico (Conjunto de Teste)")

fig_hist = go.Figure()

fig_hist.add_trace(
    go.Scatter(
        x=dates_test,
        y=y_test_true,
        mode="lines",
        name="Valor Real",
        line=dict(color="blue", width=2)
    )
)

fig_hist.add_trace(
    go.Scatter(
        x=dates_test,
        y=pred_test,
        mode="lines",
        name="Previsão do Modelo",
        line=dict(color="red", width=2, dash="dot")
    )
)

# Add error bands
residuals = y_test_true - pred_test
fig_hist.add_trace(
    go.Scatter(
        x=dates_test,
        y=residuals,
        mode="lines",
        name="Erro (Residual)",
        line=dict(color="gray", width=1),
        yaxis="y2",
        opacity=0.5
    )
)

fig_hist.update_layout(
    title="Comparação: Valores Reais vs Previsões do Modelo",
    xaxis_title="Data",
    yaxis_title=f"{target_col} (Preço)",
    yaxis2=dict(
        title="Erro (Residual)",
        overlaying="y",
        side="right",
        showgrid=False
    ),
    hovermode='x unified',
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

st.plotly_chart(fig_hist, use_container_width=True)

# Residual statistics
col_res1, col_res2, col_res3 = st.columns(3)
with col_res1:
    st.metric("Erro Médio", f"{residuals.mean():.2f}", help="Viés do modelo (deveria ser ~0)")
with col_res2:
    st.metric("Erro Std Dev", f"{residuals.std():.2f}", help="Variabilidade dos erros")
with col_res3:
    max_error = residuals.abs().max()
    st.metric("Maior Erro", f"{max_error:.2f}", help="Maior erro absoluto observado")

st.divider()

# ============================================================
# Multi-step OUT-OF-SAMPLE forecast
# ============================================================
st.markdown(f"### 🔮 Previsão Futura ({horizon} dias)")

# Usamos a última linha disponível (com lags já preenchidos) como ponto de partida
last_row = df_ml.iloc[-1:].copy()
last_date = df_ml["date"].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=horizon,
)

forecast_values = []

# Linha de features (sem date / target) em escala original
current_row = last_row.drop(columns=["date", target_col]).copy()

for _ in range(horizon):
    # 1) Prepara features para previsão (aplica scaler se necessário)
    if normalize and x_scaler is not None and y_scaler is not None:
        current_x = x_scaler.transform(current_row)
        next_pred_scaled = model.predict(current_x)[0]
        # Volta para a escala original do target
        next_pred = y_scaler.inverse_transform(
            np.array([[next_pred_scaled]])
        )[0, 0]
    else:
        next_pred = model.predict(current_row)[0]

    forecast_values.append(next_pred)

    # 2) Atualiza apenas os lags do target na linha atual (em escala ORIGINAL)
    if num_lags > 0:
        for i in range(num_lags, 1, -1):
            current_row[f"{target_col}_lag{i}"] = current_row[
                f"{target_col}_lag{i-1}"
            ]
        current_row[f"{target_col}_lag1"] = next_pred
    # As outras features (externas) permanecem constantes com o último valor conhecido.

# Calculate simple confidence interval (using historical std of residuals)
forecast_std = residuals.std()
upper_bound = [v + 1.96 * forecast_std * (1 + i*0.02) for i, v in enumerate(forecast_values)]
lower_bound = [v - 1.96 * forecast_std * (1 + i*0.02) for i, v in enumerate(forecast_values)]

# Combined chart: Historical (last 60 days) + Forecast
last_60_days = min(60, len(df_ml))
historical_dates = df_ml["date"].iloc[-last_60_days:]
historical_values = df_ml[target_col].iloc[-last_60_days:]

fig_combined = go.Figure()

# Historical actual values
fig_combined.add_trace(
    go.Scatter(
        x=historical_dates,
        y=historical_values,
        mode="lines",
        name="Histórico Real",
        line=dict(color="blue", width=2)
    )
)

# Forecast
fig_combined.add_trace(
    go.Scatter(
        x=future_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Previsão",
        line=dict(color="red", width=2, dash="dash"),
        marker=dict(size=5)
    )
)

# Confidence interval
fig_combined.add_trace(
    go.Scatter(
        x=future_dates,
        y=upper_bound,
        mode="lines",
        name="IC Superior (95%)",
        line=dict(color="rgba(255,0,0,0)", width=0),
        showlegend=False
    )
)

fig_combined.add_trace(
    go.Scatter(
        x=future_dates,
        y=lower_bound,
        mode="lines",
        name="IC Inferior (95%)",
        line=dict(color="rgba(255,0,0,0)", width=0),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        showlegend=True
    )
)

# Add vertical line separating history from forecast
fig_combined.add_vline(
    x=last_date,
    line_dash="dot",
    line_color="gray",
    annotation_text="Início da Previsão",
    annotation_position="top"
)

fig_combined.update_layout(
    title=f"Histórico (últimos {last_60_days} dias) + Previsão ({horizon} dias)",
    xaxis_title="Data",
    yaxis_title=target_col,
    hovermode='x unified',
    height=500,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

st.plotly_chart(fig_combined, use_container_width=True)

# Forecast summary
col_f1, col_f2, col_f3, col_f4 = st.columns(4)
with col_f1:
    st.metric("Último Valor Real", f"{historical_values.iloc[-1]:.2f}", help=f"Último valor conhecido ({last_date.date()})")
with col_f2:
    first_forecast = forecast_values[0]
    change_1d = ((first_forecast - historical_values.iloc[-1]) / historical_values.iloc[-1] * 100)
    st.metric("Previsão +1 dia", f"{first_forecast:.2f}", f"{change_1d:+.1f}%")
with col_f3:
    last_forecast = forecast_values[-1]
    change_horizon = ((last_forecast - historical_values.iloc[-1]) / historical_values.iloc[-1] * 100)
    st.metric(f"Previsão +{horizon} dias", f"{last_forecast:.2f}", f"{change_horizon:+.1f}%")
with col_f4:
    avg_forecast = np.mean(forecast_values)
    st.metric("Média Prevista", f"{avg_forecast:.2f}", help=f"Média das previsões para os próximos {horizon} dias")

st.divider()

# ============================================================
# Export functionality
# ============================================================
st.markdown("### 📥 Exportar Resultados")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    # Export forecast to CSV
    forecast_df = pd.DataFrame({
        'Data': future_dates.date,
        'Previsão': forecast_values,
        'IC_Superior_95': upper_bound,
        'IC_Inferior_95': lower_bound
    })

    csv_forecast = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Baixar Previsões (CSV)",
        data=csv_forecast,
        file_name=f"previsao_{target_col}_{horizon}dias_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_forecast_csv",
    )

with col_exp2:
    # Export model metrics to CSV
    metrics_df = pd.DataFrame({
        'Métrica': ['MAE', 'RMSE', 'R²', 'MAPE'],
        'Valor': [mae, rmse, r2, mape if mape else 'N/A'],
        'Contexto': [
            f"{mae_pct:.1f}% da média",
            f"{rmse_pct:.1f}% da média",
            f"{r2_pct:.1f}% explicado" if r2 > 0 else "Negativo",
            f"{mape:.1f}%" if mape else "N/A"
        ]
    })

    csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Baixar Métricas (CSV)",
        data=csv_metrics,
        file_name=f"metricas_modelo_{target_col}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_metrics_csv",
    )
