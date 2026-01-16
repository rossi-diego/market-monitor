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
from scipy import stats

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

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
        # Calculate correlation with target to suggest best features
        default_features = []
        if target_col:
            correlations = {}
            for col in valid_cols:
                if col != target_col:
                    try:
                        # Calculate correlation
                        corr_data = BASE[[target_col, col]].dropna()
                        if len(corr_data) > 10:
                            corr = corr_data.corr().iloc[0, 1]
                            correlations[col] = abs(corr)
                    except:
                        pass

            # Get top 5 features by correlation
            if correlations:
                sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                default_features = [feat[0] for feat in sorted_features[:5]]

        # If no correlations found, use first 3
        if not default_features:
            default_features = [c for c in valid_cols if c != target_col][:3]

        feature_cols = st.multiselect(
            "Features (variáveis explicativas)",
            options=valid_cols,
            default=default_features,
            help="💡 As 5 features mais correlacionadas com o target são selecionadas por padrão"
        )

        # Show correlation info
        if feature_cols and target_col and correlations:
            with st.expander(f"📊 Correlação das Features com {target_col}", expanded=False):
                # Show correlations for selected features
                selected_corrs = [(f, correlations.get(f, 0)) for f in feature_cols if f in correlations]
                selected_corrs.sort(key=lambda x: x[1], reverse=True)

                st.markdown("**Top 5 Features Mais Correlacionadas:**")
                for i, (feat, corr) in enumerate(selected_corrs[:5], 1):
                    corr_strength = "Forte" if corr > 0.7 else "Moderada" if corr > 0.4 else "Fraca"
                    corr_color = "🟢" if corr > 0.7 else "🟡" if corr > 0.4 else "🔴"
                    st.caption(f"{i}. **{feat}**: {corr:.3f} {corr_color} ({corr_strength})")

    # At least 1 feature or lag must exist
    if len(feature_cols) == 0:
        st.info("💡 Dica: Selecione features ou configure lags abaixo para treinar o modelo.")

# Container 2: Model Configuration
with st.container(border=True):
    st.markdown("### 🤖 Configuração do Algoritmo")

    col_model, col_lags, col_horizon = st.columns(3)

    with col_model:
        models_dict = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.01),
            "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5),
            "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            "SVR (Support Vector)": SVR(kernel='rbf', C=100, epsilon=0.1),
            "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, learning_rate_init=0.001, random_state=42),
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
        if HAS_LGBM:
            models_dict["LightGBM"] = LGBMRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )

        model_label = st.selectbox(
            "Algoritmo",
            list(models_dict.keys()),
            help="Ridge: linear rápido | Lasso: com regularização L1 | RF: não-linear robusto | XGB/LGBM: máxima performance | SVR: kernel baseado | MLP: redes neurais"
        )
        model = models_dict[model_label]

        # Model description
        model_descriptions = {
            "Ridge Regression": "✅ Rápido e interpretável\n✅ Bom para relações lineares\n✅ Robusto com multicolinearidade\n⚠️ Pode não capturar não-linearidades",
            "Lasso Regression": "✅ Seleção automática de features\n✅ Simples e interpretável\n⚠️ Menos eficaz com muitas features\n⚠️ Sensível a escala",
            "Elastic Net": "✅ Combina Ridge + Lasso\n✅ Bom com muitas features correlacionadas\n✅ Seleção de features\n⚠️ Menos preciso que ensemble methods",
            "Random Forest": "✅ Robusto a outliers\n✅ Captura não-linearidades\n✅ Menos propenso a overfitting\n✅ Paralelizável",
            "Gradient Boosting": "✅ Melhor que RF em muitos casos\n✅ Captura padrões complexos\n⚠️ Risco de overfitting\n⚠️ Mais lento para treinar",
            "SVR (Support Vector)": "✅ Bom com high-dimensional data\n✅ Kernel flex para não-linearidades\n⚠️ Requer normalização\n⚠️ Lento com muitos dados",
            "Neural Network (MLP)": "✅ Máxima flexibilidade\n✅ Padrões muito complexos\n⚠️ Caixa preta (difícil interpretar)\n⚠️ Requer mais dados",
            "XGBoost": "✅ Melhor performance geral\n✅ Captura padrões muito complexos\n✅ Feature importance confiável\n⚠️ Requer mais dados para treinar",
            "LightGBM": "✅ Mais rápido que XGBoost\n✅ Menor consumo de memória\n✅ Excelente performance\n⚠️ Pode overfittar com poucos dados",
        }
        st.caption(model_descriptions.get(model_label, ""))

    with col_lags:
        num_lags = st.slider(
            "Número de lags",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
            help="Valores históricos do target: lag1=ontem, lag2=anteontem, etc."
        )
        if num_lags > 0:
            st.caption(f"✓ Usando {num_lags} valores históricos")
        else:
            st.caption("✓ Apenas features externas")

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

# Ensure date column is datetime type
df_ml["date"] = pd.to_datetime(df_ml["date"], errors="coerce")

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

    # MAPE calculation (handling division by zero properly)
    def safe_mape(y_true, y_pred):
        """Calculate MAPE with proper handling of edge cases."""
        if len(y_true) == 0:
            return None
        
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() == 0:
            return None
        
        mape_val = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape_val

    mape = safe_mape(y_test_true.values, pred_test)

    # Additional metrics: MASE (Mean Absolute Scaled Error)
    naive_forecast = y_test_true.iloc[:-1].values
    naive_error = np.mean(np.abs(np.diff(y_test_true.values)))
    mase = mae / (naive_error + 1e-8) if naive_error > 0 else np.inf

    # Mean of actual values for context
    y_mean = y_test_true.mean()

    # Display metrics in columns
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        mae_pct = (mae / y_mean * 100) if y_mean != 0 else 0
        mae_color = "🟢" if mae_pct < 3 else "🟡" if mae_pct < 7 else "🔴"
        st.metric(
            "MAE",
            f"{mae:.2f}",
            f"{mae_color} {mae_pct:.1f}%",
            help="Erro médio absoluto em unidades do preço"
        )

    with col_m2:
        rmse_pct = (rmse / y_mean * 100) if y_mean != 0 else 0
        rmse_color = "🟢" if rmse_pct < 5 else "🟡" if rmse_pct < 10 else "🔴"
        st.metric(
            "RMSE",
            f"{rmse:.2f}",
            f"{rmse_color} {rmse_pct:.1f}%",
            help="Raiz do erro quadrático médio (penaliza grandes erros)"
        )

    with col_m3:
        r2_color = "🟢" if r2 > 0.7 else "🟡" if r2 > 0.3 else "🔴"
        r2_pct = r2 * 100 if r2 > 0 else 0
        st.metric(
            "R²",
            f"{r2:.3f}",
            f"{r2_color} {r2_pct:.1f}%",
            help="Coeficiente de determinação (% explicada)"
        )

    with col_m4:
        if mape is not None and not np.isinf(mape):
            mape_color = "🟢" if mape < 5 else "🟡" if mape < 10 else "🔴"
            st.metric(
                "MAPE",
                f"{mape:.1f}%",
                f"{mape_color}",
                help="Erro percentual médio absoluto"
            )
        else:
            st.metric("MAPE", "N/A", help="Não calculável")

    with col_m5:
        if not np.isinf(mase):
            mase_color = "🟢" if mase < 1.0 else "🟡" if mase < 1.5 else "🔴"
            st.metric(
                "MASE",
                f"{mase:.2f}",
                f"{mase_color}",
                help="Erro escalado pelo naive forecast (< 1.0 = melhor que naive)"
            )
        else:
            st.metric("MASE", "N/A", help="Erro escalado")

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
        - Mais interpretável que RMSE por usar escala original

        **RMSE (Root Mean Squared Error)**
        - RMSE: {rmse:.2f} ({rmse_pct:.1f}% da média)
        - Penaliza outliers e erros grandes mais fortemente
        - {"✅ Bom desempenho" if rmse_pct < 5 else "⚠️ Aceitável" if rmse_pct < 10 else "❌ Revisar modelo"}

        **R² (Coefficient of Determination)**
        - R²: {r2:.3f} ({r2_pct:.1f}% da variância explicada)
        - {"✅ Modelo forte" if r2 > 0.7 else "⚠️ Modelo moderado" if r2 > 0.3 else "❌ Modelo fraco" if r2 > 0 else "❌ Pior que baseline (média)"}
        - {f"Explica {r2_pct:.0f}% da variabilidade dos dados" if r2 > 0 else "Não consegue melhorar a baseline"}

        **MAPE (Mean Absolute Percentage Error)**
        - {f"MAPE: {mape:.1f}%" if mape and not np.isinf(mape) else "N/A"}
        - {f"✅ Excelente precisão" if mape and mape < 5 else f"⚠️ Precisão moderada" if mape and mape < 10 else f"❌ Baixa precisão" if mape else "N/A"}
        - Erro em % do valor real (mais comparável entre diferentes escalas)

        **MASE (Mean Absolute Scaled Error)**
        - {f"MASE: {mase:.2f}" if not np.isinf(mase) else "N/A"}
        - {"✅ Melhor que naive forecast" if mase < 1.0 else "⚠️ Similar ao naive" if mase < 1.5 else "❌ Pior que naive"}
        - Compara performance com um modelo simples (naive forecast)
        - MASE < 1.0 significa que o modelo é melhor que apenas repetir o último valor

        ### 🧠 Qual métrica usar?
        - **Para interpretação**: Use **MAE** ou **MAPE** (mesma unidade/proporção do preço)
        - **Para otimização**: Minimize **RMSE** (mais sensível a outliers)
        - **Para comparação**: Use **R²** ou **MASE** (independente da escala)
        - **Combinado**: Bom modelo tem MAE baixo + R² alto + MASE < 1.0
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
# Residual Analysis (Simplified)
# ============================================================
with st.expander("🔍 Análise de Resíduos (Diagnósticos do Modelo)", expanded=False):
    st.markdown("""
    ### 📊 O que são Resíduos?

    **Resíduos = Valor Real - Previsão**
    - **Positivo**: Modelo subestimou (previu menos)
    - **Negativo**: Modelo superestimou (previu mais)
    - **Próximo de 0**: Previsão precisa
    """)

    # Calculate residuals from test set
    residuals = y_test_true.values - pred_test

    # Simple residual statistics
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)

    with col_r1:
        res_mean = residuals.mean()
        mean_color = "🟢" if abs(res_mean) < 1 else "🟡" if abs(res_mean) < 3 else "🔴"
        st.metric("Viés Médio", f"{res_mean:.2f}", f"{mean_color}",
                 help="Deveria ser ~0. Se positivo, modelo subestima. Se negativo, superestima.")

    with col_r2:
        res_std = residuals.std()
        st.metric("Desvio Padrão", f"{res_std:.2f}",
                 help="Variabilidade dos erros. Menor é melhor.")

    with col_r3:
        max_error = np.abs(residuals).max()
        st.metric("Maior Erro", f"{max_error:.2f}",
                 help="Pior previsão observada")

    with col_r4:
        # Percentage of errors within 1 std
        within_1std = (np.abs(residuals) <= res_std).sum() / len(residuals) * 100
        st.metric("Dentro de 1σ", f"{within_1std:.0f}%",
                 help="% de erros dentro de 1 desvio padrão (~68% é normal)")

    # Simple residual plot
    fig_residuals = go.Figure()

    fig_residuals.add_trace(
        go.Scatter(
            x=dates_test,
            y=residuals,
            mode="markers",
            name="Erros",
            marker=dict(
                size=6,
                color=residuals,
                colorscale='RdYlGn_r',  # Red for positive errors, green for negative
                showscale=True,
                colorbar=dict(title="Erro"),
                line=dict(width=0.5, color='white')
            ),
            hovertemplate="<b>Data:</b> %{x}<br><b>Erro:</b> %{y:.2f}<extra></extra>"
        )
    )

    # Add zero line
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="gray", line_width=2)

    fig_residuals.update_layout(
        title="Erros do Modelo ao Longo do Tempo",
        xaxis_title="Data",
        yaxis_title="Erro (Real - Previsto)",
        height=400,
        hovermode='x unified',
    )

    st.plotly_chart(fig_residuals, use_container_width=True)

    # Interpretation
    st.markdown("### ✅ Como Interpretar")

    if abs(res_mean) < 1 and within_1std > 60:
        st.success("✅ **Modelo equilibrado**: Erros pequenos e bem distribuídos!")
    elif abs(res_mean) > 3:
        st.warning(f"⚠️ **Viés detectado**: Modelo tende a {'subestimar' if res_mean > 0 else 'superestimar'}. Considere ajustar features.")
    elif within_1std < 50:
        st.warning("⚠️ **Erros dispersos**: Muitos erros grandes. Considere adicionar mais features ou lags.")
    else:
        st.info("ℹ️ **Modelo aceitável**: Alguns erros, mas razoável para previsões.")

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

try:
    # Usamos a última linha disponível (com lags já preenchidos) como ponto de partida
    last_row = df_ml.iloc[-1:].copy()

    # Ensure date is Timestamp type and handle properly
    last_date = pd.to_datetime(df_ml["date"].iloc[-1])

    # Create future dates starting from the day after last date
    # Use timedelta properly to avoid the error
    future_dates = pd.DatetimeIndex([
        last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)
    ])

    forecast_values = []

    # Linha de features (sem date / target) em escala original
    # Convert DataFrame row to Series for easier manipulation
    current_row = last_row.drop(columns=["date", target_col]).copy().iloc[0]

    for step in range(horizon):
        # 1) Prepara features para previsão (aplica scaler se necessário)
        # Reshape para 2D para o modelo
        current_row_2d = current_row.values.reshape(1, -1)
        
        if normalize and x_scaler is not None and y_scaler is not None:
            current_x = x_scaler.transform(current_row_2d)
            next_pred_scaled = model.predict(current_x)[0]
            # Volta para a escala original do target
            next_pred = float(y_scaler.inverse_transform(
                np.array([[next_pred_scaled]])
            )[0, 0])
        else:
            next_pred = float(model.predict(current_row_2d)[0])

        forecast_values.append(next_pred)

        # 2) Atualiza apenas os lags do target na linha atual (em escala ORIGINAL)
        if num_lags > 0:
            for i in range(num_lags, 1, -1):
                lag_col = f"{target_col}_lag{i}"
                prev_lag_col = f"{target_col}_lag{i-1}"
                if lag_col in current_row.index and prev_lag_col in current_row.index:
                    current_row[lag_col] = current_row[prev_lag_col]
            current_row[f"{target_col}_lag1"] = next_pred
        # As outras features (externas) permanecem constantes com o último valor conhecido.

    forecast_values = np.array(forecast_values, dtype=float)

    # Calculate adaptive confidence interval (increases with forecast horizon)
    forecast_std = float(residuals.std())
    forecast_steps = np.arange(1, horizon + 1, dtype=float)
    
    # Uncertainty grows with horizon: std * (1 + 0.05 * step)
    uncertainty_multiplier = 1.0 + (0.05 * forecast_steps)
    upper_bound = forecast_values + (1.96 * forecast_std * uncertainty_multiplier)
    lower_bound = forecast_values - (1.96 * forecast_std * uncertainty_multiplier)

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

    # Forecast uncertainty info
    with st.expander("⚠️ Entender a Incerteza das Previsões"):
        st.markdown(f"""
        ### 📊 Intervalo de Confiança (95%)
        
        A área cinzenta ao redor da previsão representa a **incerteza** do modelo:
        
        - **Intervalo estreito**: Modelo confiante na previsão
        - **Intervalo largo**: Maior incerteza (observe com cautela)
        - **Intervalo cresce com o tempo**: Normal! Previsões distantes são menos certeiras
        
        ### 🔢 Como é calculado?
        
        - Baseado na **variabilidade dos erros históricos** (resíduos)
        - Aumenta proporcionalmente com a distância da previsão
        - Pressupõe que padrões futuros serão similares aos passados
        
        ### ⚠️ Limitações Importantes:
        
        - **Não prevê events**: Choques de mercado, notícias, etc não são previstos
        - **Pressupõe continuidade**: Assume que padrões históricos persistem
        - **Pior longe no futuro**: A incerteza cresce com o horizonte
        - **Sensível ao período de treinamento**: Diferentes períodos = diferentes previsões
        
        **Para {horizon} dias:** Intervalo estimado = {forecast_std:.2f} ± {1.96 * forecast_std:.2f} (base) até {1.96 * forecast_std * (1 + 0.05*horizon):.2f} (end)
        """)

    st.divider()

except Exception as e:
    st.error(f"❌ Erro ao gerar previsão: {str(e)}")
    st.info("💡 Dica: Verifique se há dados suficientes e se as configurações estão adequadas.")

# ============================================================
# Export functionality
# ============================================================
st.markdown("### 📥 Exportar Resultados")

col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    try:
        # Export forecast to CSV
        forecast_df = pd.DataFrame({
            'Data': [d.strftime('%Y-%m-%d') for d in future_dates],
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
    except Exception as e:
        st.warning(f"⚠️ Erro ao exportar previsões: {str(e)}")

with col_exp2:
    # Export model metrics to CSV
    mape_val = f"{mape:.1f}%" if mape and not np.isinf(mape) else "N/A"
    mase_val = f"{mase:.2f}" if not np.isinf(mase) else "N/A"
    
    metrics_df = pd.DataFrame({
        'Métrica': ['MAE', 'RMSE', 'R²', 'MAPE', 'MASE'],
        'Valor': [mae, rmse, r2, mape_val, mase_val],
        'Contexto': [
            f"{mae_pct:.1f}% da média",
            f"{rmse_pct:.1f}% da média",
            f"{r2_pct:.1f}% explicado" if r2 > 0 else "Negativo",
            mape_val,
            mase_val
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
