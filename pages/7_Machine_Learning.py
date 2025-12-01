"""
Machine Learning Forecast Page
------------------------------

This page allows the user to:

1. Select an asset (target variable).
2. Choose which dataset columns will be used as FEATURES.
3. Choose number of LAGS (0 to 10).
4. Select ML model (Ridge, Random Forest, XGBoost).
5. Choose the DATE RANGE used to train/test the model.
6. View model performance vs actual (historical backtest).
7. Inspect FEATURE IMPORTANCE (coeffs or importances).
8. Produce OUT-OF-SAMPLE FORECAST for up to 45 future days.

The page also includes explanations for:
- What are lags?
- How the model learns temporal structure.
- What is multi-step forecasting?
- What MAE, RMSE and R² represent.
"""

# ============================================================
# Imports & Setup
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.data_pipeline import df
from src.utils import apply_theme, date_range_picker, section

# Theme
apply_theme()

# ============================================================
# Base Data
# ============================================================
BASE = df.copy()
BASE["date"] = pd.to_datetime(BASE["date"], errors="coerce")
BASE = BASE.sort_values("date")

# ============================================================
# Header & Explanation
# ============================================================
section(
    "🔮 Machine Learning Forecast",
    "Selecione um ativo, colunas de entrada e gere previsões automáticas.",
    "🤖",
)

st.markdown(
    """
### 📘 Como funciona o modelo?

**1) Lags:**  
Lags são valores históricos da própria série da variável target.  
Ex.: se o target for o óleo na bolsa:  
- *Lag 1 = preço(t-1)*  
- *Lag 2 = preço(t-2)*  

Eles ajudam o modelo a capturar **tendência, momentum e autocorrelação**.

**2) Features externas:**  
São outras colunas do dataset (ex.: dólar, farelo, prêmio, palma...) que ajudam o modelo
a explicar o comportamento do ativo target.

**3) Multi-step forecasting:**  
A previsão de até 45 dias é feita **iterativamente**:  
o modelo prevê o dia seguinte, depois usa essa previsão como entrada para prever o próximo,
e assim por diante.

**4) Métricas:**
- **MAE** — erro médio absoluto (em unidades do preço).  
- **RMSE** — raiz do erro quadrático médio (penaliza mais erros grandes).  
- **R²** — fração da variância explicada pelo modelo (quanto mais próximo de 1, melhor).  
  Pode ser **negativo** quando o modelo está pior do que simplesmente prever a média histórica.

---
"""
)

# ============================================================
# 1. Target asset selection
# ============================================================
section("🎯 Selecione o ativo para prever", None, "📈")

valid_cols = [
    c for c in BASE.columns
    if c != "date" and BASE[c].dtype != "object"
]

target_col = st.selectbox("Escolha o ativo (Target)", valid_cols, index=0)

st.divider()

# ============================================================
# 2. Date range for model training/testing
# ============================================================
section("📆 Período de dados para treinar/testar o modelo", None, "📆")

train_start, train_end = date_range_picker(
    BASE["date"],
    state_key="ml_train_range",
    default_days=730,  # exemplo: ~2 anos por padrão
)

st.caption(
    f"Período selecionado para o modelo: **{train_start}** até **{train_end}**."
)
st.divider()

# ============================================================
# 3. Feature selection
# ============================================================
section("🧩 Selecione as features (variáveis explicativas)", None, "🧩")

feature_candidates = [c for c in valid_cols if c != target_col]

feature_cols = st.multiselect(
    "Selecione colunas para o modelo aprender",
    options=feature_candidates,
    default=feature_candidates[:3],
)

# At least 1 feature or lag must exist
if len(feature_cols) == 0:
    st.warning("Selecione ao menos uma feature.")
    st.stop()

st.divider()

# ============================================================
# 4. Lag selection
# ============================================================
section("⏳ Lags da série", "Máximo 10 lags", "⏳")

num_lags = st.slider("Número de lags", min_value=0, max_value=10, value=5, step=1)

st.markdown(
    """
**O que é lag?**

- **Lag 1** = valor do target ontem  
- **Lag 2** = valor do target anteontem  

Mais lags → modelo enxerga mais histórico, mas:
- aumenta a dimensão do problema,
- aumenta o risco de **overfitting**.
"""
)

st.divider()

# ============================================================
# 5. ML Model selection
# ============================================================
section("🤖 Modelo de Machine Learning", None, "🤖")

models_dict = {
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    ),
}
if HAS_XGB:
    models_dict["XGBoost"] = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

model_label = st.selectbox("Selecione o modelo", list(models_dict.keys()))
model = models_dict[model_label]

st.divider()

# ============================================================
# 6. Prediction horizon
# ============================================================
section("📅 Horizonte de previsão", None, "📅")

horizon = st.slider(
    "Dias à frente para prever",
    min_value=1,
    max_value=45,
    value=30,
)

st.caption(
    "A previsão é feita de forma **multi-step**: "
    "cada dia previsto alimenta o modelo para prever o próximo dia."
)
st.divider()

# ============================================================
# Prepare dataset (filter period + lags + features)
# ============================================================
# Filtra pelo período escolhido
mask_period = (BASE["date"].dt.date >= train_start) & (BASE["date"].dt.date <= train_end)
df_ml = BASE.loc[mask_period, ["date", target_col] + feature_cols].copy()

if df_ml.empty:
    st.error("Não há dados no período selecionado para treinar o modelo.")
    st.stop()

# Gera colunas de lag do target
for lag in range(1, num_lags + 1):
    df_ml[f"{target_col}_lag{lag}"] = df_ml[target_col].shift(lag)

# Remove linhas com NaN (causados pelos lags)
df_ml = df_ml.dropna().reset_index(drop=True)

if len(df_ml) < 50:
    st.warning(
        f"Poucos dados ({len(df_ml)} linhas) após aplicar período e lags. "
        "Considere aumentar o período ou reduzir o número de lags."
    )

# Monta X, y
X = df_ml.drop(columns=["date", target_col])
y = df_ml[target_col]

if len(df_ml) < 10:
    st.error("Dados insuficientes para treinar o modelo após filtragem e lags.")
    st.stop()

# Train-test split (80/20 temporal)
split = int(len(df_ml) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
dates_test = df_ml["date"].iloc[split:]

# Fit model
model.fit(X_train, y_train)

# Predict on historical test data
pred_test = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))
r2 = r2_score(y_test, pred_test)

# ============================================================
# 7. Show metrics
# ============================================================
section("📊 Métricas do modelo (backtest histórico)", None, "📊")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("MAE", f"{mae:.3f}")
with col_m2:
    st.metric("RMSE", f"{rmse:.3f}")
with col_m3:
    st.metric("R²", f"{r2:.3f}")

st.markdown(
    """
**Interpretação rápida:**

- **MAE** → erro médio absoluto (quanto menor, melhor).  
- **RMSE** → penaliza mais erros grandes; útil para ver se há outliers de erro.  
- **R² negativo** → o modelo está pior do que simplesmente prever a média histórica.
"""
)

st.divider()

# ============================================================
# 8. Feature importance / coefficients
# ============================================================
section("🧮 Importância das variáveis (Feature Importance)", None, "🧮")

feature_names = list(X.columns)
importances = None
importance_type = None

# Linear modelo (Ridge): coeficientes
if hasattr(model, "coef_"):
    coefs = np.array(model.coef_).ravel()
    importances = coefs
    importance_type = "coef"

# Tree-based (RandomForest, XGBoost): feature_importances_
elif hasattr(model, "feature_importances_"):
    importances = np.array(model.feature_importances_)
    importance_type = "imp"

if importances is not None:
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "abs_importance": np.abs(importances),
        }
    ).sort_values("abs_importance", ascending=False)

    st.markdown(
        """
Para modelos lineares (Ridge), os **coeficientes** indicam o efeito marginal:
- sinal (+/-) → direção do impacto
- magnitude → força do impacto absoluto

Para modelos baseados em árvores (Random Forest / XGBoost), a `feature_importances_` mostra
a importância relativa de cada variável na redução de erro do modelo.
"""
    )

    # Gráfico de barras (top 20)
    top_n = min(20, len(imp_df))
    top_imp = imp_df.head(top_n)

    fig_imp = go.Figure(
        data=go.Bar(
            x=top_imp["feature"],
            y=top_imp["importance"],
        )
    )
    fig_imp.update_layout(
        title="Importância das Features (top 20)",
        xaxis_title="Feature",
        yaxis_title="Coeficiente" if importance_type == "coef" else "Importância relativa",
        xaxis_tickangle=-45,
        margin=dict(b=120, t=60),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.dataframe(
        imp_df[["feature", "importance"]].reset_index(drop=True),
        use_container_width=True,
        height=400,
    )
else:
    st.info(
        "O modelo selecionado não expõe coeficientes nem `feature_importances_`. "
        "Por isso, a importância de features não está disponível."
    )

st.divider()

# ============================================================
# 9. Plot historical performance (real vs predicted)
# ============================================================
section("📈 Desempenho histórico (real vs previsto)", None, "📉")

fig_hist = go.Figure()
fig_hist.add_trace(
    go.Scatter(x=dates_test, y=y_test, mode="lines", name="Real")
)
fig_hist.add_trace(
    go.Scatter(x=dates_test, y=pred_test, mode="lines", name="Previsto")
)
fig_hist.update_layout(
    title="Histórico: Real vs Previsto (conjunto de teste)",
    xaxis_title="Data",
    yaxis_title=target_col,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ============================================================
# 10. Multi-step OUT-OF-SAMPLE forecast
# ============================================================
section("🔮 Forecast Futuro", f"Prevendo {horizon} dias à frente", "🔮")

last_row = df_ml.iloc[-1:].copy()
future_dates = pd.date_range(
    start=df_ml["date"].iloc[-1] + pd.Timedelta(days=1),
    periods=horizon,
    freq="D",
)

forecast_values = []

# current_row contém apenas as features + lags (sem date/target)
current_row = last_row.drop(columns=["date", target_col])

for _ in range(horizon):
    next_pred = model.predict(current_row)[0]
    forecast_values.append(next_pred)

    # Atualiza lags manualmente (apenas se houver lags)
    if num_lags > 0:
        for i in range(num_lags, 1, -1):
            current_row[f"{target_col}_lag{i}"] = current_row[f"{target_col}_lag{i-1}"]
        current_row[f"{target_col}_lag1"] = next_pred

# Plot forecast
fig_f = go.Figure()
fig_f.add_trace(
    go.Scatter(
        x=future_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast Futuro",
    )
)
fig_f.update_layout(
    title=f"Previsão para {target_col} – {horizon} dias à frente",
    xaxis_title="Data",
    yaxis_title=target_col,
)
st.plotly_chart(fig_f, use_container_width=True)
