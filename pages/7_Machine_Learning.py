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
    "🤖"
)

st.markdown("""
### 📘 Como funciona o modelo?

**1) Lags:**  
Lags são valores históricos da própria série da nossa variável target, por exemplo:  
- Lag 1 = preço de ontem  
- Lag 2 = preço de anteontem  

Eles ajudam o modelo a entender **tendência, momentum e autocorrelação**.

**2) Features externas:**  
Variáveis externas que podemos utilizar para auxiliar o aprendizado do modelo  
(ex.: dólar, farelo, prêmio, palma...).

**3) Multi-step forecasting:**  
A previsão de até 45 dias é feita **iterativamente**, um dia de cada vez:  
a previsão do dia seguinte vira entrada do dia posterior.

**4) Métricas:**
- **MAE** — erro médio absoluto (mesma unidade do preço).  
- **RMSE** — erro quadrático médio (penaliza mais erros grandes).  
- **R²** — variância explicada; pode ser **negativo** quando o modelo é pior
  do que simplesmente prever a **média histórica**.

---
""")

# ============================================================
# 1. Target asset selection
# ============================================================
section("🎯 Selecione o ativo para prever", None, "📈")

valid_cols = [
    c for c in BASE.columns
    if c not in ["date"] and BASE[c].dtype != "object"
]

target_col = st.selectbox("Escolha o ativo (Target)", valid_cols, index=0)

st.divider()

# ============================================================
# 2. Feature selection
# ============================================================
section("🧩 Selecione as features (variáveis explicativas)", None, "🧩")

feature_cols = st.multiselect(
    "Selecione colunas para o modelo aprender",
    options=valid_cols,
    default=[c for c in valid_cols if c != target_col][:3],
)

# At least 1 feature must exist
if len(feature_cols) == 0:
    st.warning("Selecione ao menos uma feature.")
    st.stop()

st.divider()

# ============================================================
# 3. Lag selection
# ============================================================
section("⏳ Lags da série", "Máximo 10 lags", "⏳")

num_lags = st.slider(
    "Número de lags",
    min_value=0,
    max_value=10,
    value=5,
    step=1,
)

st.markdown("""
Pequena explicação:
- **Lag 1** = preço de ontem  
- **Lag 2** = preço de anteontem  
- Mais lags → mais memória e complexidade → maior risco de overfitting  
""")

st.divider()

# ============================================================
# 4. ML Model selection
# ============================================================
section("🤖 Modelo de Machine Learning", None, "🤖")

models_dict = {
    "Ridge Regression": Ridge(),
    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
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
    )

model_label = st.selectbox("Selecione o modelo", list(models_dict.keys()))
model = models_dict[model_label]

st.divider()

# ============================================================
# 5. Prediction horizon
# ============================================================
section("📅 Horizonte de previsão", None, "📅")

horizon = st.slider(
    "Dias à frente para prever",
    min_value=1,
    max_value=45,
    value=30,
)

st.divider()

# ============================================================
# 6. Normalização (boa prática opcional)
# ============================================================
section("⚙️ Configuração avançada (normalização da target)", None, "🧪")

normalize_target = st.checkbox(
    "Normalizar a variável alvo durante o treino (z-score)",
    value=True,
    help=(
        "Subtrai a média e divide pelo desvio padrão no conjunto de treino. "
        "Ajuda alguns modelos a treinar de forma mais estável. "
        "As métricas e gráficos são SEMPRE mostrados em unidades originais."
    ),
)

st.caption(
    "A normalização é aplicada apenas internamente na etapa de treino. "
    "As previsões e métricas são sempre convertidas de volta para o nível de preço original."
)

st.divider()

# ============================================================
# Prepare dataset (lags + features)
# ============================================================
df_ml = BASE[["date", target_col] + feature_cols].copy()

# Generate lag columns
for lag in range(1, num_lags + 1):
    df_ml[f"{target_col}_lag{lag}"] = df_ml[target_col].shift(lag)

# Drop rows with NaN caused by lags
df_ml = df_ml.dropna().reset_index(drop=True)

if df_ml.empty:
    st.warning("Não há dados suficientes após a criação de lags. Tente reduzir o número de lags.")
    st.stop()

# Build X, y
X = df_ml.drop(columns=["date", target_col])
y = df_ml[target_col]

# Train-test split (80/20)
split = int(len(df_ml) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
dates_test = df_ml["date"].iloc[split:]

# ============================================================
# Fit model (with optional target normalization)
# ============================================================
if normalize_target:
    y_mean = y_train.mean()
    y_std = y_train.std()

    # Evita divisão por zero em caso de série quase constante
    if y_std == 0:
        y_std = 1.0

    y_train_scaled = (y_train - y_mean) / y_std

    # Treina no espaço normalizado
    model.fit(X_train, y_train_scaled)

    # Predição histórica (em espaço normalizado)
    pred_test_scaled = model.predict(X_test)

    # Converte de volta para unidade original
    pred_test = pred_test_scaled * y_std + y_mean
    y_test_true = y_test  # já em unidade original
else:
    # Sem normalização de target
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)
    y_test_true = y_test

# ============================================================
# Show metrics (sempre em unidade original)
# ============================================================
mae = mean_absolute_error(y_test_true, pred_test)
rmse = np.sqrt(mean_squared_error(y_test_true, pred_test))
r2 = r2_score(y_test_true, pred_test)

section("📊 Métricas do modelo", None, "📊")

st.write(f"**MAE:** {mae:.3f}")
st.write(f"**RMSE:** {rmse:.3f}")
st.write(f"**R²:** {r2:.3f}")

st.markdown("""
**Interpretação rápida:**

- **MAE** → erro médio absoluto (quanto o modelo erra em média, em unidades do preço).  
- **RMSE** → semelhante ao MAE, mas penaliza mais erros grandes.  
- **R² negativo** → o modelo foi pior que simplesmente prever a **média histórica**.  
""")

st.divider()

# ============================================================
# Plot historical performance (real vs predicted)
# ============================================================
section("📈 Desempenho histórico (real vs previsto)", None, "📉")

fig_hist = go.Figure()
fig_hist.add_trace(
    go.Scatter(
        x=dates_test,
        y=y_test_true,
        mode="lines",
        name="Real",
    )
)
fig_hist.add_trace(
    go.Scatter(
        x=dates_test,
        y=pred_test,
        mode="lines",
        name="Previsto (modelo)",
    )
)
fig_hist.update_layout(
    title="Histórico: Real vs Previsto",
    xaxis_title="Data",
    yaxis_title=target_col,
)
st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ============================================================
# Multi-step OUT-OF-SAMPLE forecast
# ============================================================
section("🔮 Forecast Futuro", f"Prevendo {horizon} dias à frente", "🔮")

last_row = df_ml.iloc[-1:].copy()
future_dates = pd.date_range(
    start=df_ml["date"].iloc[-1] + pd.Timedelta(days=1),
    periods=horizon,
)

forecast_values = []

# current_row: features + lags (sem a coluna date/target)
current_row = last_row.drop(columns=["date", target_col]).copy()

for _ in range(horizon):
    # Previsão no espaço certo (normalizado ou não)
    if normalize_target:
        pred_scaled = model.predict(current_row)[0]
        next_pred = pred_scaled * y_std + y_mean  # volta para unidade original
    else:
        next_pred = model.predict(current_row)[0]

    forecast_values.append(next_pred)

    # Atualiza lags com o valor PREVISTO em unidade original
    if num_lags > 0:
        for i in range(num_lags, 1, -1):
            lag_col_i = f"{target_col}_lag{i}"
            lag_col_prev = f"{target_col}_lag{i-1}"
            if lag_col_i in current_row.columns and lag_col_prev in current_row.columns:
                current_row[lag_col_i] = current_row[lag_col_prev]
        # lag1 recebe a nova previsão (em unidade original)
        lag_col_1 = f"{target_col}_lag1"
        if lag_col_1 in current_row.columns:
            current_row[lag_col_1] = next_pred

# Plot forecast (sempre em unidade original)
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
    title=f"Previsão para {target_col} – {horizon} dias",
    xaxis_title="Data",
    yaxis_title=target_col,
)
st.plotly_chart(fig_f, use_container_width=True)
