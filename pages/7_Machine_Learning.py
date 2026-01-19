"""
Machine Learning Forecast Page
------------------------------

This page allows the user to:

1. Select an asset (target variable).
2. Choose which dataset columns will be used as FEATURES.
3. Choose number of LAGS (0 to 3).
4. Select ML model from multiple options.
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

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor

# Deep Learning models
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from src.data_pipeline import df
from src.utils import apply_theme, date_range_picker
from src.asset_config import ASSETS_MAP, categorized_asset_picker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
BASE = BASE.sort_values("date").reset_index(drop=True)

# Filter to only continuation contracts (exclude monthly contracts)
# Keep only columns that are in ASSETS_MAP (continuation contracts)
valid_asset_cols = [col for col in BASE.columns if col in ASSETS_MAP.values()]
valid_cols = ["date"] + valid_asset_cols
BASE = BASE[valid_cols]

# ============================================================
# Helper Functions
# ============================================================
def add_technical_features(df, target_col, feature_cols):
    """Add technical indicators as features."""
    df_enhanced = df.copy()

    # Returns (daily percentage change)
    for col in [target_col] + feature_cols:
        if col in df_enhanced.columns:
            df_enhanced[f"{col}_return"] = df_enhanced[col].pct_change() * 100

    # Moving averages (5, 10, 20 days)
    for col in [target_col]:
        if col in df_enhanced.columns:
            df_enhanced[f"{col}_ma5"] = df_enhanced[col].rolling(window=5).mean()
            df_enhanced[f"{col}_ma10"] = df_enhanced[col].rolling(window=10).mean()
            df_enhanced[f"{col}_ma20"] = df_enhanced[col].rolling(window=20).mean()

    # Volatility (rolling std of returns)
    for col in [target_col]:
        if f"{col}_return" in df_enhanced.columns:
            df_enhanced[f"{col}_vol10"] = df_enhanced[f"{col}_return"].rolling(window=10).std()

    return df_enhanced


def calculate_feature_importance_score(df, target_col, feature_cols):
    """Calculate correlation-based importance scores for features."""
    scores = {}

    target_data = df[target_col].dropna()

    for col in feature_cols:
        if col != target_col and col in df.columns:
            try:
                # Calculate correlation
                corr_data = df[[target_col, col]].dropna()
                if len(corr_data) > 10:
                    corr = abs(corr_data.corr().iloc[0, 1])
                    scores[col] = corr
            except:
                pass

    return scores


# ============================================================
# Deep Learning Model Builders
# ============================================================
def build_lstm_model(n_features, n_timesteps=10, epochs=50, batch_size=32):
    """
    Build LSTM model for time series forecasting.

    Args:
        n_features: Number of input features
        n_timesteps: Number of timesteps to look back
        epochs: Training epochs
        batch_size: Batch size for training

    Returns:
        Compiled Keras model
    """
    if not HAS_TENSORFLOW:
        return None

    model = Sequential([
        layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.epochs = epochs
    model.batch_size = batch_size
    model.n_timesteps = n_timesteps

    return model


def build_gru_model(n_features, n_timesteps=10, epochs=50, batch_size=32):
    """
    Build GRU model (faster alternative to LSTM).

    Args:
        n_features: Number of input features
        n_timesteps: Number of timesteps to look back
        epochs: Training epochs
        batch_size: Batch size for training

    Returns:
        Compiled Keras model
    """
    if not HAS_TENSORFLOW:
        return None

    model = Sequential([
        layers.GRU(64, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features)),
        layers.Dropout(0.2),
        layers.GRU(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.epochs = epochs
    model.batch_size = batch_size
    model.n_timesteps = n_timesteps

    return model


def build_cnn_model(n_features, n_timesteps=10, epochs=50, batch_size=32):
    """
    Build 1D CNN model for time series pattern recognition.

    Args:
        n_features: Number of input features
        n_timesteps: Number of timesteps to look back
        epochs: Training epochs
        batch_size: Batch size for training

    Returns:
        Compiled Keras model
    """
    if not HAS_TENSORFLOW:
        return None

    model = Sequential([
        layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.epochs = epochs
    model.batch_size = batch_size
    model.n_timesteps = n_timesteps

    return model


def build_deep_mlp_model(n_features, epochs=100, batch_size=32):
    """
    Build deep MLP (Multi-Layer Perceptron) model.

    Args:
        n_features: Number of input features
        epochs: Training epochs
        batch_size: Batch size for training

    Returns:
        Compiled Keras model
    """
    if not HAS_TENSORFLOW:
        return None

    model = Sequential([
        layers.Dense(128, activation='relu', input_shape=(n_features,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.epochs = epochs
    model.batch_size = batch_size
    model.n_timesteps = None  # MLP doesn't need reshaping

    return model


def reshape_for_rnn(X, n_timesteps):
    """
    Reshape data for RNN models (LSTM/GRU/CNN).

    Args:
        X: Input data (samples, features)
        n_timesteps: Number of timesteps

    Returns:
        Reshaped data (samples, timesteps, features)
    """
    n_samples = X.shape[0]
    n_features = X.shape[1] // n_timesteps

    # Ensure we have enough features
    if X.shape[1] % n_timesteps != 0:
        # Trim features to fit timesteps
        n_features = X.shape[1] // n_timesteps
        X = X.iloc[:, :n_features * n_timesteps]

    return X.values.reshape((n_samples, n_timesteps, n_features))


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
    - **Lag 3** = preço de 3 dias atrás (t-3)

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
    | **RMSE** | Raiz do erro quadrático médio | Penaliza erros grandes |
    | **R²** | Coeficiente de determinação | % da variância explicada (0-1) |
    | **MAPE** | Erro percentual médio | Erro em % do valor real |

    ### 🎯 Escolhendo o Modelo

    **📊 Modelos Tradicionais (ML)**
    - **Ridge/Lasso/ElasticNet**: Rápidos, lineares, bons para relações simples
    - **Random Forest/Extra Trees**: Robustos, capturam não-linearidades
    - **Gradient Boosting/XGBoost/LightGBM/CatBoost**: Máxima performance, padrões complexos
    - **AdaBoost**: Simples e eficaz, bom com weak learners

    **🧠 Modelos de Deep Learning**
    - **LSTM** (Long Short-Term Memory): Melhor para dependências de longo prazo, captura padrões temporais complexos
    - **GRU** (Gated Recurrent Unit): Alternativa mais rápida ao LSTM, menos parâmetros, boa performance
    - **1D CNN**: Detecção de padrões em janelas temporais, mais rápido que RNNs
    - **Deep MLP**: Rede neural profunda, captura relações não-lineares complexas

    ⚠️ **Nota**: Modelos de deep learning requerem TensorFlow instalado e mais dados para treinar efetivamente.
    """)

st.divider()

# ============================================================
# Configuration Section with Tabs
# ============================================================
st.markdown("## ⚙️ Configuração do Modelo")

tab_config, tab_features, tab_advanced = st.tabs(["🎯 Target & Modelo", "📊 Features & Engenharia", "⚙️ Avançado"])

# Get valid columns (only continuation contracts)
valid_cols_raw = [c for c in BASE.columns if c not in ["date"] and BASE[c].dtype != "object"]
reverse_map = {col: label for label, col in ASSETS_MAP.items()}

# ============================================================
# TAB 1: Target & Model Selection
# ============================================================
with tab_config:
    with st.container(border=True):
        st.markdown("### 🎯 Ativo Target")

        target_col, target_label = categorized_asset_picker(
            BASE,
            state_key="ml_target",
            show_favorites=True,
        )

    with st.container(border=True):
        st.markdown("### 🤖 Seleção de Modelo(s)")

        col_mode, col_model = st.columns([1, 2])

        with col_mode:
            comparison_mode = st.checkbox(
                "Comparar múltiplos modelos",
                value=False,
                help="Compare o desempenho de todos os modelos disponíveis de uma vez"
            )

        # Build models dictionary with better organization
        models_dict = {
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Lasso Regression": Lasso(alpha=0.01, random_state=42, max_iter=2000),
            "Elastic Net": ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000),
            "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        }

        if HAS_XGB:
            models_dict["XGBoost"] = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                n_jobs=1  # Prevent threading issues
            )

        if HAS_LGBM:
            models_dict["LightGBM"] = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbosity=-1,
                force_col_wise=True
            )

        if HAS_CATBOOST:
            models_dict["CatBoost"] = CatBoostRegressor(
                n_estimators=300,
                learning_rate=0.05,
                depth=6,
                random_state=42,
                verbose=False
            )

        # Deep Learning models (will be added after configuration is done)
        # These need special parameters (epochs, timesteps) from Advanced tab
        dl_models_pending = {}

        if HAS_TENSORFLOW:
            # Mark that DL models are available but not yet configured
            dl_models_pending = {
                "LSTM": "build_lstm",
                "GRU": "build_gru",
                "1D CNN": "build_cnn",
                "Deep MLP": "build_deep_mlp"
            }

        with col_model:
            if not comparison_mode:
                # Include DL models in selection
                all_model_names = list(models_dict.keys()) + list(dl_models_pending.keys())

                model_label = st.selectbox(
                    "Algoritmo",
                    all_model_names,
                    index=0,
                    help="Escolha o algoritmo de ML/DL para treinar"
                )

                # Model description
                model_descriptions = {
                    "Ridge Regression": "✅ Rápido e estável\n✅ Bom para relações lineares\n⚠️ Não captura não-linearidades",
                    "Lasso Regression": "✅ Seleção automática de features\n✅ Interpretável\n⚠️ Pode ser instável",
                    "Elastic Net": "✅ Combina Ridge + Lasso\n✅ Equilibrado\n⚠️ Requer tuning",
                    "Random Forest": "✅ Robusto e preciso\n✅ Captura não-linearidades\n✅ Menos overfitting",
                    "Extra Trees": "✅ Mais rápido que RF\n✅ Maior aleatoriedade\n✅ Boa generalização",
                    "Gradient Boosting": "✅ Alta precisão\n✅ Padrões complexos\n⚠️ Mais lento",
                    "AdaBoost": "✅ Simples e eficaz\n✅ Bom com weak learners\n⚠️ Sensível a outliers",
                    "XGBoost": "✅ Estado da arte\n✅ Excelente performance\n✅ Rápido e eficiente",
                    "LightGBM": "✅ Mais rápido que XGBoost\n✅ Eficiente em memória\n✅ Ótima precisão",
                    "CatBoost": "✅ Melhor precisão\n✅ Robust a overfitting\n✅ Handles missing values",
                    "LSTM": "🧠 Deep Learning\n✅ Captura dependências longas\n✅ Excelente para séries temporais\n⚠️ Requer mais dados e tempo",
                    "GRU": "🧠 Deep Learning\n✅ Mais rápido que LSTM\n✅ Menos parâmetros\n✅ Boa performance geral",
                    "1D CNN": "🧠 Deep Learning\n✅ Detecção de padrões\n✅ Mais rápido que RNNs\n✅ Bom para tendências locais",
                    "Deep MLP": "🧠 Deep Learning\n✅ Captura relações complexas\n✅ Rede profunda\n⚠️ Não usa sequência temporal",
                }
                st.caption(model_descriptions.get(model_label, ""))

                # Will be set later after DL configuration
                models_to_train = {}
            else:
                st.info(f"🔄 Modo de comparação ativado: Todos os modelos serão treinados (ML + DL se disponível)")
                # Will include DL models after configuration
                models_to_train = {}

    with st.container(border=True):
        st.markdown("### 📅 Configurações Temporais")

        col_lags, col_horizon = st.columns(2)

        with col_lags:
            num_lags = st.slider(
                "Número de lags",
                min_value=0,
                max_value=10,
                value=3,
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
                max_value=60,
                value=30,
                help="Quantidade de dias futuros a prever"
            )
            st.caption(f"🔮 Prevendo {horizon} dias")

# ============================================================
# TAB 2: Feature Selection & Engineering
# ============================================================
with tab_features:
    with st.container(border=True):
        st.markdown("### 🔍 Seleção de Features")

        # Calculate correlations with target
        correlations = {}
        if target_col:
            for col in valid_cols_raw:
                if col != target_col:
                    try:
                        corr_data = BASE[[target_col, col]].dropna()
                        if len(corr_data) > 10:
                            corr = corr_data.corr().iloc[0, 1]
                            correlations[col] = abs(corr)
                    except:
                        pass

        # Smart feature selection methods
        st.markdown("#### Método de Seleção")
        selection_method = st.radio(
            "Como selecionar features?",
            ["🎯 Automático (Top N)", "✋ Manual", "🔝 Por threshold de correlação"],
            help="Escolha como selecionar as features para o modelo"
        )

        feature_cols = []

        if selection_method == "🎯 Automático (Top N)":
            num_features = st.slider(
                "Número de features",
                min_value=1,
                max_value=min(20, len(correlations)),
                value=min(5, len(correlations)),
                help="Seleciona automaticamente as N features mais correlacionadas"
            )

            if correlations:
                sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                feature_cols = [feat[0] for feat in sorted_features[:num_features]]

                st.success(f"✅ Selecionadas {len(feature_cols)} features automaticamente")

        elif selection_method == "✋ Manual":
            # Get top 5 as default
            default_features = []
            if correlations:
                sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                default_features = [feat[0] for feat in sorted_features[:5]]

            # Get labels for features
            feature_options_labels = [reverse_map.get(col, col) for col in valid_cols_raw if col != target_col]
            default_features_labels = [reverse_map.get(col, col) for col in default_features]

            selected_features_labels = st.multiselect(
                "Selecione as features manualmente",
                options=feature_options_labels,
                default=default_features_labels,
                help="💡 As 5 features mais correlacionadas estão pré-selecionadas"
            )

            # Convert labels back to columns
            label_to_col = {label: col for col, label in reverse_map.items()}
            feature_cols = [label_to_col.get(label, label) for label in selected_features_labels]

        else:  # Threshold method
            corr_threshold = st.slider(
                "Threshold de correlação",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Seleciona apenas features com |correlação| >= threshold"
            )

            if correlations:
                feature_cols = [col for col, corr in correlations.items() if corr >= corr_threshold]
                st.success(f"✅ {len(feature_cols)} features com correlação >= {corr_threshold:.2f}")

    # Show correlation matrix for selected features
    if feature_cols and len(feature_cols) > 0:
        with st.expander(f"📊 Matriz de Correlação das Features Selecionadas ({len(feature_cols)} features)", expanded=False):
            # Show correlations for selected features
            selected_corrs = [(reverse_map.get(f, f), correlations.get(f, 0)) for f in feature_cols if f in correlations]
            selected_corrs.sort(key=lambda x: x[1], reverse=True)

            col_list, col_viz = st.columns([1, 2])

            with col_list:
                st.markdown("**Features (ordenadas por correlação com target):**")
                for i, (feat_label, corr) in enumerate(selected_corrs, 1):
                    corr_strength = "Forte" if corr > 0.7 else "Moderada" if corr > 0.4 else "Fraca"
                    corr_color = "🟢" if corr > 0.7 else "🟡" if corr > 0.4 else "🔴"
                    st.caption(f"{i}. **{feat_label}**: {corr:.3f} {corr_color} ({corr_strength})")

            with col_viz:
                # Create simple correlation bar chart
                fig_corr = go.Figure()

                fig_corr.add_trace(
                    go.Bar(
                        x=[corr for _, corr in selected_corrs],
                        y=[label for label, _ in selected_corrs],
                        orientation='h',
                        marker=dict(
                            color=[corr for _, corr in selected_corrs],
                            colorscale='RdYlGn',
                            showscale=False,
                            cmin=0,
                            cmax=1
                        ),
                    )
                )

                fig_corr.update_layout(
                    title="Correlação com Target",
                    xaxis_title="Correlação Absoluta",
                    yaxis_title="",
                    height=max(300, len(selected_corrs) * 25),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=200, r=20, t=40, b=40)
                )

                st.plotly_chart(fig_corr, use_container_width=True)

    # Feature engineering toggle
    with st.container(border=True):
        st.markdown("### 🔧 Engenharia de Features")

        add_tech_features = st.checkbox(
            "Adicionar indicadores técnicos",
            value=False,
            help="Adiciona automaticamente: retornos diários, médias móveis (5/10/20), volatilidade rolling"
        )

        if add_tech_features:
            st.info("""
            ✅ **Features técnicas que serão adicionadas:**
            - 📈 Retornos diários (%) para target e features
            - 📊 Médias móveis 5/10/20 dias do target
            - 📉 Volatilidade rolling (10 dias) do target
            """)

    # At least 1 feature or lag must exist
    if len(feature_cols) == 0 and num_lags == 0:
        st.warning("⚠️ Selecione ao menos uma feature ou configure lags para treinar o modelo.")

# ============================================================
# TAB 3: Advanced Settings
# ============================================================
with tab_advanced:
    with st.container(border=True):
        st.markdown("### 📅 Período de Treinamento")

        start_model_date, end_model_date = date_range_picker(
            BASE["date"],
            state_key="ml_train_range",
            default_days=365 * 3,
        )

    with st.container(border=True):
        st.markdown("### ⚙️ Configurações de Treinamento")

        col_norm, col_split = st.columns(2)

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

        with col_split:
            train_split = st.slider(
                "% Dados de treinamento",
                min_value=60,
                max_value=90,
                value=80,
                step=5,
                help="Percentual dos dados para treino (resto vai para teste)"
            )
            st.caption(f"✓ {train_split}% treino / {100-train_split}% teste")

    # Deep Learning specific configuration
    if HAS_TENSORFLOW and len(dl_models_pending) > 0:
        with st.container(border=True):
            st.markdown("### 🧠 Configurações de Deep Learning")

            col_dl1, col_dl2, col_dl3 = st.columns(3)

            with col_dl1:
                dl_epochs = st.slider(
                    "Epochs",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Número de épocas de treinamento. Mais épocas = mais tempo mas melhor aprendizado"
                )
                st.caption(f"✓ {dl_epochs} iterações completas")

            with col_dl2:
                dl_batch_size = st.slider(
                    "Batch Size",
                    min_value=8,
                    max_value=128,
                    value=32,
                    step=8,
                    help="Tamanho do lote para atualização dos pesos"
                )
                st.caption(f"✓ {dl_batch_size} samples por batch")

            with col_dl3:
                dl_timesteps = st.slider(
                    "Timesteps (RNN/CNN)",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5,
                    help="Janela temporal para LSTM/GRU/CNN (quantos passos anteriores usar)"
                )
                st.caption(f"✓ {dl_timesteps} passos de tempo")

            st.info("""
            ℹ️ **Dicas de configuração:**
            - **Epochs**: Comece com 50. Aumente se o modelo ainda está melhorando.
            - **Batch Size**: 32 é um bom padrão. Menor = mais preciso mas mais lento.
            - **Timesteps**: 10-15 é ideal para séries diárias. Mais = captura padrões de longo prazo.
            """)

# Set default DL parameters if not configured
if not HAS_TENSORFLOW or len(dl_models_pending) == 0:
    dl_epochs = 50
    dl_batch_size = 32
    dl_timesteps = 10

# Apply date range filter
mask_model = (BASE["date"].dt.date >= start_model_date) & (BASE["date"].dt.date <= end_model_date)
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
# Build Deep Learning models now that we have configuration
# ============================================================
# This needs to happen AFTER feature configuration so we know n_features
# We'll add them to models_dict after data preparation


# ============================================================
# Prepare dataset (lags + features)
# ============================================================
st.markdown("## 🔄 Preparação dos Dados")

with st.spinner("Preparando dados e aplicando engenharia de features..."):
    # Start with base columns
    df_ml = BASE_RANGE[["date", target_col] + feature_cols].copy()

    # Ensure date column is datetime type
    df_ml["date"] = pd.to_datetime(df_ml["date"], errors="coerce")

    # Apply feature engineering if requested
    if add_tech_features:
        df_ml = add_technical_features(df_ml, target_col, feature_cols)
        st.success(f"✅ Features técnicas adicionadas. Dataset expandido para {len(df_ml.columns)-2} features totais")

    # Generate lag columns (on target)
    if num_lags > 0:
        for lag in range(1, num_lags + 1):
            df_ml[f"{target_col}_lag{lag}"] = df_ml[target_col].shift(lag)
        st.success(f"✅ {num_lags} lags criados para o target")

    # Drop rows with NaN caused by lags or missing features
    df_ml = df_ml.dropna().reset_index(drop=True)

    if df_ml.empty:
        st.error("❌ Dados insuficientes após aplicar lags e filtrar NaNs.")
        st.stop()

    # Build X, y
    X = df_ml.drop(columns=["date", target_col])
    y = df_ml[target_col]

    feature_names = X.columns.tolist()

    # Train-test split (temporal)
    split = int(len(df_ml) * (train_split / 100))
    X_train_raw, X_test_raw = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train_raw, y_test_raw = y.iloc[:split].copy(), y.iloc[split:].copy()
    dates_test = df_ml["date"].iloc[split:]

    st.info(f"""
    📊 **Dataset preparado com sucesso:**
    - Total de samples: {len(df_ml)}
    - Features finais: {len(feature_names)}
    - Treino: {len(X_train_raw)} samples ({train_split}%)
    - Teste: {len(X_test_raw)} samples ({100-train_split}%)
    """)

# ============================================================
# Build Deep Learning Models (now that we know n_features)
# ============================================================
if HAS_TENSORFLOW and len(dl_models_pending) > 0:
    n_features = len(feature_names)

    # Calculate features per timestep for RNN models
    n_features_per_timestep = max(1, n_features // dl_timesteps)

    # Build DL models
    if "LSTM" in dl_models_pending:
        models_dict["LSTM"] = build_lstm_model(
            n_features=n_features_per_timestep,
            n_timesteps=dl_timesteps,
            epochs=dl_epochs,
            batch_size=dl_batch_size
        )

    if "GRU" in dl_models_pending:
        models_dict["GRU"] = build_gru_model(
            n_features=n_features_per_timestep,
            n_timesteps=dl_timesteps,
            epochs=dl_epochs,
            batch_size=dl_batch_size
        )

    if "1D CNN" in dl_models_pending:
        models_dict["1D CNN"] = build_cnn_model(
            n_features=n_features_per_timestep,
            n_timesteps=dl_timesteps,
            epochs=dl_epochs,
            batch_size=dl_batch_size
        )

    if "Deep MLP" in dl_models_pending:
        models_dict["Deep MLP"] = build_deep_mlp_model(
            n_features=n_features,
            epochs=dl_epochs,
            batch_size=dl_batch_size
        )

# Finalize models_to_train based on mode
if not comparison_mode:
    # Single model mode
    if model_label in models_dict:
        models_to_train = {model_label: models_dict[model_label]}
    else:
        st.error(f"❌ Modelo {model_label} não disponível. Verifique se as dependências estão instaladas.")
        st.stop()
else:
    # Comparison mode - all available models
    models_to_train = models_dict.copy()

# ============================================================
# Train Model(s)
# ============================================================
st.divider()
st.markdown("## 🤖 Treinamento do(s) Modelo(s)")

# Prepare scalers
x_scaler = None
y_scaler = None

if normalize:
    x_scaler = StandardScaler()
    X_train = pd.DataFrame(
        x_scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index
    )
    X_test = pd.DataFrame(
        x_scaler.transform(X_test_raw),
        columns=X_test_raw.columns,
        index=X_test_raw.index
    )

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel()
    y_train = y_train_scaled
    y_test = y_test_raw.copy()
else:
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    y_train = y_train_raw.copy()
    y_test = y_test_raw.copy()

# Train models
trained_models = {}
model_predictions = {}
model_metrics = {}

progress_bar = st.progress(0)
status_text = st.empty()

for idx, (model_name, model) in enumerate(models_to_train.items()):
    status_text.text(f"Treinando {model_name}... ({idx+1}/{len(models_to_train)})")

    try:
        # Check if it's a Keras model
        is_keras_model = HAS_TENSORFLOW and hasattr(model, 'fit') and hasattr(model, 'epochs')

        if is_keras_model:
            # Deep Learning model - needs special handling
            epochs = model.epochs
            batch_size = model.batch_size
            n_timesteps = model.n_timesteps

            # Prepare data based on model type
            if n_timesteps is not None:
                # RNN models (LSTM, GRU, CNN) - need 3D reshaping
                X_train_dl = reshape_for_rnn(X_train, n_timesteps)
                X_test_dl = reshape_for_rnn(X_test, n_timesteps)
            else:
                # MLP - use 2D as is
                X_train_dl = X_train.values
                X_test_dl = X_test.values

            # Train with early stopping
            if HAS_TENSORFLOW:
                early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
                model.fit(
                    X_train_dl,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[early_stop],
                    validation_split=0.1
                )

            # Predict
            pred_test_scaled = model.predict(X_test_dl, verbose=0).ravel()

            if normalize and y_scaler is not None:
                pred_test = y_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()
            else:
                pred_test = pred_test_scaled

        elif model_name == "XGBoost" and HAS_XGB:
            # XGBoost specific handling
            model.fit(X_train, y_train, verbose=False)
            # Predict
            if normalize and y_scaler is not None:
                pred_test_scaled = model.predict(X_test)
                pred_test = y_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()
            else:
                pred_test = model.predict(X_test)

        elif model_name == "LightGBM" and HAS_LGBM:
            # LightGBM specific handling
            model.fit(X_train, y_train, verbose=False)
            # Predict
            if normalize and y_scaler is not None:
                pred_test_scaled = model.predict(X_test)
                pred_test = y_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()
            else:
                pred_test = model.predict(X_test)

        elif model_name == "CatBoost" and HAS_CATBOOST:
            # CatBoost specific handling
            model.fit(X_train, y_train, verbose=False)
            # Predict
            if normalize and y_scaler is not None:
                pred_test_scaled = model.predict(X_test)
                pred_test = y_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()
            else:
                pred_test = model.predict(X_test)

        else:
            # Standard sklearn models
            model.fit(X_train, y_train)
            # Predict
            if normalize and y_scaler is not None:
                pred_test_scaled = model.predict(X_test)
                pred_test = y_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()
            else:
                pred_test = model.predict(X_test)

        # Store results
        trained_models[model_name] = model
        model_predictions[model_name] = pred_test

        # Calculate metrics
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        r2 = r2_score(y_test, pred_test)

        # MAPE
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - pred_test[mask]) / y_test[mask])) * 100
        else:
            mape = np.inf

        # MASE
        naive_error = np.mean(np.abs(np.diff(y_test.values)))
        mase = mae / (naive_error + 1e-8) if naive_error > 0 else np.inf

        model_metrics[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'MASE': mase
        }

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        st.warning(f"⚠️ Erro ao treinar {model_name}: {str(e)}")

        # Show detailed error in expander for debugging
        with st.expander(f"🔧 Detalhes do erro - {model_name}"):
            st.code(error_detail)
        continue

    progress_bar.progress((idx + 1) / len(models_to_train))

status_text.text("✅ Treinamento concluído!")
progress_bar.empty()

if not trained_models:
    st.error("❌ Nenhum modelo foi treinado com sucesso.")
    st.stop()

# Select best model for single mode or show comparison
if comparison_mode:
    st.success(f"✅ {len(trained_models)} modelos treinados com sucesso!")
else:
    model_label = list(trained_models.keys())[0]
    model = trained_models[model_label]
    pred_test = model_predictions[model_label]
    y_test_true = y_test.copy()
    st.success(f"✅ Modelo {model_label} treinado com sucesso!")

# ============================================================
# Show metrics with professional cards
# ============================================================
st.divider()

if comparison_mode:
    # ============================================================
    # MODEL COMPARISON MODE
    # ============================================================
    st.markdown("## 📊 Comparação de Modelos")

    # Create comparison dataframe
    comparison_df = pd.DataFrame(model_metrics).T
    comparison_df = comparison_df.sort_values('R2', ascending=False)

    # Display comparison table
    with st.container(border=True):
        st.markdown("### 🏆 Ranking de Modelos (por R²)")

        # Format for display
        display_df = comparison_df.copy()
        display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.2f}")
        display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:.2f}")
        display_df['R2'] = display_df['R2'].apply(lambda x: f"{x:.3f}")
        display_df['MAPE'] = display_df['MAPE'].apply(lambda x: f"{x:.1f}%" if not np.isinf(x) else "N/A")
        display_df['MASE'] = display_df['MASE'].apply(lambda x: f"{x:.2f}" if not np.isinf(x) else "N/A")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=min(400, (len(display_df) + 1) * 35)
        )

    # Visual comparison charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # MAE comparison
        fig_mae = go.Figure()
        fig_mae.add_trace(
            go.Bar(
                x=list(comparison_df.index),
                y=comparison_df['MAE'],
                marker=dict(
                    color=comparison_df['MAE'],
                    colorscale='RdYlGn_r',
                    showscale=False
                ),
                text=comparison_df['MAE'].round(2),
                textposition='outside',
            )
        )
        fig_mae.update_layout(
            title="MAE por Modelo (menor = melhor)",
            xaxis_title="Modelo",
            yaxis_title="MAE",
            height=400
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with col_chart2:
        # R² comparison
        fig_r2 = go.Figure()
        fig_r2.add_trace(
            go.Bar(
                x=list(comparison_df.index),
                y=comparison_df['R2'],
                marker=dict(
                    color=comparison_df['R2'],
                    colorscale='RdYlGn',
                    showscale=False,
                    cmin=0,
                    cmax=1
                ),
                text=comparison_df['R2'].round(3),
                textposition='outside',
            )
        )
        fig_r2.update_layout(
            title="R² por Modelo (maior = melhor)",
            xaxis_title="Modelo",
            yaxis_title="R²",
            height=400
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # Best model selection
    best_model_name = comparison_df.index[0]
    st.success(f"🏆 **Melhor modelo (por R²):** {best_model_name} com R² = {comparison_df.loc[best_model_name, 'R2']:.3f}")

    # Use best model for forecast
    model = trained_models[best_model_name]
    model_label = best_model_name
    pred_test = model_predictions[best_model_name]
    y_test_true = y_test.copy()

    # Calculate residuals for best model
    residuals = y_test_true.values - pred_test

else:
    # ============================================================
    # SINGLE MODEL MODE
    # ============================================================
    st.markdown("## 📊 Performance do Modelo")

    with st.container(border=True):
        st.markdown("### 🎯 Métricas de Erro (Conjunto de Teste)")

        # Get metrics from stored results
        mae = model_metrics[model_label]['MAE']
        rmse = model_metrics[model_label]['RMSE']
        r2 = model_metrics[model_label]['R2']
        mape = model_metrics[model_label]['MAPE']
        mase = model_metrics[model_label]['MASE']

        # Mean of actual values for context
        y_mean = y_test_true.mean()

        # Calculate residuals
        residuals = y_test_true.values - pred_test

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

# Modelos tipo árvore (Random Forest, XGBoost, etc)
if hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_
    importance_type = "Importância (Gini/Gain)"

# Modelos lineares (Ridge, Lasso) – usamos o valor absoluto dos coeficientes
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
# Calculate residuals ONCE here for use everywhere
residuals = y_test_true.values - pred_test

with st.expander("🔍 Análise de Resíduos (Diagnósticos do Modelo)", expanded=False):
    st.markdown("""
    ### 📊 O que são Resíduos?

    **Resíduos = Valor Real - Previsão**
    - **Positivo**: Modelo subestimou (previu menos)
    - **Negativo**: Modelo superestimou (previu mais)
    - **Próximo de 0**: Previsão precisa
    """)

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

fig_hist.update_layout(
    title="Comparação: Valores Reais vs Previsões do Modelo",
    xaxis_title="Data",
    yaxis_title=f"{target_col} (Preço)",
    hovermode='x unified',
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ============================================================
# Multi-step OUT-OF-SAMPLE forecast
# ============================================================
st.markdown(f"### 🔮 Previsão Futura ({horizon} dias)")

try:
    # Get last row for prediction
    last_row = df_ml.iloc[-1:].copy()
    last_date = pd.to_datetime(df_ml["date"].iloc[-1])

    # Create future dates properly - use a list of Timestamps
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    future_dates = pd.DatetimeIndex(future_dates)

    forecast_values = []

    # Convert to DataFrame for easier manipulation (preserve column names)
    current_row_df = last_row.drop(columns=["date", target_col]).copy()

    # Multi-step forecasting
    for step in range(horizon):
        if normalize and x_scaler is not None and y_scaler is not None:
            # Transform using DataFrame to preserve column names
            current_x_scaled = pd.DataFrame(
                x_scaler.transform(current_row_df),
                columns=current_row_df.columns
            )
            next_pred_scaled = model.predict(current_x_scaled)[0]
            next_pred = float(y_scaler.inverse_transform(
                np.array([[next_pred_scaled]])
            )[0, 0])
        else:
            next_pred = float(model.predict(current_row_df)[0])

        forecast_values.append(next_pred)

        # Update lags if using them
        if num_lags > 0:
            for i in range(num_lags, 1, -1):
                lag_col = f"{target_col}_lag{i}"
                prev_lag_col = f"{target_col}_lag{i-1}"
                if lag_col in current_row_df.columns and prev_lag_col in current_row_df.columns:
                    current_row_df[lag_col] = current_row_df[prev_lag_col].values
            current_row_df[f"{target_col}_lag1"] = next_pred

    forecast_values = np.array(forecast_values, dtype=float)

    # Calculate confidence interval
    forecast_std = float(residuals.std())
    forecast_steps = np.arange(1, horizon + 1, dtype=float)

    # Uncertainty grows with horizon
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

    # Add vertical line separating history from forecast using add_shape
    fig_combined.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dot"),
    )

    # Add annotation for the line
    fig_combined.add_annotation(
        x=last_date,
        y=1.05,
        yref="paper",
        text="Início da Previsão",
        showarrow=False,
        font=dict(size=10, color="gray"),
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

        - **Não prevê eventos**: Choques de mercado, notícias, etc não são previstos
        - **Pressupõe continuidade**: Assume que padrões históricos persistem
        - **Pior longe no futuro**: A incerteza cresce com o horizonte
        - **Sensível ao período de treinamento**: Diferentes períodos = diferentes previsões

        **Para {horizon} dias:** Intervalo estimado = {forecast_std:.2f} ± {1.96 * forecast_std:.2f} (base) até {1.96 * forecast_std * (1 + 0.05*horizon):.2f} (final)
        """)

    st.divider()

except Exception as e:
    st.error(f"❌ Erro ao gerar previsão: {str(e)}")
    st.info("💡 Dica: Verifique se há dados suficientes e se as configurações estão adequadas.")
    import traceback
    with st.expander("🔧 Detalhes do erro (para debug)"):
        st.code(traceback.format_exc())

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
