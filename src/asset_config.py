"""
Shared asset configuration and picker for all pages.

This module provides:
- ASSET_CATEGORIES: Categorized asset mapping
- ASSETS_MAP: Flattened asset mapping (backward compatible)
- FAVORITE_ASSETS: Quick access favorites
- categorized_asset_picker(): Category-filtered asset selection UI
"""

import streamlit as st

# ============================================================
# Asset Categories
# ============================================================
ASSET_CATEGORIES = {
    "🌾 Complexo Soja": [
        ("Flat do óleo de soja (BRL - C1)", "oleo_flat_brl"),
        ("Flat do óleo de soja (USD - C1)", "oleo_flat_usd"),
        ("Flat do farelo de soja (BRL - C1)", "farelo_flat_brl"),
        ("Flat do farelo de soja (USD - C1)", "farelo_flat_usd"),
        ("Óleo de soja (BOC1)", "boc1"),
        ("Farelo de soja (SMC1)", "smc1"),
        ("Soja (SC1)", "sc1"),
        ("Óleo - Prêmio C1", "so-premp-c1"),
        ("Farelo - Prêmio C1", "sm-premp-c1"),
        ("Soja - Prêmio C1", "sb-premp-c1"),
    ],
    "🌍 Soja Regional": [
        ("Soybean Oil Brazil Paranagua A1", "bo-brzpar-a1"),
        ("Soybean Oil Brazil Paranagua B1", "bo-brzpar-b1"),
        ("Soybean Brazil Paranagua A1", "s-brzpar-a1"),
        ("Soybean Brazil Paranagua B1", "s-brzpar-b1"),
        ("Soybean Meal Brazil Paranagua A1", "sm-brzpar-a1"),
        ("Soybean Meal Brazil Paranagua B1", "sm-brzpar-b1"),
        ("Soybean Oil Lib C1", "sb-oilib-c1"),
        ("Soybean Argentina Ref", "soy-arg-ref"),
    ],
    "🌾 Grãos": [
        ("Milho (CC1)", "cc1"),
        ("Wheat (WC1)", "wc1"),
        ("Kansas Wheat (KWC1)", "kwc1"),
        ("Rice (RBC1)", "rbc1"),
    ],
    "🌴 Óleos & Gorduras": [
        ("Óleo de palma - Ringgit (FCPOC1)", "fcpoc1"),
        ("Óleo de palma - USD (FUPOC1)", "fupoc1"),
        ("Palm Oil Malaysia Crude (P1)", "palm-mycrd-p1"),
        ("Palm Oil Malaysia RBD (P1)", "palm-myrbd-p1"),
        ("Palm Oil SEA (OCRDM)", "palm-sea-ocrdm"),
        ("CPO Indonesia (M1)", "cpo-id-m1"),
        ("CPO Malaysia South (M1)", "cpo-mysth-m1"),
        ("Soybean Oil SEA (DGCF)", "soil-sea-dgcf"),
        ("Sunflower Oil SEA (CRM)", "sunf-sea-crm"),
        ("Canola (RSC1)", "rsc1"),
    ],
    "⚡ Energia": [
        ("Brent (LCOC1)", "lcoc1"),
        ("WTI Crude Oil (CLC1)", "clc1"),
        ("Heating Oil (HOC1)", "hoc1"),
        ("Natural Gas (NGC1)", "ngc1"),
    ],
    "☕ Softs": [
        ("Coffee (KCC1)", "kcc1"),
        ("Cotton (CTC1)", "ctc1"),
        ("Sugar (SBC1)", "sbc1"),
        ("Orange Juice (OJC1)", "ojc1"),
        ("Cocoa (CCC1)", "ccc1"),
    ],
    "🐄 Pecuária": [
        ("Cattle (LCC1)", "lcc1"),
        ("Lean Hogs (LHC1)", "lhc1"),
        ("Feeder Cattle (FCC1)", "fcc1"),
    ],
    "🥇 Metais": [
        ("Gold (GCC1)", "gcc1"),
        ("Silver (SIC1)", "sic1"),
        ("Copper (HGC1)", "hgc1"),
        ("Platinum (PLC1)", "plc1"),
        ("Palladium (PAC1)", "pac1"),
    ],
    "💱 Moedas": [
        ("Dólar (BRL=)", "brl="),
        ("Malaysian Ringgit (MYR=)", "myr="),
    ],
    "₿ Cripto": [
        ("Bitcoin (BTC=)", "btc="),
    ],
    "📊 Ações": [
        ("Petrobras (PETR4)", "petr4.sa"),
        ("Itaúsa (ITSA4)", "itsa4.sa"),
        ("Banco do Brasil (BBAS3)", "bbas3.sa"),
        ("Bradesco (BBDC4)", "bbdc4.sa"),
        ("Taesa (TAEE4)", "taee4.sa"),
        ("Cemig (CMIG4", "cmig4.sa"),
        ("Exxon Mobil (XOM)", "xom"),
        ("AMD (AMD)", "amd.o"),
        ("ARM Holdings (ARMH)", "arm.o"),
        ("Lilly (LLY)", "lly"),               
        ("Nvidia (NVDA)", "nvda.o"),
        ("Novo Nordisk (NVO)", "nvo"),
        ("Bunge Limited (BG)", "bg"),
        ("Archer Daniels Midland (ADM)", "adm"),
    ],
    "📊 FIIs": [
        ("VGIA11", "vgia11.sa"),
        ("BTAL11", "btal11.sa"),
        ("OIAG11", "oiag11.sa"),
        ("BTLG11", "btlg11.sa"),
        ("XPLG11", "xplg11.sa"),
    ],
    "🏦 ETFs": [
        ("IAU", "iau"),
        ("CEF", "cef"),
        ("CPER", "cper.k"),
        ("AIA", "aia.o"),
        ("DIVO", "divo.k"),
    ],
}

# Flatten to old format for compatibility
ASSETS_MAP = {}
for category_assets in ASSET_CATEGORIES.values():
    for label, col in category_assets:
        ASSETS_MAP[label] = col

# Favorites for quick access
FAVORITE_ASSETS = [
    "Flat do óleo de soja (BRL - C1)",
    "Óleo de soja (BOC1)",
    "Farelo de soja (SMC1)",
    "Soja (SC1)",
    "Óleo de palma - Ringgit (FCPOC1)",
    "Dólar (BRL=)",
]


# ============================================================
# Categorized Asset Picker
# ============================================================
def categorized_asset_picker(df_data, state_key="asset_col", show_favorites=True):
    """
    Asset picker with category filter for better UX.

    Args:
        df_data: DataFrame containing the asset columns
        state_key: Unique key for session state
        show_favorites: Whether to show favorite buttons

    Returns:
        tuple: (selected_column, label)
    """
    # Filter available assets
    available_map = {label: col for label, col in ASSETS_MAP.items() if col in df_data.columns}

    if not available_map:
        st.error("Nenhum ativo disponível")
        st.stop()

    # Initialize state
    if state_key not in st.session_state:
        # Default to first favorite that exists
        default = next((ASSETS_MAP[f] for f in FAVORITE_ASSETS if f in available_map), next(iter(available_map.values())))
        st.session_state[state_key] = default

    # Category selection
    col_cat, col_search = st.columns([1, 2])

    with col_cat:
        # Build category filter options
        category_options = ["📂 Todas as categorias"]
        for cat_name in ASSET_CATEGORIES.keys():
            # Check if category has any available assets
            cat_assets = [col for label, col in ASSET_CATEGORIES[cat_name] if col in df_data.columns]
            if cat_assets:
                count = len(cat_assets)
                category_options.append(f"{cat_name} ({count})")

        selected_category = st.selectbox(
            "Categoria",
            options=category_options,
            key=f"{state_key}_category",
            help="Filtre por categoria para encontrar ativos mais facilmente"
        )

    # Filter assets by category
    if selected_category == "📂 Todas as categorias":
        filtered_map = available_map
    else:
        # Extract category name (remove count)
        cat_name = selected_category.rsplit(" (", 1)[0]
        filtered_map = {}
        for label, col in ASSET_CATEGORIES.get(cat_name, []):
            if col in df_data.columns:
                filtered_map[label] = col

    # Favorites row (only if show_favorites and not filtered by category)
    if show_favorites and selected_category == "📂 Todas as categorias":
        st.markdown("**⭐ Favoritos**")
        fav_cols = st.columns(len(FAVORITE_ASSETS))
        for i, fav_label in enumerate(FAVORITE_ASSETS):
            if fav_label in available_map:
                fav_col_name = available_map[fav_label]
                with fav_cols[i]:
                    is_selected = (st.session_state[state_key] == fav_col_name)
                    if st.button(
                        fav_label.split("(")[0].strip(),  # Short name
                        key=f"{state_key}_fav_{i}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state[state_key] = fav_col_name
                        st.rerun()

    # Searchable dropdown
    with col_search:
        labels = list(filtered_map.keys())
        current_col = st.session_state[state_key]

        # Find current label
        current_label = next((lbl for lbl, col in filtered_map.items() if col == current_col), labels[0] if labels else None)

        if not labels:
            st.warning("Nenhum ativo disponível nesta categoria")
            return st.session_state[state_key], current_label

        selected_label = st.selectbox(
            "Buscar ativo",
            options=labels,
            index=labels.index(current_label) if current_label in labels else 0,
            key=f"{state_key}_select",
            help="Digite para buscar ou role para navegar"
        )

        # Update state if changed
        new_col = filtered_map[selected_label]
        if new_col != st.session_state[state_key]:
            st.session_state[state_key] = new_col

    return st.session_state[state_key], selected_label


def get_available_assets(df_data):
    """
    Get list of available assets from ASSETS_MAP that exist in the dataframe.

    Args:
        df_data: DataFrame to check for columns

    Returns:
        dict: Filtered assets map with only available assets
    """
    return {label: col for label, col in ASSETS_MAP.items() if col in df_data.columns}
