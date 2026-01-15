# AI Copilot Instructions for Market Monitor Panel

## Project Overview
**Market Monitor Panel** is a Streamlit-based financial analytics dashboard for commodity market analysis, focused on Brazilian agricultural commodities (soybeans, oils, meal ratios). The application analyzes price ratios, correlations, technical indicators (RSI, moving averages), and arbitrage opportunities.

## Architecture

### Core Structure
- **`app.py`**: Streamlit multi-page router - initializes page navigation with 7 analysis modules
- **`src/`**: Shared utilities and data processing
  - `data_pipeline.py`: Data loading, cleaning, ratio calculations (Óleo/Farelo, Óleo/Palma, Óleo/Diesel)
  - `visualization.py`: Plotly/Matplotlib charting (ratio analysis, technical overlays)
  - `utils.py`: UI helpers (theme, asset pickers, RSI calculation, date ranges)
  - `config.py`: Path management for data and source directories
- **`pages/`**: Individual Streamlit apps (numbered 1-7, auto-discovered by nav)
- **`data/`**: CSV source files and market intelligence data
- **`notebooks/`**: Jupyter exploration scripts (separate from main app)

### Data Flow Pattern
1. **Load**: `load_data_csv()` in `data_pipeline.py` standardizes date columns and sorts data
2. **Transform**: Apply ratio formulas (conversion factors: `TON_OLEO = 22.0462`, `TON_FARELO = 1.1023`)
3. **Filter**: Date range masking using `pd.to_datetime().dt.date` for comparison
4. **Visualize**: Pass to Plotly/Matplotlib via `visualization.py` functions
5. **Interact**: Streamlit state management (`st.session_state`) for widget persistence

## Key Patterns & Conventions

### Data Handling
- **CSV structure**: Always expect `date` column (or variants like `Date`); `load_data_csv()` normalizes to lowercase
- **Date standardization**: Use `pd.to_datetime(..., errors="coerce")` to handle parsing failures gracefully
- **Ratio calculations**: Pre-computed in `data_pipeline.py` as globals (`oleo_farelo`, `oleo_palma`, etc.)
- **DataFrame layout**: Sort by date ascending, reset index before analysis

### UI/Visualization
- **Theme**: Dark mode via `THEME` dict in `utils.py`; apply with `apply_theme()` in every page
- **Asset pickers**: Use `asset_picker()` (grid buttons) or `asset_picker_dropdown()` (searchable dropdown)
- **Section headers**: `section(text, subtitle=None, icon="")` for styled headings
- **Plotly defaults**: Template `"plotly_dark"`, explicit height specification, include ticker/label in series name
- **Empty data handling**: Return `_empty_fig_plotly(msg="...")` rather than erroring

### Technical Indicators
- **RSI**: `rsi(df, ticker_col, date_col='date', window=14)` returns df with `RSI` column
- **Moving Averages**: List support in visualization (e.g., `ma_windows=[20, 50, 90, 200]`)
- **Rolling metrics**: Use `df.rolling(window, min_periods=max(2, window//4))` to avoid sparse data artifacts

### Code Organization
- **Naming**: Portuguese variable names acceptable (e.g., `view_of`, `mask_op`), but keep function names descriptive
- **Docstrings**: Include parameter types, returns, and error conditions (see `load_data_csv()` example)
- **Imports**: Group stdlib → pandas/numpy → src modules; use absolute imports from `src/`
- **Constants**: Define at module top (e.g., conversion factors, default windows)

## Critical Issues & Common Bugs

### Known Notebook Issues
**File**: `notebooks/report_ratio.ipynb` contains critical bugs:
- **Line 24**: `RATIO_OLEO_PALMA` and `RATIO_OLEO_DIESEL` both assigned to `"Óleo/Farelo"` instead of their own values
- **Line 38**: `df_op` masked with `mask_of` instead of `mask_op` → wrong data sliced
- **Line 75**: Reference to undefined `view` and `RATIO_LABEL` variables → NameError
- **Fix approach**: Use corresponding variables (`view_op`, `view_od`, `view_of`) and maintain 1:1 mapping

### Variable Lifecycle
- Avoid reassigning `y_col` across multiple ratio loads (each ratio should have isolated scope)
- Ensure all user-facing variables referenced in f-strings are defined in same scope
- Check mask variable names match dataframe names (e.g., `mask_op` → `view_op`)

## Debugging & Development Workflows

### Running the App
```bash
streamlit run app.py
```
Streamlit auto-reloads on file changes; check browser console and terminal for errors.

### Testing Data Loads
In notebooks/scripts, verify CSV column names first:
```python
df = pd.read_csv("data/commodities_data.csv")
print(df.columns)  # Confirm 'date' or 'Date' exists
```

### Date Filtering Issues
Always convert both series and filter date to same type:
```python
df["date"] = pd.to_datetime(df["date"], errors="coerce")
start_date = pd.to_datetime(START_DATE).date()  # .date() strips time
mask = (df["date"].dt.date >= start_date)  # Use .dt.date for comparison
```

## Integration Points & Dependencies

### External Libraries
- **Streamlit 1.37.1**: Page config, widgets, session state, caching
- **Plotly 5.20+**: Interactive charts with `make_subplots` for multi-axis layouts
- **Pandas 2.2.2**: All data manipulation; `.rolling()`, `.groupby()`, `.merge()`
- **Scikit-learn 1.4.2**: For ML page (7) - correlation matrices, preprocessing
- **XGBoost 1.7+**: Optional for ML predictions

### Page Communication
- Pages are isolated; share data via `src/data_pipeline.py` functions and CSV files
- Use `st.session_state` for cross-widget state within a page only
- For cross-page data: ensure both pages call `load_data_csv()` independently or refactor into shared functions

## Coding Standards for Modifications

1. **New page additions**: Copy structure from existing pages; call `apply_theme()` first
2. **New ratios**: Add to `RATIOS` dict in `data_pipeline.py` with `(dataframe, column_name)` tuple
3. **New visualizations**: Extend `visualization.py`; accept x/y series + theme dict
4. **Error handling**: Check for empty DataFrames before plotting; return informative messages
5. **Docstring format**: Use NumPy-style docstrings (see `load_data_csv()` in `data_pipeline.py`)

## Language & Locale
- **Application language**: Portuguese (UI text, comments, variable names)
- **Date format**: ISO 8601 in code; display format handled by Streamlit/Plotly
- **Decimal separator**: Pandas handles locale-agnostic; format output explicitly for display

---
*Last updated: January 2026. For large refactors or new features, preserve backward compatibility with existing CSV structures.*
