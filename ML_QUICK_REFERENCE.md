# Machine Learning Page - Quick Reference Guide

## 🎯 Model Selection Flowchart

```
Do you want to understand WHY?
├─ YES → Ridge or Lasso (interpretable coefficients)
└─ NO → Go to next question

Do you have < 1000 rows or few features?
├─ YES → Ridge, Lasso, Random Forest
└─ NO → XGBoost, LightGBM, Neural Network

Does your data have strong non-linear patterns?
├─ YES → Gradient Boosting, XGBoost, Neural Network
└─ NO → Ridge, Lasso, Random Forest

Do you want fastest training?
├─ YES → LightGBM or Ridge
└─ NO → XGBoost or Neural Network
```

---

## 📊 Metrics Interpretation Quick Guide

### MAE (Mean Absolute Error)
- **What**: Average error in original units
- **Good**: Low values
- **Use When**: You care about typical error magnitude
- **Example**: MAE=5 means "off by 5 on average"

### RMSE (Root Mean Squared Error)
- **What**: Square root of average squared error
- **Good**: Low values
- **Use When**: Large errors are especially bad
- **Example**: RMSE=7 penalizes outliers more than MAE=5

### R² (Coefficient of Determination)
- **What**: % of variance explained by model
- **Range**: 0 to 1 (can be negative)
- **Good**: > 0.7 is strong
- **Example**: R²=0.85 means model explains 85% of variation

### MAPE (Mean Absolute Percentage Error)
- **What**: Average % error (relative to actual values)
- **Good**: Low values (< 5% is excellent)
- **Use When**: Comparing across different scales
- **Example**: MAPE=3% means "off by 3% on average"

### MASE (Mean Absolute Scaled Error)
- **What**: Error relative to naive forecast (just repeat last value)
- **Range**: 0 to infinity
- **Good**: < 1.0 (better than naive)
- **Example**: MASE=0.8 means "80% as bad as naive forecast"

---

## 🔴 Red Flags & Troubleshooting

### Problem: R² is negative
- **Cause**: Model worse than just predicting the average
- **Solution**: 
  - Add more relevant features
  - Try different model
  - Check data for errors
  - Use more/better lags

### Problem: Large RMSE but low MAE
- **Cause**: Few very large errors
- **Solution**:
  - Check for outliers in data
  - Try Robust models
  - Increase lags
  - Verify data quality

### Problem: MAPE is very high
- **Cause**: Target values near zero or highly variable
- **Solution**:
  - Use MAE instead (MAPE less reliable)
  - Add features explaining variability
  - Use longer training period
  - Consider log transformation

### Problem: Mean of residuals ≠ 0
- **Cause**: Model has systematic bias
- **Solution**:
  - Retrain model (should correct itself)
  - Add bias correction term
  - Check if features are properly scaled
  - Verify no data leakage

### Problem: Residuals show pattern/trend
- **Cause**: Model missing temporal pattern
- **Solution**:
  - Increase number of lags
  - Add seasonal features
  - Try model that captures trends
  - Check for structural breaks

---

## 🚀 Getting Better Predictions

### Quick Wins (Easy, High Impact)

1. **Add More Lags** (5-10 usually good)
   - Captures short-term momentum
   - Usually helps accuracy 2-5%

2. **Select Better Features** 
   - Remove unrelated variables
   - Keep only significant correlations
   - Reduces noise, improves R²

3. **Extend Training Period**
   - More data = better patterns
   - But remove old irrelevant data
   - 2-5 years usually optimal

4. **Use XGBoost or LightGBM**
   - Usually 5-15% better than Ridge/RF
   - Minimal tuning needed
   - Better for commodity prices

### Medium Effort (More Tuning)

5. **Feature Engineering**
   - Add ratios/differences
   - Create interaction terms
   - Add external indicators

6. **Hyperparameter Tuning**
   - Different learning rates
   - Different tree depths
   - Different regularization

7. **Ensemble Methods**
   - Combine predictions from multiple models
   - Reduces overfitting

### Advanced (Complex)

8. **Custom Loss Functions**
   - Optimize for specific metric
   - Handle asymmetric costs

9. **Time Series Specific Models**
   - ARIMA, SARIMA
   - Prophet, LSTM
   - Better for seasonal data

---

## 📈 Forecast Confidence Levels

### What the Interval Means

```
90% Probability the True Value Falls Here:
    ↓────────────────────────────↓
    │  Prediction  │  Prediction  │
    │   -1.645σ    │   +1.645σ    │
    ↓────────────────────────────↓
    
95% Probability (95% Confidence):
    ↓────────────────────────────┬──────────────┐
    │  Prediction  │  Prediction  │              │
    │   -1.96σ     │   +1.96σ     │   Wider!    │
    ↓────────────────────────────┴──────────────┐
```

### Width Interpretation

| Width | Meaning | Action |
|-------|---------|--------|
| Very Narrow (2-3%) | High confidence | Can use for trading |
| Medium (5-10%) | Moderate confidence | Use with caution |
| Wide (>15%) | Low confidence | Multiple scenarios |
| Very Wide (>25%) | Highly uncertain | Don't rely on point forecast |

---

## 🎓 Common Mistakes to Avoid

❌ **Using too many features**
- More features ≠ better predictions
- Overfitting risk increases
- ✅ Use feature importance to select top 5-10

❌ **Training on all available data**
- No way to test performance
- Metrics are overly optimistic
- ✅ Reserve 20% for testing

❌ **Ignoring residual diagnostics**
- May miss model problems
- Underlying patterns undetected
- ✅ Always check residual plots

❌ **Trusting predictions 6+ months out**
- Uncertainty too high
- Assumes patterns persist
- ✅ Use 30-60 day forecasts, retrain monthly

❌ **Not scaling/normalizing**
- Models like SVR perform badly
- Neural networks converge slowly
- ✅ Use StandardScaler (already done for you)

---

## 💾 Exporting & Using Results

### CSV Export Contents

**Forecast File**:
- Date: Future dates being predicted
- Previsão: Point forecast (most likely value)
- IC_Superior_95: Upper bound of 95% confidence interval
- IC_Inferior_95: Lower bound of 95% confidence interval

**Metrics File**:
- All model performance metrics
- Context percentages
- Useful for reporting

### Using in Spreadsheets

1. Import forecast CSV to Excel/Sheets
2. Create column with: `= FORECAST_VALUE + RAND()*(IC_SUP - IC_INF)`
3. Use for Monte Carlo simulations
4. Risk management applications

---

## 🔗 When to Retrain

- **Weekly**: If used for daily trading decisions
- **Monthly**: For medium-term forecasts  
- **Quarterly**: For long-term planning
- **When performance drops**: MAPE increases 30%+

### Signs to Retrain Now

- Last 10 residuals all positive/negative
- MAPE increases above threshold
- Market regime change (volatility spike)
- New data patterns emerging

---

## 📞 Questions?

- **Why is R² negative?** → Model worse than average
- **Should I use 50 lags?** → Usually 5-10 optimal
- **Can I predict 6 months out?** → Not reliably
- **Which model is best?** → Usually XGBoost or LightGBM
- **How much data do I need?** → At least 250-500 rows

Check the interpretation guides in the app for detailed explanations!
