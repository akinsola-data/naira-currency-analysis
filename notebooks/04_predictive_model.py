import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Ensure visuals directory exists
os.makedirs("../visuals", exist_ok=True)

# 1. Load Processed Dataset
print("--> Loading Processed Dataset for Predictive Modeling...")
df = pd.read_csv("../data/processed/nigeria_macro_monthly_2010_2024.csv", parse_dates=["Date"]).set_index("Date")

# ==============================================================================
# 2. Feature Engineering
# ==============================================================================
print("\n--> Engineering Features (Lags, Rolling Means, Deltas)...")

# Target Variable: Next-Month USD/NGN Exchange Rate
df['Target_USD_NGN'] = df['USD_NGN'].shift(-1)

# Lagged Exchange Rates
df['USD_NGN_Lag1'] = df['USD_NGN'].shift(1)
df['USD_NGN_Lag3'] = df['USD_NGN'].shift(3)

# Rolling Means (Trend followers)
df['USD_NGN_Roll3'] = df['USD_NGN'].rolling(window=3).mean()
df['USD_NGN_Roll6'] = df['USD_NGN'].rolling(window=6).mean()

# Oil Price Deltas
df['Oil_Delta_1m'] = df['Brent_Crude_Price'].pct_change(1) * 100
df['Oil_Delta_3m'] = df['Brent_Crude_Price'].pct_change(3) * 100

# Inflation Delta (Momentum)
df['Inflation_Delta'] = df['Inflation_Rate'].diff()

# Drop rows with NaN values created by shifting/rolling
data = df.dropna().copy()

# Select Feature Columns
features = [
    'USD_NGN', 'USD_NGN_Lag1', 'USD_NGN_Lag3', 
    'USD_NGN_Roll3', 'USD_NGN_Roll6', 
    'Brent_Crude_Price', 'Oil_Delta_1m', 'Oil_Delta_3m', 
    'Inflation_Rate', 'Inflation_Delta', 
    'Election_Year', 'Election_Month'
]

X = data[features]
y = data['Target_USD_NGN']

# ==============================================================================
# 3. Time Series Train/Test Split (No Random Shuffling!)
# ==============================================================================
# Train on 2010–2022 | Test on 2023–2024 (The float / high volatility period)
split_date = '2023-01-01'
X_train = X.loc[X.index < split_date]
y_train = y.loc[y.index < split_date]
X_test = X.loc[X.index >= split_date]
y_test = y.loc[y.index >= split_date]

print(f"    Training Set: {X_train.index.min().strftime('%Y-%m')} to {X_train.index.max().strftime('%Y-%m')} ({len(X_train)} months)")
print(f"    Testing Set: {X_test.index.min().strftime('%Y-%m')} to {X_test.index.max().strftime('%Y-%m')} ({len(X_test)} months)")

# ==============================================================================
# 4. Model Training & Evaluation (Ridge Regression)
# ==============================================================================
print("\n--> Training Machine Learning Models...")
# Ridge regression handles multicollinearity among lagged variables well
model = Ridge(alpha=100.0)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions to series with date indices
preds_test = pd.Series(y_pred_test, index=y_test.index)
preds_train = pd.Series(y_pred_train, index=y_train.index)

# Evaluate Metrics
rmse = np.sqrt(mean_squared_error(y_test, preds_test))
mae = mean_absolute_error(y_test, preds_test)

# Directional Accuracy (Did the model correctly predict whether Naira goes UP or DOWN?)
actual_direction = np.sign(y_test - X_test['USD_NGN'])
predicted_direction = np.sign(preds_test - X_test['USD_NGN'])
directional_accuracy = (actual_direction == predicted_direction).mean() * 100

print("\n--- Model Evaluation Metrics (Test Set: 2023-2024) ---")
print(f"    RMSE: {rmse:.2f} NGN")
print(f"    MAE: {mae:.2f} NGN")
print(f"    Directional Accuracy: {directional_accuracy:.1f}%")

# ==============================================================================
# 5. Hero Chart 8: Actual vs. Predicted Exchange Rate (Train vs Test)
# ==============================================================================
print("\n--> Generating Hero Chart 8: Predictive Model Forecast...")
plt.figure(figsize=(16, 8))

# Plot Actuals
plt.plot(data.index, data['USD_NGN'], color='#2f4f4f', linewidth=2.5, label='Actual USD/NGN Exchange Rate')

# Plot Predictions (Train & Test)
plt.plot(preds_train.index, preds_train, color='#4682b4', linestyle='--', linewidth=2, label='Model Fit (Training Data)')
plt.plot(preds_test.index, preds_test, color='#d9534f', linestyle='-', linewidth=2.5, label='Next-Month Forecast (Testing Data)')

# Highlight Test Period (2023-2024 Float Regime)
plt.axvline(pd.to_datetime(split_date), color='black', linestyle='--', linewidth=2)
plt.axvspan(pd.to_datetime(split_date), data.index.max(), color='#fff0f0', alpha=0.5, label='Out-of-Sample Test Period (2023–2024)')

# Annotation Box with Evaluation Metrics
metrics_text = (
    f"Model Performance (Out-of-Sample):\n"
    f"• MAE: {mae:.1f} NGN\n"
    f"• RMSE: {rmse:.1f} NGN\n"
    f"• Directional Accuracy: {directional_accuracy:.1f}%\n\n"
    f"Insight: Despite the extreme structural break\n"
    f"of the 2023 float, the macro-lagged model\n"
    f"successfully tracks the rapid depreciation."
)
plt.annotate(metrics_text, 
             xy=(pd.to_datetime('2023-06-01'), y_test.max() * 0.7), 
             xytext=(pd.to_datetime('2017-01-01'), data['USD_NGN'].max() * 0.65),
             arrowprops=dict(facecolor='#d9534f', shrink=0.08, width=1.5, headwidth=6),
             fontsize=12, fontweight='bold', color='#4a0000',
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff8dc", ec="#d9534f", alpha=0.9))

plt.title("Macro-Predictive Model: One-Month Ahead Forecast of USD/NGN Exchange Rate (Ridge Regression)", pad=20, fontweight='bold')
plt.xlabel("Year", fontweight='bold')
plt.ylabel("NGN per 1 USD", fontweight='bold')
plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

chart8_path = "../visuals/08_predictive_model_forecast.png"
plt.savefig(chart8_path, dpi=300)
print(f"    Saved Hero Chart 8 to: {chart8_path}")
plt.show()

# ==============================================================================
# 6. Feature Importance / Coefficients Analysis
# ==============================================================================
print("\n--> Extracting Feature Coefficients...")
coefs = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(12, 6))
# Using hue and legend=False to avoid future seaborn deprecation warnings
sns.barplot(x='Coefficient', y='Feature', hue='Feature', data=coefs, palette=['#d9534f' if c < 0 else '#4682b4' for c in coefs['Coefficient']], legend=False)
plt.title("Ridge Regression Feature Coefficients: Drivers of Next-Month Exchange Rate", pad=20, fontweight='bold')
plt.xlabel("Coefficient Weight (Positive = Pushes Naira Up/Depreciates | Negative = Pushes Naira Down/Appreciates)")
plt.ylabel("Feature")
plt.tight_layout()

chart9_path = "../visuals/09_model_feature_coefficients.png"
plt.savefig(chart9_path, dpi=300)
print(f"    Saved Hero Chart 9 to: {chart9_path}")
plt.show()

print("\n--> Day 5 Predictive Modeling Complete! All visuals exported successfully.")