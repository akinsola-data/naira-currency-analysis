import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("statsmodels not found in local sandbox. Using baseline calculations.")

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Ensure visuals directory exists
os.makedirs("../visuals", exist_ok=True)

# 1. Load Processed Dataset
print("--> Loading Processed Dataset for Time Series Decomposition...")
df = pd.read_csv("../data/processed/nigeria_macro_monthly_2010_2024.csv", parse_dates=["Date"]).set_index("Date")

# ==============================================================================
# 2. Time Series Decomposition (Trend + Seasonality + Residuals)
# ==============================================================================
print("\n--> Generating Hero Chart 5: Time Series Decomposition...")
if STATSMODELS_AVAILABLE:
    # Using multiplicative decomposition as currency fluctuations scale with the level
    decomp = seasonal_decompose(df['USD_NGN'], model='multiplicative', period=12)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    ax1.plot(df.index, df['USD_NGN'], color='#b22222', linewidth=2.5)
    ax1.set_ylabel('Observed (NGN/USD)', fontweight='bold')
    ax1.set_title('USD/NGN Exchange Rate — Multiplicative Time Series Decomposition (2010–2024)', fontweight='bold', pad=15)
    
    ax2.plot(df.index, decomp.trend, color='#2f4f4f', linewidth=2.5)
    ax2.set_ylabel('Trend', fontweight='bold')
    
    ax3.plot(df.index, decomp.seasonal, color='#4682b4', linewidth=2)
    ax3.set_ylabel('Seasonality', fontweight='bold')
    
    ax4.scatter(df.index, decomp.resid, color='#d9534f', s=15)
    ax4.axhline(1, color='black', linestyle='--', linewidth=1)
    ax4.set_ylabel('Residuals', fontweight='bold')
    ax4.set_xlabel('Year', fontweight='bold')
    
    plt.tight_layout()
    chart5_path = "../visuals/05_time_series_decomposition.png"
    plt.savefig(chart5_path, dpi=300)
    print(f"    Saved Hero Chart 5 to: {chart5_path}")
    plt.show()
else:
    # Baseline fallback plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df.index, df['USD_NGN'], color='#b22222', linewidth=2.5, label='Observed')
    ax.plot(df.index, df['USD_NGN'].rolling(12).mean(), color='#2f4f4f', linewidth=2.5, linestyle='--', label='12-Month Trend')
    plt.title('USD/NGN Exchange Rate — Observed vs 12-Month Moving Trend (2010–2024)', fontweight='bold', pad=15)
    plt.legend()
    plt.tight_layout()
    chart5_path = "../visuals/05_time_series_decomposition.png"
    plt.savefig(chart5_path, dpi=300)
    print(f"    Saved Hero Chart 5 to: {chart5_path}")
    plt.close()

# ==============================================================================
# 3. Volatility Clustering & Structural Break Analysis (The 2023 Float)
# ==============================================================================
print("\n--> Generating Hero Chart 6: Volatility Clustering & Structural Break...")
# Calculate monthly returns and 6-month rolling standard deviation (volatility)
df['Monthly_Return'] = df['USD_NGN'].pct_change() * 100
df['Rolling_Volatility'] = df['Monthly_Return'].rolling(window=6).std()

fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(df.index, df['Rolling_Volatility'], color='#d9534f', linewidth=2.5, label='6-Month Rolling Volatility (Std Dev of % Change)')

# Highlight the June 2023 Float Structural Break
float_date = pd.to_datetime('2023-06-14')
ax.axvline(float_date, color='#1a1a1a', linestyle='--', linewidth=2)
ax.axvspan(float_date, df.index.max(), color='#ffcccc', alpha=0.3, label='Post-Float Regime (Structural Break)')

# Annotation explaining Volatility Clustering & Structural Break
ax.annotate("Structural Break: June 2023 Float\nTransition from suppressed crawling peg\nto high-volatility floating regime.\nNotice extreme volatility clustering post-2023.", 
             xy=(float_date, df['Rolling_Volatility'].loc[float_date:].max() * 0.8), 
             xytext=(pd.to_datetime('2018-01-01'), df['Rolling_Volatility'].max() * 0.7),
             arrowprops=dict(facecolor='#1a1a1a', shrink=0.08, width=1.5, headwidth=6),
             fontsize=12, fontweight='bold', color='#4a0000',
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff0f0", ec="#d9534f", alpha=0.9))

plt.title("Volatility Clustering: Was the June 2023 Currency Float a Structural Break?", pad=20, fontweight='bold')
plt.xlabel("Year", fontweight='bold')
plt.ylabel("Rolling Volatility (%)", fontweight='bold')
plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

chart6_path = "../visuals/06_volatility_clustering_2023_float.png"
plt.savefig(chart6_path, dpi=300)
print(f"    Saved Hero Chart 6 to: {chart6_path}")
plt.show()

# ==============================================================================
# 4. Rolling Correlation Windows (Oil vs. Naira)
# ==============================================================================
print("\n--> Generating Hero Chart 7: Rolling Correlation Windows...")
# Calculate 12-month rolling correlation between Oil returns and Naira returns
df['Oil_Return'] = df['Brent_Crude_Price'].pct_change() * 100
df['Rolling_Corr_Oil_Naira'] = df['Monthly_Return'].rolling(window=12).corr(df['Oil_Return'])

fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(df.index, df['Rolling_Corr_Oil_Naira'], color='#2f4f4f', linewidth=2.5, label='12-Month Rolling Correlation (USD/NGN vs Brent Crude)')
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Highlight regime shift where correlation flips from negative (normal petro-state) to positive/erratic
ax.axvspan(pd.to_datetime('2021-01-01'), df.index.max(), color='#e6f2ff', alpha=0.5, label='Decoupling Window (2021–2024)')

ax.annotate("Regime Breakdown (2021–2024)\nHistorically, correlation was frequently negative\n(Higher oil = stronger currency).\nPost-2021, correlation completely breaks down\nand swings wildly.", 
             xy=(pd.to_datetime('2022-06-01'), 0.3), 
             xytext=(pd.to_datetime('2015-01-01'), 0.5),
             arrowprops=dict(facecolor='#2f4f4f', shrink=0.08, width=1.5, headwidth=6),
             fontsize=12, fontweight='bold', color='#1a1a1a',
             bbox=dict(boxstyle="round,pad=0.5", fc="#f0f8ff", ec="#2f4f4f", alpha=0.9))

plt.title("Rolling Correlation Breakdown: USD/NGN vs. Brent Crude Oil (12-Month Windows)", pad=20, fontweight='bold')
plt.xlabel("Year", fontweight='bold')
plt.ylabel("Rolling Correlation Coefficient (r)", fontweight='bold')
plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

chart7_path = "../visuals/07_rolling_correlation_oil_naira.png"
plt.savefig(chart7_path, dpi=300)
print(f"    Saved Hero Chart 7 to: {chart7_path}")
plt.show()

print("\n--> Day 4 Time Series Decomposition Complete! All visuals exported successfully.")