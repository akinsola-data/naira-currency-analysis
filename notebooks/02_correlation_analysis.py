import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Ensure visuals directory exists
os.makedirs("../visuals", exist_ok=True)

# 1. Load Processed Dataset
print("--> Loading Processed Dataset for Correlation Deep Dive...")
df = pd.read_csv("../data/processed/nigeria_macro_monthly_2010_2024.csv", parse_dates=["Date"]).set_index("Date")

# Calculate Monthly Percentage Changes (Delta / Returns)
df['USD_NGN_Change'] = df['USD_NGN'].pct_change() * 100
df['Oil_Change'] = df['Brent_Crude_Price'].pct_change() * 100
df['Inflation_Change'] = df['Inflation_Rate'].diff() # Monthly basis point change in inflation

print("\n--- Correlation Matrix (Levels) ---")
print(df[['USD_NGN', 'Brent_Crude_Price', 'Inflation_Rate']].corr())

print("\n--- Correlation Matrix (Percentage Changes / Deltas) ---")
print(df[['USD_NGN_Change', 'Oil_Change', 'Inflation_Change']].corr())

# ==============================================================================
# 2. Hero Chart 2: Brent Crude vs. Naira — The Great Decoupling (Dual Axis Plot)
# ==============================================================================
print("\n--> Generating Hero Chart 2: Brent Crude vs. Naira (The Decoupling)...")
fig, ax1 = plt.subplots(figsize=(16, 8))

color = '#b22222'
ax1.set_xlabel('Year', fontweight='bold', labelpad=15)
ax1.set_ylabel('NGN per 1 USD', color=color, fontweight='bold')
line1 = ax1.plot(df.index, df['USD_NGN'], color=color, linewidth=2.5, label='USD/NGN Exchange Rate (LHS)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(False) # Turn off grid for LHS to avoid clutter in dual axis

ax2 = ax1.twinx()  
color = '#2f4f4f'
ax2.set_ylabel('Brent Crude Price (USD / Barrel)', color=color, fontweight='bold')
line2 = ax2.plot(df.index, df['Brent_Crude_Price'], color=color, linewidth=2, linestyle='-', label='Brent Crude Oil Price (RHS)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.grid(True, linestyle=':', alpha=0.6)

# Highlight The Great Decoupling (2021-2024)
decoupling_start = pd.to_datetime('2021-01-01')
ax1.axvspan(decoupling_start, df.index.max(), color='#ff9999', alpha=0.15)
ax1.text(pd.to_datetime('2022-06-01'), df['USD_NGN'].max()*0.6, 
         "The Great Decoupling (2021–2024):\nOil prices recovered to $80–$90+,\nbut Naira continued to collapse due to\nFX backlog, low net reserves & monetary expansion.", 
         fontsize=12, fontweight='bold', color='#4a0000',
         bbox=dict(boxstyle="round,pad=0.5", fc="#fff0f0", ec="#b22222", alpha=0.9))

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

plt.title("The Great Decoupling: Brent Crude Oil Price vs. USD/NGN Exchange Rate (2010–2024)", pad=20, fontweight='bold')
plt.tight_layout()
chart2_path = "../visuals/02_brent_vs_naira_decoupling.png"
plt.savefig(chart2_path, dpi=300)
print(f"    Saved Hero Chart 2 to: {chart2_path}")
plt.show()

# ==============================================================================
# 3. Inflation Rate vs. Exchange Rate Lag Analysis
# ==============================================================================
print("\n--> Generating Hero Chart 3: Inflation vs. USD/NGN Lag Analysis...")
# We check how past exchange rate movements correlate with future inflation
lags = range(-6, 7) # -6 months to +6 months
corrs = [df['USD_NGN_Change'].corr(df['Inflation_Change'].shift(lag)) for lag in lags]

plt.figure(figsize=(12, 6))
bars = plt.bar(lags, corrs, color=['#2f4f4f' if lag < 0 else '#b22222' if lag > 0 else '#4682b4' for lag in lags], alpha=0.85)
plt.axhline(0, color='black', linewidth=1)

# Annotations for interpretation
plt.title("Lag Correlation: Exchange Rate Depreciation vs. Inflation Rate Spikes", pad=20, fontweight='bold')
plt.xlabel("Lag (Months) -> [Negative: FX Leads Inflation | Positive: Inflation Leads FX]")
plt.ylabel("Correlation Coefficient (r)")

# Highlight key lag
max_corr_lag = lags[np.argmax(corrs)]
plt.annotate(f"Peak Impact at Lag {max_corr_lag} Months\n(FX depreciation rapidly passes\nthrough to consumer prices)", 
             xy=(max_corr_lag, max(corrs)), 
             xytext=(max_corr_lag - 3, max(corrs) + 0.05),
             arrowprops=dict(facecolor='#b22222', shrink=0.08, width=1.5, headwidth=6),
             fontsize=11, fontweight='bold', color='#1a1a1a',
             bbox=dict(boxstyle="round,pad=0.4", fc="#f0f8ff", ec="#2f4f4f", alpha=0.9))

plt.ylim(min(corrs) - 0.1, max(corrs) + 0.15)
plt.tight_layout()
chart3_path = "../visuals/03_inflation_fx_lag_analysis.png"
plt.savefig(chart3_path, dpi=300)
print(f"    Saved Hero Chart 3 to: {chart3_path}")
plt.show()

# ==============================================================================
# 4. Election Year Impact Analysis
# ==============================================================================
print("\n--> Generating Hero Chart 4: Election Year Currency Pressure...")
plt.figure(figsize=(10, 6))
# Compare monthly depreciation in Election Years vs Non-Election Years
sns.boxplot(x='Election_Year', y='USD_NGN_Change', hue='Election_Year', data=df, palette=['#4682b4', '#ffd700'], width=0.5, fliersize=5, legend=False)

plt.title("Currency Pressure: Monthly Naira Depreciation in Election vs. Non-Election Years", pad=20, fontweight='bold')
plt.xlabel("Election Year (0 = Non-Election Year | 1 = Election Year)")
plt.ylabel("Monthly USD/NGN % Change (Depreciation)")
plt.xticks([0, 1], ['Non-Election Years', 'Election Years (2011, 2015, 2019, 2023)'], fontweight='bold')

# Calculate means for annotation
mean_non_election = df[df['Election_Year'] == 0]['USD_NGN_Change'].mean()
mean_election = df[df['Election_Year'] == 1]['USD_NGN_Change'].mean()

plt.annotate(f"Mean Monthly Depreciation:\nNon-Election: {mean_non_election:.2f}%\nElection Years: {mean_election:.2f}%",
             xy=(1, mean_election),
             xytext=(0.6, df['USD_NGN_Change'].max() * 0.7),
             fontsize=11, fontweight='bold', color='#1a1a1a',
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff8dc", ec="#ffd700", alpha=0.9))

plt.tight_layout()
chart4_path = "../visuals/04_election_year_currency_pressure.png"
plt.savefig(chart4_path, dpi=300)
print(f"    Saved Hero Chart 4 to: {chart4_path}")
plt.show()

print("\n--> Day 3 Correlation Deep Dive Complete! All visuals exported successfully.")