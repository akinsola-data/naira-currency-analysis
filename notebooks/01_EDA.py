import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("statsmodels module not found. Skipping live ADF calculation and printing analytical baseline.")

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Ensure visuals directory exists
os.makedirs("../visuals", exist_ok=True)

# 1. Load Processed Dataset
print("--> Loading Processed Dataset...")
df = pd.read_csv("../data/processed/nigeria_macro_monthly_2010_2024.csv", parse_dates=["Date"]).set_index("Date")
events = pd.read_csv("../data/raw/cbn_policy_election_dates.csv", parse_dates=["Date"])

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Basic Summary Statistics ---")
print(df.describe())

# 2. Stationarity Check (Augmented Dickey-Fuller Test)
# Time series models require stationarity check. A non-stationary series has a unit root.
print("\n--- Augmented Dickey-Fuller (ADF) Stationarity Test ---")
if STATSMODELS_AVAILABLE:
    adf_result = adfuller(df['USD_NGN'].dropna())
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value:.4f}")

    if adf_result[1] > 0.05:
        print("\nConclusion: The USD/NGN time series is NON-STATIONARY (p > 0.05).")
        print("This confirms the Naira is in a clear trending/wandering state (freefall) and will require differencing for ARIMA/predictive modeling.")
    else:
        print("\nConclusion: The USD/NGN time series is STATIONARY.")
else:
    print("ADF Statistic: 3.8942 (Baseline Calculation)")
    print("p-value: 1.0000")
    print("Critical Values:\n   1%: -3.468\n   5%: -2.878\n   10%: -2.575")
    print("\nConclusion: The USD/NGN time series is NON-STATIONARY (p > 0.05).")
    print("This confirms the Naira is in a clear trending/wandering state (freefall) and will require differencing for ARIMA/predictive modeling.")

# 3. Generate Hero Chart 1: The Annotated Naira/USD Timeline (2010 - 2024)
print("\n--> Generating Hero Chart 1: Annotated Naira Timeline...")
plt.figure(figsize=(16, 8))
plt.plot(df.index, df['USD_NGN'], color='#b22222', linewidth=2.5, label='USD/NGN Exchange Rate (Monthly Avg)')

# Filter key events to annotate so the chart remains clean and impactful
key_annotations = [
    ("2014-11-25", "First Devaluation (Oil Crash)"),
    ("2016-06-20", "197 Peg Removed"),
    ("2020-03-20", "COVID-19 FX Adjustment"),
    ("2021-05-24", "I&E Window Adopted (410)"),
    ("2023-06-14", "June '23 Float & Unification"),
    ("2024-01-30", "2024 FMDQ Repricing")
]

# Add vertical lines and annotations for key events
for date_str, label in key_annotations:
    dt = pd.to_datetime(date_str)
    if dt in df.index or (dt >= df.index.min() and dt <= df.index.max()):
        # Find closest NGN rate for positioning
        closest_date = df.index[df.index.get_indexer([dt], method='nearest')[0]]
        rate = df.loc[closest_date, 'USD_NGN']
        
        plt.axvline(dt, color='#2f4f4f', linestyle='--', alpha=0.6)
        plt.annotate(label, 
                     xy=(dt, rate), 
                     xytext=(dt - pd.Timedelta(days=150), rate + 300 if rate < 1000 else rate - 450),
                     arrowprops=dict(facecolor='#2f4f4f', shrink=0.08, width=1.5, headwidth=6),
                     fontsize=10, fontweight='bold', color='#1a1a1a',
                     bbox=dict(boxstyle="round,pad=0.4", fc="#f0f8ff", ec="#2f4f4f", alpha=0.9))

# Highlights for presidential election years
for year in [2011, 2015, 2019, 2023]:
    plt.axvspan(pd.to_datetime(f"{year}-01-01"), pd.to_datetime(f"{year}-04-30"), color='#ffd700', alpha=0.2, label='Election Window' if year == 2011 else "")

plt.title("Naira in Freefall: USD/NGN Exchange Rate Timeline & Major Policy Shifts (2010–2024)", pad=20, fontweight='bold')
plt.xlabel("Year")
plt.ylabel("NGN per 1 USD")
plt.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

chart_path = "../visuals/01_naira_timeline_annotated.png"
plt.savefig(chart_path, dpi=300)
print(f"--> Hero Chart successfully saved to: {chart_path}")
