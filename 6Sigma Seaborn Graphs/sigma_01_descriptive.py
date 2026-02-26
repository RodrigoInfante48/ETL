"""
sigma_01_descriptive.py
========================
6 Sigma Analysis - Block 1: Descriptive Statistics & Distribution
Author: Rodrigo Infante
Dataset: World Bank Economic Indicators (2000-2023)
Run: python sigma_01_descriptive.py

Requires: pip install seaborn matplotlib pandas scipy sqlalchemy psycopg2-binary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── Load data directly from World Bank API (no DB needed) ──────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from extract import extract_all
    from transform import transform
    print("Loading data from World Bank API...")
    raw = extract_all()
    df = transform(raw)
    print(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns\n")
except Exception as e:
    print(f"API load failed ({e}), generating synthetic demo data...")
    np.random.seed(42)
    countries = ["Colombia","Brazil","Mexico","Argentina","Chile",
                 "Peru","Ecuador","United States","China","Germany"]
    years = list(range(2000, 2024))
    rows = []
    for c in countries:
        for y in years:
            rows.append({
                "country_name": c,
                "country_code": c[:2].upper(),
                "year": y,
                "gdp_current_usd": np.random.lognormal(25, 1.5),
                "gdp_per_capita_usd": np.random.lognormal(9, 1.2),
                "inflation_annual_pct": np.random.normal(4, 5),
                "unemployment_pct": np.random.normal(7, 3),
                "population_total": np.random.lognormal(16, 2),
            })
    df = pd.DataFrame(rows)

INDICATORS = {
    "gdp_per_capita_usd": "GDP per Capita (USD)",
    "inflation_annual_pct": "Inflation (%)",
    "unemployment_pct": "Unemployment (%)",
    "gdp_current_usd": "GDP Total (USD)",
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("Set2", 10)

# ── Figure 1: Distribution + Normality per indicator ──────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("6 Sigma — Block 1: Distribution & Normality Analysis", fontsize=16, fontweight="bold")

for col_idx, (col, label) in enumerate(INDICATORS.items()):
    data = df[col].dropna()

    # Histogram + KDE
    ax1 = axes[0, col_idx]
    sns.histplot(data, kde=True, ax=ax1, color=COLORS[col_idx], bins=30, edgecolor="white")
    mean_val = data.mean()
    std_val  = data.std()
    ax1.axvline(mean_val, color="red",    linestyle="--", linewidth=2, label=f"μ = {mean_val:,.2f}")
    ax1.axvline(mean_val + std_val, color="orange", linestyle=":", linewidth=1.5, label=f"+1σ")
    ax1.axvline(mean_val - std_val, color="orange", linestyle=":", linewidth=1.5, label=f"-1σ")
    ax1.axvline(mean_val + 3*std_val, color="darkred", linestyle="-.", linewidth=1.2, label=f"+3σ")
    ax1.axvline(mean_val - 3*std_val, color="darkred", linestyle="-.", linewidth=1.2, label=f"-3σ")
    ax1.set_title(label, fontsize=11, fontweight="bold")
    ax1.set_xlabel("")
    ax1.legend(fontsize=7)

    # Skewness & Kurtosis annotation
    skew = data.skew()
    kurt = data.kurtosis()
    _, pvalue = stats.shapiro(data.sample(min(len(data), 5000), random_state=42))
    normal_str = "Normal ✓" if pvalue > 0.05 else "Non-Normal ✗"
    ax1.set_xlabel(f"Skew={skew:.2f}  Kurt={kurt:.2f}  {normal_str}", fontsize=8)

    # Q-Q Plot
    ax2 = axes[1, col_idx]
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax2.plot(osm, osr, "o", color=COLORS[col_idx], markersize=2, alpha=0.5)
    ax2.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=2)
    ax2.set_title(f"Q-Q Plot — {label}", fontsize=10)
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.text(0.05, 0.92, f"R²={r**2:.3f}", transform=ax2.transAxes,
             fontsize=9, color="darkred", fontweight="bold")

plt.tight_layout()
plt.savefig("sigma_01_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_01_distribution.png")

# ── Figure 2: Summary Statistics Table (6 Sigma format) ───────────────────
fig2, ax = plt.subplots(figsize=(16, 5))
ax.axis("off")

summary_rows = []
for col, label in INDICATORS.items():
    d = df[col].dropna()
    _, p = stats.shapiro(d.sample(min(len(d), 5000), random_state=42))
    summary_rows.append([
        label,
        f"{len(d):,}",
        f"{d.mean():,.2f}",
        f"{d.median():,.2f}",
        f"{d.std():,.2f}",
        f"{d.min():,.2f}",
        f"{d.max():,.2f}",
        f"{d.skew():.3f}",
        f"{d.kurtosis():.3f}",
        f"{p:.4f}",
        "Normal" if p > 0.05 else "Non-Normal",
    ])

cols = ["Indicator", "N", "Mean", "Median", "Std Dev", "Min", "Max",
        "Skewness", "Kurtosis", "Shapiro p", "Distribution"]

table = ax.table(cellText=summary_rows, colLabels=cols,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# Style header
for j in range(len(cols)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Color-code last column
for i, row in enumerate(summary_rows, start=1):
    color = "#d5f5e3" if row[-1] == "Normal" else "#fadbd8"
    table[i, -1].set_facecolor(color)

ax.set_title("6 Sigma Descriptive Statistics Summary", fontsize=14,
             fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("sigma_01_stats_table.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_01_stats_table.png")
print("\nBlock 1 complete.")
