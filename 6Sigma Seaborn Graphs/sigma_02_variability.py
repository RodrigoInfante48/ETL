"""
sigma_02_variability.py
========================
6 Sigma Analysis - Block 2: Variability, Boxplots & Outlier Detection
Author: Rodrigo Infante
Dataset: World Bank Economic Indicators (2000-2023)
Run: python sigma_02_variability.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from extract import extract_all
    from transform import transform
    raw = extract_all()
    df = transform(raw)
except Exception:
    np.random.seed(42)
    countries = ["Colombia","Brazil","Mexico","Argentina","Chile",
                 "Peru","Ecuador","United States","China","Germany"]
    years = list(range(2000, 2024))
    rows = []
    for c in countries:
        for y in years:
            rows.append({
                "country_name": c, "country_code": c[:2].upper(), "year": y,
                "gdp_per_capita_usd": np.random.lognormal(9, 1.2),
                "inflation_annual_pct": np.random.normal(4, 5),
                "unemployment_pct": np.random.normal(7, 3),
                "gdp_current_usd": np.random.lognormal(25, 1.5),
            })
    df = pd.DataFrame(rows)

INDICATORS = {
    "gdp_per_capita_usd": "GDP per Capita (USD)",
    "inflation_annual_pct": "Inflation (%)",
    "unemployment_pct": "Unemployment (%)",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = "Set2"

# ── Figure 1: Boxplots by Country ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle("6 Sigma — Block 2: Variability & Spread by Country", fontsize=15, fontweight="bold")

for ax, (col, label) in zip(axes, INDICATORS.items()):
    order = (df.groupby("country_name")[col].median()
               .sort_values(ascending=True).index.tolist())
    sns.boxplot(data=df, x=col, y="country_name", order=order,
                palette=PALETTE, ax=ax, flierprops=dict(marker="o", color="red",
                markersize=4, alpha=0.6))
    sns.stripplot(data=df, x=col, y="country_name", order=order,
                  color="black", size=2, alpha=0.3, jitter=True, ax=ax)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("")
    ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.5, label="Global Mean")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("sigma_02_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_02_boxplots.png")

# ── Figure 2: Violin Plots ─────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(22, 8))
fig2.suptitle("6 Sigma — Block 2: Density Distribution per Country (Violin)", fontsize=15, fontweight="bold")

for ax, (col, label) in zip(axes2, INDICATORS.items()):
    order = (df.groupby("country_name")[col].median()
               .sort_values(ascending=True).index.tolist())
    sns.violinplot(data=df, x="country_name", y=col, order=order,
                   palette=PALETTE, ax=ax, inner="quartile", cut=0)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("")
    ax.axhline(df[col].mean(), color="red", linestyle="--", linewidth=1.5, label="Global Mean")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("sigma_02_violin.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_02_violin.png")

# ── Figure 3: IQR Outlier Detection (6 Sigma method) ──────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(22, 7))
fig3.suptitle("6 Sigma — Block 2: Outlier Detection (IQR + 3σ Rule)", fontsize=15, fontweight="bold")

for ax, (col, label) in zip(axes3, INDICATORS.items()):
    data = df[["country_name", "year", col]].dropna().copy()
    mean = data[col].mean()
    std  = data[col].std()
    q1   = data[col].quantile(0.25)
    q3   = data[col].quantile(0.75)
    iqr  = q3 - q1

    # IQR outliers
    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr
    # 3-Sigma outliers
    lower_3s  = mean - 3 * std
    upper_3s  = mean + 3 * std

    data["outlier_iqr"]   = (data[col] < lower_iqr) | (data[col] > upper_iqr)
    data["outlier_3sigma"] = (data[col] < lower_3s) | (data[col] > upper_3s)

    normal = data[~data["outlier_iqr"]]
    out_iqr = data[data["outlier_iqr"]]
    out_3s  = data[data["outlier_3sigma"]]

    ax.scatter(normal["year"], normal[col], color="steelblue", alpha=0.5,
               s=20, label="Normal")
    ax.scatter(out_iqr["year"], out_iqr[col], color="orange", alpha=0.8,
               s=50, marker="D", label=f"IQR Outlier (n={len(out_iqr)})")
    ax.scatter(out_3s["year"], out_3s[col], color="red", alpha=0.9,
               s=80, marker="X", label=f"3σ Outlier (n={len(out_3s)})")

    ax.axhline(mean,      color="blue",   linestyle="--", linewidth=1, label=f"μ={mean:.2f}")
    ax.axhline(upper_3s,  color="red",    linestyle=":",  linewidth=1, label=f"+3σ={upper_3s:.2f}")
    ax.axhline(lower_3s,  color="red",    linestyle=":",  linewidth=1, label=f"-3σ={lower_3s:.2f}")
    ax.axhline(upper_iqr, color="orange", linestyle="-.", linewidth=1)
    ax.axhline(lower_iqr, color="orange", linestyle="-.", linewidth=1)

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.legend(fontsize=7)

    pct_out = len(out_iqr) / len(data) * 100
    ax.text(0.02, 0.02, f"Outlier rate: {pct_out:.1f}%", transform=ax.transAxes,
            fontsize=9, color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig("sigma_02_outliers.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_02_outliers.png")

# ── Figure 4: CV (Coefficient of Variation) — Variability ranking ─────────
fig4, ax4 = plt.subplots(figsize=(14, 6))
fig4.suptitle("6 Sigma — Block 2: Coefficient of Variation by Country & Indicator", fontsize=14, fontweight="bold")

cv_data = []
for col, label in INDICATORS.items():
    for country in df["country_name"].unique():
        sub = df[df["country_name"] == country][col].dropna()
        if len(sub) > 2 and sub.mean() != 0:
            cv = (sub.std() / abs(sub.mean())) * 100
            cv_data.append({"Country": country, "Indicator": label, "CV (%)": cv})

cv_df = pd.DataFrame(cv_data)
pivot_cv = cv_df.pivot(index="Country", columns="Indicator", values="CV (%)").fillna(0)

sns.heatmap(pivot_cv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax4,
            linewidths=0.5, cbar_kws={"label": "CV (%)"})
ax4.set_title("Coefficient of Variation — Higher = More Variability (Less Stable)", fontsize=11)
ax4.set_xlabel("")
ax4.set_ylabel("")

plt.tight_layout()
plt.savefig("sigma_02_cv_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_02_cv_heatmap.png")
print("\nBlock 2 complete.")
