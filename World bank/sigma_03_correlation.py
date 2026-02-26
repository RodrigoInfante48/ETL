"""
sigma_03_correlation.py  —  6 Sigma Block 3: Correlation & Regression
Run: python sigma_03_correlation.py
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
except Exception as e:
    print(f"API unavailable ({e}). Using synthetic demo data...\n")
    np.random.seed(42)
    countries = ["Colombia","Brazil","Mexico","Argentina","Chile",
                 "Peru","Ecuador","United States","China","Germany"]
    years = list(range(2000, 2024))
    rows = []
    for c in countries:
        base_gdp = np.random.lognormal(9, 1)
        for y in years:
            gdp = base_gdp * (1 + np.random.normal(0.02, 0.05))
            base_gdp = gdp
            rows.append({
                "country_name": c, "country_code": c[:2].upper(), "year": y,
                "gdp_per_capita_usd": gdp,
                "inflation_annual_pct": max(-5, np.random.normal(4,5) - gdp/50000),
                "unemployment_pct": max(0, np.random.normal(7,3) - gdp/100000),
                "gdp_current_usd": gdp * np.random.lognormal(16, 0.5),
                "population_total": np.random.lognormal(16, 2),
            })
    df = pd.DataFrame(rows)

NUMERIC_COLS = ["gdp_per_capita_usd","gdp_current_usd",
                "inflation_annual_pct","unemployment_pct","population_total"]
LABELS = {"gdp_per_capita_usd":"GDP/Capita","gdp_current_usd":"GDP Total",
          "inflation_annual_pct":"Inflation %","unemployment_pct":"Unemp. %",
          "population_total":"Population"}

df_num = df[NUMERIC_COLS].dropna()
df_log = df_num.copy()
for col in ["gdp_per_capita_usd","gdp_current_usd","population_total"]:
    df_log[col] = np.log1p(df_log[col].clip(lower=0))

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── Figure 1: Correlation Heatmap (raw + log) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("6 Sigma — Block 3: Correlation Matrix", fontsize=15, fontweight="bold")
for ax, data, title in zip(axes, [df_num, df_log],
                           ["Pearson Correlation (Raw)", "Pearson Correlation (Log)"]):
    corr = data.rename(columns=LABELS).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("sigma_03_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_03_correlation.png")

# ── Figure 2: Pair Plot ────────────────────────────────────────────────────
pair_cols = ["gdp_per_capita_usd","inflation_annual_pct","unemployment_pct"]
df_pair = df[pair_cols + ["country_name"]].dropna().copy()
palette_map = dict(zip(df_pair["country_name"].unique(),
                       sns.color_palette("tab10", df_pair["country_name"].nunique())))
g = sns.PairGrid(df_pair, vars=pair_cols, hue="country_name", palette=palette_map)
g.map_upper(sns.scatterplot, alpha=0.4, s=15)
g.map_lower(sns.kdeplot, alpha=0.5, fill=True)
g.map_diag(sns.histplot, kde=True, alpha=0.5)
g.add_legend(title="Country", bbox_to_anchor=(1.02, 0.5), loc="center left")
g.figure.suptitle("6 Sigma — Block 3: Pair Plot", fontsize=13, fontweight="bold", y=1.01)
plt.savefig("sigma_03_pairplot.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved: sigma_03_pairplot.png")

# ── Figure 3: Regression Scatter ──────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7))
fig3.suptitle("6 Sigma — Block 3: Regression Analysis", fontsize=15, fontweight="bold")
pairs = [
    ("gdp_per_capita_usd","unemployment_pct","GDP per Capita vs Unemployment"),
    ("gdp_per_capita_usd","inflation_annual_pct","GDP per Capita vs Inflation"),
]
for ax, (xcol, ycol, title) in zip(axes3, pairs):
    data = df[[xcol, ycol, "country_name"]].dropna()
    sns.scatterplot(data=data, x=xcol, y=ycol, hue="country_name", alpha=0.5, s=30, ax=ax)
    sns.regplot(data=data, x=xcol, y=ycol, scatter=False, ax=ax,
                color="red", line_kws={"linewidth": 2.5})
    slope, intercept, r, p, se = stats.linregress(data[xcol].values, data[ycol].values)
    p_str = f"p={p:.4f}" if p >= 0.0001 else "p<0.0001"
    ax.text(0.03, 0.93,
            f"R²={r**2:.3f}   r={r:.3f}   {p_str}\ny={slope:.4f}x+{intercept:.2f}",
            transform=ax.transAxes, fontsize=9, color="darkred", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("GDP per Capita (USD)")
    ax.set_ylabel(ycol.replace("_"," ").title())
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, title="Country", fontsize=7,
              bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("sigma_03_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_03_regression.png")

# ── Figure 4: p-value Significance Matrix ─────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 8))
fig4.suptitle("6 Sigma — Block 3: Correlation p-value Significance Matrix",
              fontsize=13, fontweight="bold")
cols_to_test = NUMERIC_COLS
labels_list  = [LABELS[c] for c in cols_to_test]
n = len(cols_to_test)
pval_matrix = np.ones((n, n))
corr_matrix = np.zeros((n, n))
data_clean  = df[cols_to_test].dropna()
for i in range(n):
    for j in range(n):
        if i != j:
            r, p = stats.pearsonr(data_clean.iloc[:, i], data_clean.iloc[:, j])
            pval_matrix[i, j] = p
            corr_matrix[i, j] = r
corr_df = pd.DataFrame(corr_matrix, index=labels_list, columns=labels_list)
pval_df  = pd.DataFrame(pval_matrix, index=labels_list, columns=labels_list)
annot = pd.DataFrame("", index=labels_list, columns=labels_list)
for i in range(n):
    for j in range(n):
        if i == j:
            annot.iloc[i, j] = "—"
        else:
            r   = corr_df.iloc[i, j]
            p   = pval_df.iloc[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            annot.iloc[i, j] = f"{r:.2f}{sig}"
sns.heatmap(corr_df, annot=annot, fmt="", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax4, linewidths=0.5,
            cbar_kws={"label": "Pearson r"})
ax4.set_title("* p<0.05   ** p<0.01   *** p<0.001", fontsize=10)
plt.tight_layout()
plt.savefig("sigma_03_significance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_03_significance.png")
print("\nBlock 3 complete.")
