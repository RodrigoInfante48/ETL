"""
sigma_05_capability_dashboard.py
==================================
6 Sigma Analysis - Block 5: Process Capability (Cp, Cpk) + Executive Dashboard
Author: Rodrigo Infante
Dataset: World Bank Economic Indicators (2000-2023)
Run: python sigma_05_capability_dashboard.py

Charts produced:
  Fig 1 — Process Capability (Cp / Cpk) for each indicator
  Fig 2 — Sigma Level Scorecard (DPMO proxy per country)
  Fig 3 — Year-over-Year Delta heatmap (% change)
  Fig 4 — Executive Summary Dashboard (9-panel)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Data Load ──────────────────────────────────────────────────────────────
try:
    from extract import extract_all
    from transform import transform
    print("Loading data from World Bank API...")
    raw = extract_all()
    df = transform(raw)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns\n")
except Exception as e:
    print(f"API unavailable ({e}). Using synthetic demo data...\n")
    np.random.seed(42)
    countries = ["Colombia", "Brazil", "Mexico", "Argentina", "Chile",
                 "Peru", "Ecuador", "United States", "China", "Germany"]
    years = list(range(2000, 2024))
    rows = []
    for c in countries:
        base_gdp = np.random.lognormal(9, 1)
        for y in years:
            base_gdp *= (1 + np.random.normal(0.025, 0.04))
            rows.append({
                "country_name": c, "country_code": c[:2].upper(), "year": y,
                "gdp_per_capita_usd": base_gdp,
                "inflation_annual_pct": np.random.normal(4, 5),
                "unemployment_pct": np.random.normal(7, 3),
                "gdp_current_usd": base_gdp * np.random.lognormal(16, 0.3),
                "population_total": np.random.lognormal(16, 2),
            })
    df = pd.DataFrame(rows)

sns.set_theme(style="whitegrid", font_scale=1.05)

INDICATORS = {
    "gdp_per_capita_usd": "GDP per Capita (USD)",
    "inflation_annual_pct": "Inflation (%)",
    "unemployment_pct": "Unemployment (%)",
}

# ── Specification limits (macro-economic policy targets / benchmarks) ──────
# These serve as the "customer specification" in the 6 Sigma framework.
# Adjust based on your own criteria / policy targets.
SPEC_LIMITS = {
    "gdp_per_capita_usd": {
        "LSL": 1_000,        # minimum acceptable GDP/capita
        "USL": 80_000,       # upper benchmark (advanced economy threshold)
        "target": 15_000,    # LATAM aspirational target
    },
    "inflation_annual_pct": {
        "LSL": -2.0,         # deflation threshold
        "USL": 10.0,         # central bank upper bound (aggressive)
        "target": 3.0,       # ideal stable inflation
    },
    "unemployment_pct": {
        "LSL": 0.0,          # floor
        "USL": 15.0,         # structural unemployment ceiling
        "target": 5.0,       # full-employment target
    },
}

D2 = 1.128   # d2 constant for n=2 (I-MR)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: compute Cp, Cpk, sigma level, DPMO
# ══════════════════════════════════════════════════════════════════════════════
def capability_metrics(data, lsl, usl, target):
    """Returns dict with Cp, Cpk, Sigma Level, DPMO, mean, std."""
    mu  = data.mean()
    sd  = data.std(ddof=1)
    if sd == 0:
        return None
    cp  = (usl - lsl) / (6 * sd)
    cpu = (usl - mu)  / (3 * sd)
    cpl = (mu  - lsl) / (3 * sd)
    cpk = min(cpu, cpl)
    # Sigma level ≈ Cpk * 3  (short-term; add 1.5 shift for long-term)
    sigma_level = cpk * 3
    # DPMO estimate via normal distribution (process outside spec)
    p_above = 1 - stats.norm.cdf((usl - mu) / sd)
    p_below = stats.norm.cdf((lsl - mu) / sd)
    dpmo = (p_above + p_below) * 1_000_000
    return {
        "mean": mu, "std": sd,
        "Cp": cp, "Cpk": cpk, "CPU": cpu, "CPL": cpl,
        "sigma_level": sigma_level, "DPMO": dpmo,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Process Capability per indicator (global data)
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, len(INDICATORS), figsize=(22, 8))
fig1.suptitle("6 Sigma — Block 5: Process Capability Analysis (Cp / Cpk)",
              fontsize=15, fontweight="bold")

capability_summary = []   # for use in Fig 4

for ax, (col, label) in zip(axes1, INDICATORS.items()):
    sl = SPEC_LIMITS[col]
    data = df[col].dropna()
    cm   = capability_metrics(data, sl["LSL"], sl["USL"], sl["target"])
    if cm is None:
        ax.text(0.5, 0.5, "Insufficient data", ha="center")
        continue

    # Distribution curve
    x_min = min(data.min(), sl["LSL"] - abs(sl["LSL"]) * 0.1)
    x_max = max(data.max(), sl["USL"] + abs(sl["USL"]) * 0.1)
    x_range = np.linspace(x_min, x_max, 400)
    pdf_vals = stats.norm.pdf(x_range, cm["mean"], cm["std"])

    ax.plot(x_range, pdf_vals, "steelblue", linewidth=2.5, label="Process dist.")
    ax.fill_between(x_range, pdf_vals,
                    where=(x_range >= sl["LSL"]) & (x_range <= sl["USL"]),
                    color="green", alpha=0.25, label="Within spec")
    ax.fill_between(x_range, pdf_vals,
                    where=(x_range < sl["LSL"]),
                    color="red", alpha=0.35, label="Below LSL")
    ax.fill_between(x_range, pdf_vals,
                    where=(x_range > sl["USL"]),
                    color="red", alpha=0.35, label="Above USL")

    ymax = pdf_vals.max()
    ax.axvline(sl["LSL"],      color="red",    linestyle="--", linewidth=2,
               label=f"LSL = {sl['LSL']:,.0f}")
    ax.axvline(sl["USL"],      color="red",    linestyle="--", linewidth=2,
               label=f"USL = {sl['USL']:,.0f}")
    ax.axvline(sl["target"],   color="purple", linestyle="-",  linewidth=1.8,
               label=f"Target = {sl['target']:,.0f}")
    ax.axvline(cm["mean"],     color="navy",   linestyle="-",  linewidth=2,
               label=f"μ = {cm['mean']:,.2f}")
    ax.axvline(cm["mean"] + 3*cm["std"], color="orange", linestyle=":",
               linewidth=1.5, label="±3σ")
    ax.axvline(cm["mean"] - 3*cm["std"], color="orange", linestyle=":", linewidth=1.5)

    # Metrics annotation box
    color_cpk = "green" if cm["Cpk"] >= 1.33 else ("orange" if cm["Cpk"] >= 1.0 else "red")
    info = (
        f"Cp   = {cm['Cp']:.3f}\n"
        f"Cpk  = {cm['Cpk']:.3f}\n"
        f"σ level = {cm['sigma_level']:.2f}\n"
        f"DPMO = {cm['DPMO']:,.0f}\n"
        f"μ = {cm['mean']:,.2f}\n"
        f"σ = {cm['std']:,.2f}"
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=9,
            va="top", color=color_cpk, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=color_cpk, alpha=0.9))

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=7, loc="upper right")

    capability_summary.append({
        "Indicator": label, "Cp": cm["Cp"], "Cpk": cm["Cpk"],
        "Sigma Level": cm["sigma_level"], "DPMO": cm["DPMO"],
        "Mean": cm["mean"], "Std": cm["std"],
    })

plt.tight_layout()
plt.savefig("sigma_05_capability.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_05_capability.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Sigma Level Scorecard per Country (color-coded)
# ══════════════════════════════════════════════════════════════════════════════
sigma_rows = []
for country in df["country_name"].unique():
    sub = df[df["country_name"] == country]
    row = {"Country": country}
    for col, label in INDICATORS.items():
        sl = SPEC_LIMITS[col]
        data = sub[col].dropna()
        if len(data) < 5:
            row[label] = np.nan
            continue
        cm = capability_metrics(data, sl["LSL"], sl["USL"], sl["target"])
        row[label] = round(cm["sigma_level"], 2) if cm else np.nan
    sigma_rows.append(row)

sigma_df = pd.DataFrame(sigma_rows).set_index("Country")

# Cap display to ±6 for readability
sigma_display = sigma_df.clip(-1, 6)

fig2, ax2 = plt.subplots(figsize=(14, 7))
sns.heatmap(sigma_display, annot=sigma_df.round(2), fmt="g",
            cmap="RdYlGn", vmin=-1, vmax=6, ax=ax2,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Sigma Level (Cpk × 3)"})

ax2.set_title("6 Sigma — Block 5: Sigma Level Scorecard by Country & Indicator\n"
              "(Green ≥ 4σ | Yellow 2–4σ | Red < 2σ)",
              fontsize=13, fontweight="bold")
ax2.set_xlabel("")
ax2.set_ylabel("")
plt.tight_layout()
plt.savefig("sigma_05_sigma_scorecard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_05_sigma_scorecard.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — YoY Delta Heatmap (% change per indicator per country)
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, len(INDICATORS), figsize=(24, 8))
fig3.suptitle("6 Sigma — Block 5: Year-over-Year % Change Heatmap",
              fontsize=15, fontweight="bold")

for ax, (col, label) in zip(axes3, INDICATORS.items()):
    pct_rows = []
    for country in df["country_name"].unique():
        sub = df[df["country_name"] == country].sort_values("year").set_index("year")
        if col not in sub.columns:
            continue
        pct = sub[col].pct_change() * 100
        for yr, val in pct.items():
            pct_rows.append({"Country": country, "Year": yr, "Delta%": val})

    pct_df = pd.DataFrame(pct_rows)
    pivot_pct = pct_df.pivot(index="Country", columns="Year", values="Delta%")

    # Filter to available years
    sns.heatmap(pivot_pct, cmap="RdYlGn", center=0, vmin=-30, vmax=30,
                ax=ax, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "YoY % change"},
                annot=False)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)

plt.tight_layout()
plt.savefig("sigma_05_yoy_delta.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_05_yoy_delta.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Executive Summary Dashboard (9-panel)
# ══════════════════════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(24, 20))
fig4.suptitle(
    "6 Sigma Executive Dashboard — World Bank Economic Indicators (2000–2023)",
    fontsize=17, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig4, hspace=0.45, wspace=0.35)

PALETTE_COUNTRY = sns.color_palette("tab10", df["country_name"].nunique())
country_color   = dict(zip(df["country_name"].unique(), PALETTE_COUNTRY))

# ── Panel (0,0): GDP per capita trend ─────────────────────────────────────
ax00 = fig4.add_subplot(gs[0, 0])
for country, color in country_color.items():
    sub = df[df["country_name"] == country].sort_values("year")
    ax00.plot(sub["year"], sub["gdp_per_capita_usd"], linewidth=1.8,
              color=color, label=country, alpha=0.85)
ax00.set_title("GDP per Capita Trend", fontsize=11, fontweight="bold")
ax00.set_xlabel("Year"); ax00.set_ylabel("USD")
ax00.legend(fontsize=5.5, ncol=2, loc="upper left")

# ── Panel (0,1): Inflation Heatmap ────────────────────────────────────────
ax01 = fig4.add_subplot(gs[0, 1])
pivot_inf = df.pivot_table(index="country_name", columns="year",
                           values="inflation_annual_pct")
sns.heatmap(pivot_inf, cmap="RdYlGn_r", center=3, ax=ax01,
            linewidths=0.3, cbar_kws={"shrink": 0.6},
            xticklabels=4, yticklabels=True)
ax01.set_title("Inflation Heatmap (%)", fontsize=11, fontweight="bold")
ax01.set_xlabel("Year"); ax01.set_ylabel("")
ax01.set_xticklabels(ax01.get_xticklabels(), rotation=90, fontsize=7)

# ── Panel (0,2): Unemployment boxplot ─────────────────────────────────────
ax02 = fig4.add_subplot(gs[0, 2])
order_unemp = (df.groupby("country_name")["unemployment_pct"]
               .median().sort_values().index.tolist())
sns.boxplot(data=df, x="unemployment_pct", y="country_name",
            order=order_unemp, palette="Set2", ax=ax02,
            flierprops=dict(marker="o", color="red", markersize=3))
ax02.axvline(df["unemployment_pct"].mean(), color="red",
             linestyle="--", linewidth=1.5, label="Global Mean")
ax02.set_title("Unemployment Distribution", fontsize=11, fontweight="bold")
ax02.set_xlabel("Unemployment (%)"); ax02.set_ylabel("")
ax02.legend(fontsize=7)

# ── Panel (1,0): Sigma Scorecard mini-table ────────────────────────────────
ax10 = fig4.add_subplot(gs[1, 0])
ax10.axis("off")
cap_df = pd.DataFrame(capability_summary)
if not cap_df.empty:
    table_data = []
    for _, row in cap_df.iterrows():
        table_data.append([
            row["Indicator"],
            f"{row['Cp']:.2f}",
            f"{row['Cpk']:.2f}",
            f"{row['Sigma Level']:.2f}",
            f"{row['DPMO']:,.0f}",
        ])
    tbl = ax10.table(
        cellText=table_data,
        colLabels=["Indicator", "Cp", "Cpk", "Sigma Lvl", "DPMO"],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.9)
    for j in range(5):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(table_data, start=1):
        cpk_val = float(row[2])
        color = "#d5f5e3" if cpk_val >= 1.33 else ("#fef9e7" if cpk_val >= 1.0 else "#fadbd8")
        for j in range(5):
            tbl[i, j].set_facecolor(color)
ax10.set_title("Capability Summary", fontsize=11, fontweight="bold", pad=40)

# ── Panel (1,1): Scatter GDP/Capita vs Unemployment with regression ────────
ax11 = fig4.add_subplot(gs[1, 1])
data_scatter = df[["gdp_per_capita_usd", "unemployment_pct",
                   "country_name"]].dropna()
sns.scatterplot(data=data_scatter, x="gdp_per_capita_usd",
                y="unemployment_pct", hue="country_name",
                alpha=0.45, s=20, ax=ax11, legend=False)
sns.regplot(data=data_scatter, x="gdp_per_capita_usd",
            y="unemployment_pct", scatter=False, ax=ax11,
            color="red", line_kws={"linewidth": 2.5})
r, p = stats.pearsonr(data_scatter["gdp_per_capita_usd"],
                       data_scatter["unemployment_pct"])
ax11.text(0.03, 0.92, f"r = {r:.3f}  (p<0.001)" if p < 0.001
          else f"r = {r:.3f}  p={p:.4f}",
          transform=ax11.transAxes, fontsize=9, color="darkred",
          fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
ax11.set_title("GDP/Capita vs Unemployment", fontsize=11, fontweight="bold")
ax11.set_xlabel("GDP per Capita (USD)"); ax11.set_ylabel("Unemployment (%)")

# ── Panel (1,2): Correlation heatmap (log-transformed) ────────────────────
ax12 = fig4.add_subplot(gs[1, 2])
num_cols = ["gdp_per_capita_usd", "gdp_current_usd",
            "inflation_annual_pct", "unemployment_pct"]
labels_short = ["GDP/cap", "GDP total", "Inflation", "Unemploy."]
df_log = df[num_cols].copy().dropna()
df_log["gdp_per_capita_usd"] = np.log1p(df_log["gdp_per_capita_usd"].clip(0))
df_log["gdp_current_usd"]    = np.log1p(df_log["gdp_current_usd"].clip(0))
corr = df_log.rename(columns=dict(zip(num_cols, labels_short))).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax12, linewidths=0.5,
            cbar_kws={"shrink": 0.7})
ax12.set_title("Correlation Matrix (log)", fontsize=11, fontweight="bold")

# ── Panel (2,0): EWMA for Inflation (global) ──────────────────────────────
ax20 = fig4.add_subplot(gs[2, 0])
LAMBDA = 0.2
yearly_inf = (df.groupby("year")["inflation_annual_pct"]
              .mean().dropna().reset_index().sort_values("year"))
vals = yearly_inf["inflation_annual_pct"].values
yrs  = yearly_inf["year"].values
ewma = np.zeros(len(vals))
ewma[0] = vals.mean()
for t in range(1, len(vals)):
    ewma[t] = LAMBDA * vals[t] + (1 - LAMBDA) * ewma[t - 1]
ax20.plot(yrs, vals, "o", color="lightcoral", alpha=0.6, markersize=5,
          label="Observed")
ax20.plot(yrs, ewma, "o-", color="steelblue", linewidth=2, markersize=5,
          label=f"EWMA (λ={LAMBDA})")
mu0   = vals.mean()
sig0  = vals.std()
ax20.axhline(mu0 + 3*sig0, color="red",   linestyle="--", linewidth=1.5,
             label="±3σ")
ax20.axhline(mu0 - 3*sig0, color="red",   linestyle="--", linewidth=1.5)
ax20.axhline(mu0,          color="green", linestyle="-",  linewidth=1.5,
             label=f"μ = {mu0:.2f}%")
ax20.set_title("EWMA — Inflation (Global)", fontsize=11, fontweight="bold")
ax20.set_xlabel("Year"); ax20.set_ylabel("Inflation (%)")
ax20.legend(fontsize=7)

# ── Panel (2,1): CV heatmap (stability) ───────────────────────────────────
ax21 = fig4.add_subplot(gs[2, 1])
cv_rows = []
for col, label in INDICATORS.items():
    for country in df["country_name"].unique():
        sub = df[df["country_name"] == country][col].dropna()
        if len(sub) > 2 and sub.mean() != 0:
            cv = (sub.std() / abs(sub.mean())) * 100
            cv_rows.append({"Country": country, "Indicator": label, "CV (%)": cv})
cv_df_all = pd.DataFrame(cv_rows)
pivot_cv = cv_df_all.pivot(index="Country", columns="Indicator",
                           values="CV (%)").fillna(0)
sns.heatmap(pivot_cv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax21,
            linewidths=0.5, cbar_kws={"label": "CV (%)", "shrink": 0.7})
ax21.set_title("Coefficient of Variation\n(Higher = Less Stable)",
               fontsize=11, fontweight="bold")
ax21.set_xlabel(""); ax21.set_ylabel("")

# ── Panel (2,2): Sigma Level Bar Chart ────────────────────────────────────
ax22 = fig4.add_subplot(gs[2, 2])
if not sigma_df.empty:
    sigma_melt = sigma_df.reset_index().melt(
        id_vars="Country", var_name="Indicator", value_name="Sigma"
    )
    sigma_melt_clean = sigma_melt.dropna()
    sigma_melt_clean = sigma_melt_clean.assign(
        Sigma=sigma_melt_clean["Sigma"].clip(-1, 6)
    )
    sns.barplot(data=sigma_melt_clean, x="Sigma", y="Country",
                hue="Indicator", ax=ax22, palette="Set1", alpha=0.85)
    ax22.axvline(1.33, color="black", linestyle="--", linewidth=1.5,
                 label="Cpk = 1.33 (4σ)")
    ax22.axvline(1.0,  color="gray",  linestyle=":",  linewidth=1.2,
                 label="Cpk = 1.0 (3σ)")
    ax22.set_title("Sigma Level by Country & Indicator",
                   fontsize=11, fontweight="bold")
    ax22.set_xlabel("Sigma Level (Cpk × 3)")
    ax22.set_ylabel("")
    ax22.legend(fontsize=7, loc="lower right")

plt.savefig("sigma_05_executive_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_05_executive_dashboard.png")

print("\n" + "="*60)
print("  Block 5 complete — 4 charts generated.")
print("  sigma_05_capability.png")
print("  sigma_05_sigma_scorecard.png")
print("  sigma_05_yoy_delta.png")
print("  sigma_05_executive_dashboard.png")
print("="*60)
print("\n  FULL 6 SIGMA ANALYSIS COMPLETE")
print("  Blocks 1-5 → 17 charts total")
print("="*60)
