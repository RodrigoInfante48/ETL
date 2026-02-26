"""
sigma_04_control_charts.py
===========================
6 Sigma Analysis - Block 4: Statistical Process Control (SPC)
Author: Rodrigo Infante
Dataset: World Bank Economic Indicators (2000-2023)
Run: python sigma_04_control_charts.py

Charts produced:
  Fig 1 — I-MR Control Chart (Individuals + Moving Range) per indicator
  Fig 2 — X-bar / S chart (mean ± control limits by country)
  Fig 3 — Run chart with trend & shift detection (Nelson Rules)
  Fig 4 — EWMA (Exponentially Weighted Moving Average) — early-warning chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# d2 constant for n=2 (used in I-MR charts)
D2 = 1.128


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — I-MR Control Chart (global series, each indicator)
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(len(INDICATORS), 2, figsize=(22, 6 * len(INDICATORS)))
fig1.suptitle("6 Sigma — Block 4: I-MR Control Charts (Individuals & Moving Range)",
              fontsize=15, fontweight="bold", y=1.01)

for row_idx, (col, label) in enumerate(INDICATORS.items()):
    # Build global yearly mean so the time series is ordered
    yearly = (df.groupby("year")[col].mean().dropna().reset_index()
              .sort_values("year"))
    values = yearly[col].values
    years_x = yearly["year"].values

    # Moving Range
    mr = np.abs(np.diff(values))
    mr_bar = mr.mean()
    x_bar  = values.mean()
    sigma_hat = mr_bar / D2

    ucl_x = x_bar + 3 * sigma_hat
    lcl_x = x_bar - 3 * sigma_hat
    ucl_mr = 3.267 * mr_bar        # D4 constant for n=2
    lcl_mr = 0.0                   # D3 constant for n=2 (= 0)

    # ── Individuals chart ──────────────────────────────────────────────
    ax_i = axes1[row_idx, 0]
    ax_i.plot(years_x, values, "o-", color="steelblue", linewidth=2,
              markersize=5, label="Observed")
    ax_i.axhline(x_bar,  color="green",  linestyle="-",  linewidth=2,
                 label=f"CL = {x_bar:,.2f}")
    ax_i.axhline(ucl_x, color="red",    linestyle="--", linewidth=1.8,
                 label=f"UCL = {ucl_x:,.2f}")
    ax_i.axhline(lcl_x, color="red",    linestyle="--", linewidth=1.8,
                 label=f"LCL = {lcl_x:,.2f}")
    ax_i.axhline(x_bar + sigma_hat, color="orange", linestyle=":", linewidth=1,
                 label="+1σ / -1σ")
    ax_i.axhline(x_bar - sigma_hat, color="orange", linestyle=":", linewidth=1)
    ax_i.axhline(x_bar + 2 * sigma_hat, color="goldenrod", linestyle=":",
                 linewidth=1, label="+2σ / -2σ")
    ax_i.axhline(x_bar - 2 * sigma_hat, color="goldenrod", linestyle=":", linewidth=1)

    # Flag out-of-control points
    ooc = (values > ucl_x) | (values < lcl_x)
    ax_i.scatter(years_x[ooc], values[ooc], color="red", zorder=5,
                 s=80, marker="X", label="Out of Control")

    ax_i.set_title(f"Individuals (I) Chart\n{label}", fontsize=11, fontweight="bold", pad=8)
    ax_i.set_xlabel("Year")
    ax_i.set_ylabel(label)
    ax_i.legend(fontsize=7, loc="upper left")
    n_ooc = ooc.sum()
    ax_i.text(0.98, 0.04, f"OOC points: {n_ooc}",
              transform=ax_i.transAxes, ha="right", fontsize=9,
              color="red", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # ── Moving Range chart ─────────────────────────────────────────────
    ax_mr = axes1[row_idx, 1]
    ax_mr.plot(years_x[1:], mr, "s-", color="mediumpurple", linewidth=2,
               markersize=5, label="Moving Range")
    ax_mr.axhline(mr_bar,  color="green", linestyle="-",  linewidth=2,
                  label=f"MR̄ = {mr_bar:,.2f}")
    ax_mr.axhline(ucl_mr, color="red",   linestyle="--", linewidth=1.8,
                  label=f"UCL = {ucl_mr:,.2f}")
    ax_mr.axhline(lcl_mr, color="red",   linestyle="--", linewidth=1.8,
                  label=f"LCL = {lcl_mr:.2f}")
    ooc_mr = mr > ucl_mr
    ax_mr.scatter(years_x[1:][ooc_mr], mr[ooc_mr], color="red", zorder=5,
                  s=80, marker="X", label="Out of Control")
    ax_mr.set_title(f"Moving Range (MR) Chart\n{label}", fontsize=11, fontweight="bold", pad=8)
    ax_mr.set_xlabel("Year")
    ax_mr.set_ylabel("Moving Range")
    ax_mr.legend(fontsize=7, loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("sigma_04_imr_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_04_imr_charts.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — X-bar / S chart: mean ± control limits BY COUNTRY over years
# ══════════════════════════════════════════════════════════════════════════════
TARGET_COUNTRIES = ["Colombia", "Brazil", "United States", "China", "Germany"]
PALETTE = sns.color_palette("Set2", len(TARGET_COUNTRIES))

fig2, axes2 = plt.subplots(len(INDICATORS), 1, figsize=(18, 6 * len(INDICATORS)))
fig2.suptitle("6 Sigma — Block 4: X-bar ± 3σ Control Bands by Country",
              fontsize=15, fontweight="bold", y=1.01)

for row_idx, (col, label) in enumerate(INDICATORS.items()):
    ax = axes2[row_idx]
    for i, country in enumerate(TARGET_COUNTRIES):
        sub = df[df["country_name"] == country].sort_values("year")
        sub = sub.dropna(subset=[col])
        if sub.empty:
            continue
        x_vals = sub["year"].values
        y_vals = sub[col].values
        mu = y_vals.mean()
        sigma = y_vals.std()

        ax.plot(x_vals, y_vals, "o-", color=PALETTE[i], linewidth=2,
                markersize=4, label=country)
        ax.fill_between(x_vals, mu - 3 * sigma, mu + 3 * sigma,
                        color=PALETTE[i], alpha=0.08)
        ax.fill_between(x_vals, mu - sigma, mu + sigma,
                        color=PALETTE[i], alpha=0.12)

    # Global reference lines
    g_mean = df[col].mean()
    g_std  = df[col].std()
    ax.axhline(g_mean,           color="black",  linestyle="-",  linewidth=1.5,
               label=f"Global μ = {g_mean:,.2f}")
    ax.axhline(g_mean + 3*g_std, color="red",    linestyle="--", linewidth=1.5,
               label=f"Global ±3σ = {3*g_std:,.2f}")
    ax.axhline(g_mean - 3*g_std, color="red",    linestyle="--", linewidth=1.5)
    ax.set_title(f"{label}\nCountry Trajectories with ±1σ / ±3σ Bands",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("sigma_04_xbar_bands.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_04_xbar_bands.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Run Chart + Nelson Rule Violations
# Nelson Rule 1: > UCL or < LCL
# Nelson Rule 2: 9 consecutive points same side of CL
# Nelson Rule 3: 6 consecutive increasing or decreasing
# ══════════════════════════════════════════════════════════════════════════════
def detect_nelson(values, ucl, lcl, cl):
    """Returns boolean masks for Nelson Rule violations 1, 2, 3."""
    n = len(values)
    r1 = (values > ucl) | (values < lcl)

    # Rule 2: 9 in a row on same side
    r2 = np.zeros(n, dtype=bool)
    for i in range(8, n):
        window = values[i-8:i+1]
        if np.all(window > cl) or np.all(window < cl):
            r2[i-8:i+1] = True

    # Rule 3: 6 in a row strictly increasing or decreasing
    r3 = np.zeros(n, dtype=bool)
    for i in range(5, n):
        window = values[i-5:i+1]
        diffs = np.diff(window)
        if np.all(diffs > 0) or np.all(diffs < 0):
            r3[i-5:i+1] = True

    return r1, r2, r3


fig3, axes3 = plt.subplots(len(INDICATORS), 1, figsize=(18, 6 * len(INDICATORS)))
fig3.suptitle("6 Sigma — Block 4: Run Chart with Nelson Rule Violations",
              fontsize=15, fontweight="bold", y=1.01)

for row_idx, (col, label) in enumerate(INDICATORS.items()):
    ax = axes3[row_idx]
    yearly = (df.groupby("year")[col].mean().dropna()
              .reset_index().sort_values("year"))
    values = yearly[col].values
    years_x = yearly["year"].values

    mr = np.abs(np.diff(values))
    mr_bar = mr.mean()
    sigma_hat = mr_bar / D2
    cl  = values.mean()
    ucl = cl + 3 * sigma_hat
    lcl = cl - 3 * sigma_hat

    r1, r2, r3 = detect_nelson(values, ucl, lcl, cl)

    ax.plot(years_x, values, "o-", color="steelblue", linewidth=2,
            markersize=5, zorder=2)
    ax.axhline(cl,  color="green", linestyle="-",  linewidth=2, label=f"CL = {cl:,.2f}")
    ax.axhline(ucl, color="red",   linestyle="--", linewidth=1.8,
               label=f"UCL = {ucl:,.2f}")
    ax.axhline(lcl, color="red",   linestyle="--", linewidth=1.8,
               label=f"LCL = {lcl:,.2f}")
    ax.fill_between(years_x, lcl, ucl, color="green", alpha=0.04)

    # Highlight violations
    if r1.any():
        ax.scatter(years_x[r1], values[r1], color="red", zorder=5,
                   s=100, marker="X", label=f"Rule 1 (n={r1.sum()})")
    if r2.any():
        ax.scatter(years_x[r2], values[r2], color="orange", zorder=5,
                   s=80, marker="^", label=f"Rule 2 — 9 same side (n={r2.sum()})")
    if r3.any():
        ax.scatter(years_x[r3], values[r3], color="purple", zorder=5,
                   s=80, marker="D", label=f"Rule 3 — 6 trend (n={r3.sum()})")

    ax.set_title(f"{label} — Run Chart", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.legend(fontsize=8, loc="upper left")

    total_viol = (r1 | r2 | r3).sum()
    ax.text(0.98, 0.04, f"Total violations: {total_viol}",
            transform=ax.transAxes, ha="right", fontsize=9,
            color="darkred", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("sigma_04_run_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_04_run_charts.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — EWMA Chart (lambda=0.2, early-warning for small shifts)
# ══════════════════════════════════════════════════════════════════════════════
LAMBDA = 0.2      # smoothing factor (0.1–0.3 typical for SPC)
L_SIGMA = 3.0     # control limit multiplier

fig4, axes4 = plt.subplots(len(INDICATORS), 1, figsize=(18, 6 * len(INDICATORS)))
fig4.suptitle(
    f"6 Sigma — Block 4: EWMA Control Chart (λ={LAMBDA}) — Early Shift Detection",
    fontsize=15, fontweight="bold", y=1.01
)

for row_idx, (col, label) in enumerate(INDICATORS.items()):
    ax = axes4[row_idx]
    yearly = (df.groupby("year")[col].mean().dropna()
              .reset_index().sort_values("year"))
    values = yearly[col].values
    years_x = yearly["year"].values
    n = len(values)

    sigma0 = values.std(ddof=1)
    mu0    = values.mean()

    # Compute EWMA
    ewma = np.zeros(n)
    ewma[0] = mu0
    for t in range(1, n):
        ewma[t] = LAMBDA * values[t] + (1 - LAMBDA) * ewma[t - 1]

    # Dynamic control limits
    ucl_ewma = np.zeros(n)
    lcl_ewma = np.zeros(n)
    for t in range(n):
        factor = np.sqrt(LAMBDA / (2 - LAMBDA) * (1 - (1 - LAMBDA) ** (2 * (t + 1))))
        ucl_ewma[t] = mu0 + L_SIGMA * sigma0 * factor
        lcl_ewma[t] = mu0 - L_SIGMA * sigma0 * factor

    ooc_ewma = (ewma > ucl_ewma) | (ewma < lcl_ewma)

    ax.plot(years_x, values,    "o",  color="lightsteelblue", markersize=5,
            alpha=0.6, label="Observed", zorder=2)
    ax.plot(years_x, ewma,      "o-", color="steelblue", linewidth=2.5,
            markersize=6, label="EWMA", zorder=3)
    ax.plot(years_x, ucl_ewma,  "r--", linewidth=1.8, label="UCL / LCL")
    ax.plot(years_x, lcl_ewma,  "r--", linewidth=1.8)
    ax.fill_between(years_x, lcl_ewma, ucl_ewma, color="green", alpha=0.06)
    ax.axhline(mu0, color="green", linestyle="-", linewidth=1.5,
               label=f"Target (μ) = {mu0:,.2f}")

    if ooc_ewma.any():
        ax.scatter(years_x[ooc_ewma], ewma[ooc_ewma], color="red",
                   zorder=5, s=100, marker="X",
                   label=f"OOC EWMA (n={ooc_ewma.sum()})")

    ax.set_title(f"{label} — EWMA (λ={LAMBDA})", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.legend(fontsize=8, loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("sigma_04_ewma.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_04_ewma.png")

print("\nBlock 4 complete — 4 charts generated.")
print("  sigma_04_imr_charts.png")
print("  sigma_04_xbar_bands.png")
print("  sigma_04_run_charts.png")
print("  sigma_04_ewma.png")
