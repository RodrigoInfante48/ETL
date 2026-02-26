"""
sigma_04_control_charts.py  —  6 Sigma Block 4: SPC Control Charts
Run: python sigma_04_control_charts.py
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
    print("Loading data from World Bank API...")
    raw = extract_all()
    df = transform(raw)
    print(f"Loaded: {df.shape[0]} rows\n")
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
            base_gdp *= (1 + np.random.normal(0.025, 0.04))
            rows.append({
                "country_name": c, "country_code": c[:2].upper(), "year": y,
                "gdp_per_capita_usd": base_gdp,
                "inflation_annual_pct": np.random.normal(4, 5),
                "unemployment_pct": np.random.normal(7, 3),
                "gdp_current_usd": base_gdp * np.random.lognormal(16, 0.3),
            })
    df = pd.DataFrame(rows)

sns.set_theme(style="whitegrid", font_scale=1.05)
INDICATORS = {
    "gdp_per_capita_usd": "GDP per Capita (USD)",
    "inflation_annual_pct": "Inflation (%)",
    "unemployment_pct": "Unemployment (%)",
}
D2 = 1.128  # constant for n=2 I-MR chart

# ── Figure 1: I-MR Control Chart ──────────────────────────────────────────
fig1, axes1 = plt.subplots(len(INDICATORS), 2, figsize=(20, 5 * len(INDICATORS)))
fig1.suptitle("6 Sigma — Block 4: I-MR Control Charts", fontsize=15, fontweight="bold")

for row_idx, (col, label) in enumerate(INDICATORS.items()):
    yearly = df.groupby("year")[col].mean().dropna().reset_index().sort_values("year")
    values = yearly[col].values
    years_x = yearly["year"].values
    mr = np.abs(np.diff(values))
    mr_bar    = mr.mean()
    x_bar     = values.mean()
    sigma_hat = mr_bar / D2
    ucl_x  = x_bar + 3 * sigma_hat
    lcl_x  = x_bar - 3 * sigma_hat
    ucl_mr = 3.267 * mr_bar

    ax_i = axes1[row_idx, 0]
    ax_i.plot(years_x, values, "o-", color="steelblue", linewidth=2, markersize=5, label="Observed")
    ax_i.axhline(x_bar,  color="green", linestyle="-",  linewidth=2, label=f"CL={x_bar:,.2f}")
    ax_i.axhline(ucl_x,  color="red",   linestyle="--", linewidth=1.8, label=f"UCL={ucl_x:,.2f}")
    ax_i.axhline(lcl_x,  color="red",   linestyle="--", linewidth=1.8, label=f"LCL={lcl_x:,.2f}")
    ax_i.axhline(x_bar + sigma_hat,   color="orange",   linestyle=":", linewidth=1, label="±1σ")
    ax_i.axhline(x_bar - sigma_hat,   color="orange",   linestyle=":", linewidth=1)
    ax_i.axhline(x_bar + 2*sigma_hat, color="goldenrod", linestyle=":", linewidth=1, label="±2σ")
    ax_i.axhline(x_bar - 2*sigma_hat, color="goldenrod", linestyle=":", linewidth=1)
    ooc = (values > ucl_x) | (values < lcl_x)
    ax_i.scatter(years_x[ooc], values[ooc], color="red", zorder=5, s=80, marker="X",
                 label=f"OOC (n={ooc.sum()})")
    ax_i.set_title(f"Individuals Chart — {label}", fontsize=11, fontweight="bold")
    ax_i.set_xlabel("Year"); ax_i.set_ylabel(label); ax_i.legend(fontsize=7)

    ax_mr = axes1[row_idx, 1]
    ax_mr.plot(years_x[1:], mr, "s-", color="mediumpurple", linewidth=2, markersize=5, label="MR")
    ax_mr.axhline(mr_bar,  color="green", linestyle="-",  linewidth=2, label=f"MR̄={mr_bar:,.2f}")
    ax_mr.axhline(ucl_mr,  color="red",   linestyle="--", linewidth=1.8, label=f"UCL={ucl_mr:,.2f}")
    ax_mr.axhline(0,       color="red",   linestyle="--", linewidth=1.8, label="LCL=0")
    ooc_mr = mr > ucl_mr
    ax_mr.scatter(years_x[1:][ooc_mr], mr[ooc_mr], color="red", zorder=5, s=80, marker="X")
    ax_mr.set_title(f"Moving Range Chart — {label}", fontsize=11, fontweight="bold")
    ax_mr.set_xlabel("Year"); ax_mr.set_ylabel("Moving Range"); ax_mr.legend(fontsize=7)

plt.tight_layout()
plt.savefig("sigma_04_imr_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_04_imr_charts.png")

# ── Figure 2: X-bar ± 3σ bands by country ─────────────────────────────────
TARGET_COUNTRIES = ["Colombia","Brazil","United States","China","Germany"]
PALETTE = sns.color_palette("Set2", len(TARGET_COUNTRIES))
fig2, axes2 = plt.subplots(len(INDICATORS), 1, figsize=(18, 5 * len(INDICATORS)))
fig2.suptitle("6 Sigma — Block 4: Country Trajectories with ±3σ Bands",
              fontsize=15, fontweight="bold")
for row_idx, (col, label) in enumerate(INDICATORS.items()):
    ax = axes2[row_idx]
    for i, country in enumerate(TARGET_COUNTRIES):
        sub = df[df["country_name"] == country].sort_values("year").dropna(subset=[col])
        if sub.empty: continue
        x_vals = sub["year"].values; y_vals = sub[col].values
        mu = y_vals.mean(); sigma = y_vals.std()
        ax.plot(x_vals, y_vals, "o-", color=PALETTE[i], linewidth=2, markersize=4, label=country)
        ax.fill_between(x_vals, mu-3*sigma, mu+3*sigma, color=PALETTE[i], alpha=0.07)
        ax.fill_between(x_vals, mu-sigma,   mu+sigma,   color=PALETTE[i], alpha=0.12)
    g_mean = df[col].mean(); g_std = df[col].std()
    ax.axhline(g_mean,          color="black", linestyle="-",  linewidth=1.5, label=f"Global μ={g_mean:,.2f}")
    ax.axhline(g_mean+3*g_std,  color="red",   linestyle="--", linewidth=1.5, label="Global ±3σ")
    ax.axhline(g_mean-3*g_std,  color="red",   linestyle="--", linewidth=1.5)
    ax.set_title(f"{label} — ±1σ/±3σ Bands", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel(label)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01,1), loc="upper left")
plt.tight_layout()
plt.savefig("sigma_04_xbar_bands.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_04_xbar_bands.png")

# ── Figure 3: Run Chart + Nelson Rules ────────────────────────────────────
def detect_nelson(values, ucl, lcl, cl):
    n = len(values)
    r1 = (values > ucl) | (values < lcl)
    r2 = np.zeros(n, dtype=bool)
    for i in range(8, n):
        w = values[i-8:i+1]
        if np.all(w > cl) or np.all(w < cl): r2[i-8:i+1] = True
    r3 = np.zeros(n, dtype=bool)
    for i in range(5, n):
        d = np.diff(values[i-5:i+1])
        if np.all(d > 0) or np.all(d < 0): r3[i-5:i+1] = True
    return r1, r2, r3

fig3, axes3 = plt.subplots(len(INDICATORS), 1, figsize=(18, 5 * len(INDICATORS)))
fig3.suptitle("6 Sigma — Block 4: Run Chart with Nelson Rule Violations",
              fontsize=15, fontweight="bold")
for row_idx, (col, label) in enumerate(INDICATORS.items()):
    ax = axes3[row_idx]
    yearly = df.groupby("year")[col].mean().dropna().reset_index().sort_values("year")
    values = yearly[col].values; years_x = yearly["year"].values
    mr = np.abs(np.diff(values)); sigma_hat = mr.mean() / D2
    cl = values.mean(); ucl = cl + 3*sigma_hat; lcl = cl - 3*sigma_hat
    r1, r2, r3 = detect_nelson(values, ucl, lcl, cl)
    ax.plot(years_x, values, "o-", color="steelblue", linewidth=2, markersize=5, zorder=2)
    ax.axhline(cl,  color="green", linestyle="-",  linewidth=2, label=f"CL={cl:,.2f}")
    ax.axhline(ucl, color="red",   linestyle="--", linewidth=1.8, label=f"UCL={ucl:,.2f}")
    ax.axhline(lcl, color="red",   linestyle="--", linewidth=1.8, label=f"LCL={lcl:,.2f}")
    ax.fill_between(years_x, lcl, ucl, color="green", alpha=0.04)
    if r1.any(): ax.scatter(years_x[r1], values[r1], color="red",    zorder=5, s=100, marker="X",  label=f"Rule 1 (n={r1.sum()})")
    if r2.any(): ax.scatter(years_x[r2], values[r2], color="orange", zorder=5, s=80,  marker="^",  label=f"Rule 2 — 9 same side (n={r2.sum()})")
    if r3.any(): ax.scatter(years_x[r3], values[r3], color="purple", zorder=5, s=80,  marker="D",  label=f"Rule 3 — 6 trend (n={r3.sum()})")
    ax.set_title(f"{label} — Run Chart", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel(label); ax.legend(fontsize=8)
    total_viol = (r1|r2|r3).sum()
    ax.text(0.98, 0.04, f"Total violations: {total_viol}", transform=ax.transAxes,
            ha="right", fontsize=9, color="darkred", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
plt.tight_layout()
plt.savefig("sigma_04_run_charts.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_04_run_charts.png")

# ── Figure 4: EWMA Chart ───────────────────────────────────────────────────
LAMBDA = 0.2
fig4, axes4 = plt.subplots(len(INDICATORS), 1, figsize=(18, 5 * len(INDICATORS)))
fig4.suptitle(f"6 Sigma — Block 4: EWMA Chart (λ={LAMBDA}) — Early Shift Detection",
              fontsize=15, fontweight="bold")
for row_idx, (col, label) in enumerate(INDICATORS.items()):
    ax = axes4[row_idx]
    yearly = df.groupby("year")[col].mean().dropna().reset_index().sort_values("year")
    values = yearly[col].values; years_x = yearly["year"].values; n = len(values)
    sigma0 = values.std(ddof=1); mu0 = values.mean()
    ewma = np.zeros(n); ewma[0] = mu0
    for t in range(1, n):
        ewma[t] = LAMBDA * values[t] + (1 - LAMBDA) * ewma[t-1]
    ucl_ewma = np.zeros(n); lcl_ewma = np.zeros(n)
    for t in range(n):
        factor = np.sqrt(LAMBDA/(2-LAMBDA) * (1 - (1-LAMBDA)**(2*(t+1))))
        ucl_ewma[t] = mu0 + 3 * sigma0 * factor
        lcl_ewma[t] = mu0 - 3 * sigma0 * factor
    ooc_ewma = (ewma > ucl_ewma) | (ewma < lcl_ewma)
    ax.plot(years_x, values, "o", color="lightsteelblue", markersize=5, alpha=0.6, label="Observed")
    ax.plot(years_x, ewma,   "o-", color="steelblue", linewidth=2.5, markersize=6, label="EWMA")
    ax.plot(years_x, ucl_ewma, "r--", linewidth=1.8, label="UCL/LCL")
    ax.plot(years_x, lcl_ewma, "r--", linewidth=1.8)
    ax.fill_between(years_x, lcl_ewma, ucl_ewma, color="green", alpha=0.06)
    ax.axhline(mu0, color="green", linestyle="-", linewidth=1.5, label=f"μ={mu0:,.2f}")
    if ooc_ewma.any():
        ax.scatter(years_x[ooc_ewma], ewma[ooc_ewma], color="red", zorder=5,
                   s=100, marker="X", label=f"OOC (n={ooc_ewma.sum()})")
    ax.set_title(f"{label} — EWMA (λ={LAMBDA})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel(label); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("sigma_04_ewma.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sigma_04_ewma.png")
print("\nBlock 4 complete.")
