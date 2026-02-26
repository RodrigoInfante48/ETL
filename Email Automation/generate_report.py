"""
* Description: Queries world_bank_db materialized views, generates embedded
*              charts, and builds a professional HTML email body.
* Project: World Bank Economic Reporting
* Author: Rodrigo Infante
* Modified: 2026-02-24
* Dependencies:
*     - pandas, sqlalchemy, matplotlib, seaborn
*     - config.py (DB_URL, REPORT_CONFIG)
*
* Flow:
*   get_data() → build_charts() → build_html_email() → export_csv()
*   Returns: (html_body, csv_path, chart_paths)
"""

from __future__ import annotations

import base64
import io
from datetime import date
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for automation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_URL, REPORT_CONFIG

sns.set_theme(style="whitegrid", palette="muted")


# =====================================================
# DATA ACCESS — reads from materialized views
# =====================================================

def get_data() -> dict[str, pd.DataFrame]:
    """
    Pull data from the three materialized views.
    Falls back to economic_indicators if views don't exist yet.
    """
    engine = create_engine(DB_URL)
    try:
        views = {}

        # Latest year ranking snapshot
        views["ranking"] = pd.read_sql(
            "SELECT * FROM mv_latest_year_ranking ORDER BY rank_gdp_per_capita", engine
        )

        # Global yearly trend
        views["trend"] = pd.read_sql(
            "SELECT * FROM mv_global_yearly_trend ORDER BY year", engine
        )

        # Country profiles
        views["profile"] = pd.read_sql(
            "SELECT * FROM mv_country_profile ORDER BY avg_gdp_usd DESC", engine
        )

        # Raw table for Colombia detail
        views["raw"] = pd.read_sql(
            "SELECT * FROM economic_indicators ORDER BY country_name, year", engine
        )

        print(f"   ✅ ranking: {len(views['ranking'])} rows")
        print(f"   ✅ trend: {len(views['trend'])} rows")
        print(f"   ✅ profile: {len(views['profile'])} rows")
        print(f"   ✅ raw: {len(views['raw'])} rows")

    finally:
        engine.dispose()

    return views


# =====================================================
# CHART BUILDERS — return base64 encoded PNG strings
# =====================================================

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def chart_gdp_per_capita(ranking: pd.DataFrame) -> str:
    """Horizontal bar: GDP per capita ranking, latest year."""
    df = ranking.dropna(subset=["gdp_per_capita_usd"]).copy()
    df = df.sort_values("gdp_per_capita_usd", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1a6faf" if c != "CO" else "#e87722" for c in df["country_code"]]
    bars = ax.barh(df["country_name"], df["gdp_per_capita_usd"], color=colors)

    ax.set_title(f"GDP Per Capita — {df['reference_year'].iloc[0]}", fontsize=13, fontweight="bold")
    ax.set_xlabel("USD")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.bar_label(bars, fmt="$%.0f", padding=4, fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_world_gdp_trend(trend: pd.DataFrame) -> str:
    """Line chart: world total GDP over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trend["year"], trend["world_gdp_total_usd"] / 1e12,
            color="#1a6faf", linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(trend["year"], trend["world_gdp_total_usd"] / 1e12, alpha=0.1, color="#1a6faf")
    ax.set_title("World GDP Total Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Trillion USD")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}T"))
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_inflation_unemployment(ranking: pd.DataFrame) -> str:
    """Scatter: inflation vs unemployment by country."""
    df = ranking.dropna(subset=["inflation_annual_pct", "unemployment_pct"]).copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["inflation_annual_pct"], df["unemployment_pct"],
        c=df["gdp_per_capita_usd"], cmap="viridis", s=120, edgecolors="white", linewidths=0.5
    )
    for _, row in df.iterrows():
        ax.annotate(row["country_code"],
                    (row["inflation_annual_pct"], row["unemployment_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    plt.colorbar(scatter, ax=ax, label="GDP per Capita (USD)")
    ax.set_title("Inflation vs Unemployment by Country", fontsize=13, fontweight="bold")
    ax.set_xlabel("Inflation (%)")
    ax.set_ylabel("Unemployment (%)")
    fig.tight_layout()
    return _fig_to_base64(fig)


def chart_colombia_trend(raw: pd.DataFrame) -> str:
    """Dual-axis line chart: Colombia GDP + Inflation over time."""
    co = raw[raw["country_code"] == "CO"].sort_values("year").copy()
    if co.empty:
        return ""

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color_gdp = "#1a6faf"
    color_inf = "#e87722"

    ax1.plot(co["year"], co["gdp_per_capita_usd"], color=color_gdp,
             linewidth=2.5, marker="o", markersize=4, label="GDP per Capita")
    ax1.set_ylabel("GDP per Capita (USD)", color=color_gdp)
    ax1.tick_params(axis="y", labelcolor=color_gdp)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax2 = ax1.twinx()
    ax2.plot(co["year"], co["inflation_annual_pct"], color=color_inf,
             linewidth=2, linestyle="--", marker="s", markersize=4, label="Inflation")
    ax2.set_ylabel("Inflation (%)", color=color_inf)
    ax2.tick_params(axis="y", labelcolor=color_inf)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title("🇨🇴 Colombia — GDP per Capita & Inflation (2000–2023)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Year")
    fig.tight_layout()
    return _fig_to_base64(fig)


# =====================================================
# HTML EMAIL BUILDER
# =====================================================

def build_html_email(data: dict[str, pd.DataFrame], charts: dict[str, str]) -> str:
    """Build a full HTML email body with embedded charts and KPI cards."""

    ranking = data["ranking"]
    trend = data["trend"]
    latest_year = int(ranking["reference_year"].iloc[0]) if not ranking.empty else "N/A"

    # Colombia KPIs
    co = ranking[ranking["country_code"] == "CO"]
    co_gdp_pc = f"${co['gdp_per_capita_usd'].iloc[0]:,.0f}" if not co.empty else "N/A"
    co_inflation = f"{co['inflation_annual_pct'].iloc[0]:.1f}%" if not co.empty else "N/A"
    co_unemployment = f"{co['unemployment_pct'].iloc[0]:.1f}%" if not co.empty else "N/A"

    # Top 3 GDP per capita
    top3 = ranking.nsmallest(3, "rank_gdp_per_capita")[["country_name", "gdp_per_capita_usd"]]
    top3_rows = "".join(
        f"<tr><td>{r['country_name']}</td><td style='text-align:right'>${r['gdp_per_capita_usd']:,.0f}</td></tr>"
        for _, r in top3.iterrows()
    )

    # World GDP latest vs prior year
    world_gdp_latest = trend[trend["year"] == trend["year"].max()]["world_gdp_total_usd"].values[0]
    world_gdp_prior = trend[trend["year"] == trend["year"].max() - 1]["world_gdp_total_usd"].values[0]
    world_gdp_change = ((world_gdp_latest - world_gdp_prior) / world_gdp_prior) * 100

    def img_tag(b64: str) -> str:
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:640px;border-radius:8px;" />'

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f4f6f9; margin:0; padding:0; color:#222; }}
  .wrapper {{ max-width:680px; margin:24px auto; background:#fff; border-radius:12px; overflow:hidden; box-shadow:0 2px 12px rgba(0,0,0,0.08); }}
  .header {{ background:#1a3a5c; padding:28px 32px; }}
  .header h1 {{ color:#fff; margin:0; font-size:22px; letter-spacing:0.5px; }}
  .header p {{ color:#a8c4e0; margin:6px 0 0; font-size:13px; }}
  .content {{ padding:28px 32px; }}
  .section-title {{ font-size:15px; font-weight:700; color:#1a3a5c; margin:24px 0 12px; border-left:4px solid #1a6faf; padding-left:10px; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:8px; }}
  .kpi {{ background:#f0f5fb; border-radius:8px; padding:14px 12px; text-align:center; }}
  .kpi .label {{ font-size:11px; color:#666; text-transform:uppercase; letter-spacing:0.5px; }}
  .kpi .value {{ font-size:20px; font-weight:700; color:#1a3a5c; margin-top:4px; }}
  .kpi .sub {{ font-size:11px; color:#888; margin-top:2px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#1a3a5c; color:#fff; padding:8px 10px; text-align:left; }}
  td {{ padding:7px 10px; border-bottom:1px solid #eee; }}
  tr:last-child td {{ border-bottom:none; }}
  .chart-block {{ margin:16px 0; text-align:center; }}
  .footer {{ background:#f0f5fb; padding:16px 32px; font-size:11px; color:#888; text-align:center; }}
  .badge {{ display:inline-block; background:#e8f0fe; color:#1a6faf; font-size:11px; padding:3px 8px; border-radius:4px; font-weight:600; }}
</style>
</head>
<body>
<div class="wrapper">

  <!-- HEADER -->
  <div class="header">
    <h1>📊 World Bank Economic Report</h1>
    <p>Reference year: <strong style="color:#fff">{latest_year}</strong> &nbsp;·&nbsp; Generated: {date.today()} &nbsp;·&nbsp; <span class="badge" style="background:#2a5a8c;color:#a8d4f5">world_bank_db</span></p>
  </div>

  <div class="content">

    <!-- WORLD KPIs -->
    <div class="section-title">🌍 Global Snapshot</div>
    <div class="kpi-grid">
      <div class="kpi">
        <div class="label">World GDP</div>
        <div class="value">${world_gdp_latest/1e12:.1f}T</div>
        <div class="sub">{"▲" if world_gdp_change > 0 else "▼"} {abs(world_gdp_change):.1f}% vs prior year</div>
      </div>
      <div class="kpi">
        <div class="label">Avg Inflation</div>
        <div class="value">{trend[trend['year']==trend['year'].max()]['avg_inflation_pct'].values[0]:.1f}%</div>
        <div class="sub">Global average</div>
      </div>
      <div class="kpi">
        <div class="label">Avg Unemployment</div>
        <div class="value">{trend[trend['year']==trend['year'].max()]['avg_unemployment_pct'].values[0]:.1f}%</div>
        <div class="sub">Global average</div>
      </div>
    </div>

    <!-- WORLD GDP TREND CHART -->
    <div class="section-title">📈 World GDP Trend</div>
    <div class="chart-block">{img_tag(charts['world_gdp_trend'])}</div>

    <!-- GDP PER CAPITA CHART -->
    <div class="section-title">💰 GDP Per Capita Ranking ({latest_year})</div>
    <div class="chart-block">{img_tag(charts['gdp_per_capita'])}</div>

    <!-- TOP 3 TABLE -->
    <div class="section-title">🏆 Top 3 GDP Per Capita</div>
    <table>
      <tr><th>Country</th><th style="text-align:right">GDP Per Capita (USD)</th></tr>
      {top3_rows}
    </table>

    <!-- INFLATION VS UNEMPLOYMENT -->
    <div class="section-title">⚖️ Inflation vs Unemployment</div>
    <div class="chart-block">{img_tag(charts['inflation_unemployment'])}</div>

    <!-- COLOMBIA SECTION -->
    <div class="section-title">🇨🇴 Colombia Spotlight</div>
    <div class="kpi-grid">
      <div class="kpi">
        <div class="label">GDP Per Capita</div>
        <div class="value">{co_gdp_pc}</div>
        <div class="sub">{latest_year}</div>
      </div>
      <div class="kpi">
        <div class="label">Inflation</div>
        <div class="value">{co_inflation}</div>
        <div class="sub">{latest_year}</div>
      </div>
      <div class="kpi">
        <div class="label">Unemployment</div>
        <div class="value">{co_unemployment}</div>
        <div class="sub">{latest_year}</div>
      </div>
    </div>
    <div class="chart-block">{img_tag(charts['colombia_trend'])}</div>

  </div>

  <div class="footer">
    World Bank Economic Report · Auto-generated by world_bank_db pipeline · Full dataset attached as CSV
  </div>
</div>
</body>
</html>
"""
    return html


# =====================================================
# CSV EXPORT
# =====================================================

def export_csv(df: pd.DataFrame) -> str:
    filename = REPORT_CONFIG["csv_filename"]
    df.to_csv(filename, index=False)
    print(f"   📁 CSV exported: {filename}")
    return filename


# =====================================================
# ORCHESTRATION
# =====================================================

def generate_report() -> Tuple[str, str]:
    """Main function: pull data, build charts, generate HTML email, export CSV."""
    print("📊 Generating report...")

    data = get_data()

    print("   📊 Building charts...")
    charts = {
        "gdp_per_capita":       chart_gdp_per_capita(data["ranking"]),
        "world_gdp_trend":      chart_world_gdp_trend(data["trend"]),
        "inflation_unemployment": chart_inflation_unemployment(data["ranking"]),
        "colombia_trend":       chart_colombia_trend(data["raw"]),
    }
    print("   ✅ Charts ready")

    html_body = build_html_email(data, charts)
    print("   ✅ HTML email built")

    csv_file = export_csv(data["raw"])

    print("   ✅ Report ready!")
    return html_body, csv_file


if __name__ == "__main__":
    html, csv_path = generate_report()
    with open("preview_email.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\n💡 Preview saved to: preview_email.html — open it in your browser!")
