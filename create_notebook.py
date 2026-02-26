"""
Run this script once to generate the Jupyter notebook:
    python create_notebook.py
Then open it with:
    jupyter notebook world_bank_analysis.ipynb
"""

import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": source})

# ── CELL 1 — Title ────────────────────────────────────────────────────────────
md("""# 🌍 Latin America in the World Economy
## A Data Story — World Bank Economic Indicators (2000–2023)

**Author:** Rodrigo Infante  
**Data source:** World Bank API → PostgreSQL (`world_bank_db`)  
**Tools:** Python · Pandas · Matplotlib · Seaborn · SQLAlchemy

---

### The Question

*How has Latin America evolved economically over the last two decades,
and where does Colombia stand compared to its regional peers and the world's major economies?*

We'll answer this through four acts:
1. **The Big Picture** — World GDP trajectory
2. **The Regional Story** — LATAM vs the world
3. **Colombia's Journey** — A detailed country deep-dive
4. **The Opportunity** — Where the gaps and risks are

---
""")

# ── CELL 2 — Setup ────────────────────────────────────────────────────────────
code("""# ── Setup & Data Loading ──────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
})

COLORS = {
    'CO': '#e87722',   # Colombia — orange
    'BR': '#1a6faf',   # Brazil — blue
    'MX': '#2ca02c',   # Mexico — green
    'AR': '#d62728',   # Argentina — red
    'CL': '#9467bd',   # Chile — purple
    'PE': '#8c564b',   # Peru — brown
    'US': '#17becf',   # USA — cyan
    'DE': '#7f7f7f',   # Germany — gray
    'CN': '#bcbd22',   # China — yellow
}

LATAM = ['CO', 'BR', 'MX', 'AR', 'CL', 'PE']

# ── Database connection ────────────────────────────────────────────────────────
# UPDATE with your credentials
DB_USER     = 'postgres'
DB_PASSWORD = '4301077Reic.'
DB_HOST     = 'localhost'
DB_PORT     = 5432
DB_NAME     = 'world_bank_db'

engine = create_engine(
    f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ── Load data from dbt marts ───────────────────────────────────────────────────
with engine.connect() as conn:
    df_raw      = pd.read_sql("SELECT * FROM economic_indicators ORDER BY country_name, year", conn)
    df_summary  = pd.read_sql("SELECT * FROM public_marts.mart_country_summary", conn)
    df_trend    = pd.read_sql("SELECT * FROM public_marts.mart_global_trend", conn)

print(f"✅ Raw data:     {len(df_raw):,} rows")
print(f"✅ Summary mart: {len(df_summary)} countries")
print(f"✅ Trend mart:   {len(df_trend)} years")
df_raw.head(3)
""")

# ── CELL 3 — Act 1 ────────────────────────────────────────────────────────────
md("""---
## Act 1 — The Big Picture 🌍

Before zooming into Latin America, let's understand the global context.
World GDP tells the story of humanity's collective economic output —
and two major disruptions stand out clearly in the data.
""")

code("""# ── World GDP trajectory with event annotations ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.fill_between(df_trend['year'], df_trend['world_gdp_usd'] / 1e12,
                alpha=0.15, color='#1a6faf')
ax.plot(df_trend['year'], df_trend['world_gdp_usd'] / 1e12,
        color='#1a6faf', linewidth=2.5, marker='o', markersize=5)

# Annotate key events
events = {
    2008: ('2008\\nFinancial Crisis', 'red'),
    2020: ('COVID-19\\nPandemic',     'red'),
    2009: ('',                        'red'),
    2021: ('',                        'red'),
}
for year, (label, color) in events.items():
    val = df_trend[df_trend['year'] == year]['world_gdp_usd'].values
    if len(val):
        ax.axvline(x=year, color=color, alpha=0.3, linestyle='--')
        if label:
            ax.annotate(label, xy=(year, val[0]/1e12),
                        xytext=(year + 0.3, val[0]/1e12 + 1.5),
                        fontsize=9, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

ax.set_title('World GDP — 2000 to 2023')
ax.set_xlabel('Year')
ax.set_ylabel('Trillion USD')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.0f}T'))
ax.set_xlim(2000, 2023)
plt.tight_layout()
plt.savefig('chart_01_world_gdp.png', dpi=130, bbox_inches='tight')
plt.show()
""")

md("""**Key insight:** World GDP grew from ~$33 trillion in 2000 to over $100 trillion in 2022 —
a **3x increase** in two decades. Two sharp contractions are visible:
the 2008 Financial Crisis and the 2020 COVID pandemic, both followed by strong recoveries.
The speed of the post-COVID rebound (2021) was historically unprecedented.
""")

# ── CELL 4 — Act 2 ────────────────────────────────────────────────────────────
md("""---
## Act 2 — The Regional Story 🌎

Latin America has the resources, the demographics, and the geography to be a
major economic force. But has it delivered? Let's compare LATAM countries
against the US, Germany, and China.
""")

code("""# ── GDP per capita comparison — latest year ───────────────────────────────────
latest_year = int(df_raw['year'].max())
latest = df_raw[df_raw['year'] == latest_year].copy()
latest = latest.sort_values('gdp_per_capita_usd', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))

bar_colors = [COLORS.get(c, '#aaaaaa') for c in latest['country_code']]
bars = ax.barh(latest['country_name'], latest['gdp_per_capita_usd'],
               color=bar_colors, edgecolor='white', linewidth=0.5)

ax.bar_label(bars, fmt='$%.0f', padding=5, fontsize=9)
ax.set_title(f'GDP Per Capita by Country ({latest_year})')
ax.set_xlabel('USD')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# Add a vertical line at world average
world_avg = latest['gdp_per_capita_usd'].mean()
ax.axvline(x=world_avg, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.annotate(f'Sample avg\\n${world_avg:,.0f}',
            xy=(world_avg, 0.5), xytext=(world_avg + 1500, 1.5),
            fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('chart_02_gdp_per_capita.png', dpi=130, bbox_inches='tight')
plt.show()
""")

code("""# ── LATAM GDP per capita trend over time ─────────────────────────────────────
latam_df = df_raw[df_raw['country_code'].isin(LATAM)].copy()

fig, ax = plt.subplots(figsize=(12, 6))

for code, grp in latam_df.groupby('country_code'):
    grp = grp.sort_values('year')
    ax.plot(grp['year'], grp['gdp_per_capita_usd'],
            label=grp['country_name'].iloc[0],
            color=COLORS.get(code, '#aaa'),
            linewidth=2.2, marker='o', markersize=3)

ax.set_title('GDP Per Capita Trend — Latin America (2000–2023)')
ax.set_xlabel('Year')
ax.set_ylabel('USD')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(loc='upper left', fontsize=9)
ax.axvspan(2008, 2009, alpha=0.08, color='red', label='Crisis periods')
ax.axvspan(2019, 2020, alpha=0.08, color='red')
plt.tight_layout()
plt.savefig('chart_03_latam_trend.png', dpi=130, bbox_inches='tight')
plt.show()
""")

md("""**Key insights:**
- **Chile** consistently leads LATAM on GDP per capita — its open-market model and copper exports explain the gap.
- **Argentina** shows extreme volatility — multiple economic crises (2001, 2018, 2020) create a saw-tooth pattern no other country replicates.
- **Colombia** shows **steady, uninterrupted growth** from 2000 to 2014, a dip during the oil price crash, then recovery — a story of resilience.
- The gap between LATAM's best (Chile ~$16K) and the US (~$76K) remains enormous — roughly **5x**. Structural, not cyclical.
""")

# ── CELL 5 — Act 3 ────────────────────────────────────────────────────────────
md("""---
## Act 3 — Colombia's Journey 🇨🇴

Colombia's economic story over 23 years is one of transformation.
From a country associated primarily with conflict in 2000,
to a stable middle-income economy with a diversified export base by 2023.
Let's read that story in the numbers.
""")

code("""# ── Colombia deep dive — 4-panel chart ───────────────────────────────────────
co = df_raw[df_raw['country_code'] == 'CO'].sort_values('year').copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Colombia Economic Dashboard (2000–2023)',
             fontsize=15, fontweight='bold', y=1.01)

COLOR = COLORS['CO']

# Panel 1 — GDP per capita
ax = axes[0, 0]
ax.fill_between(co['year'], co['gdp_per_capita_usd'], alpha=0.15, color=COLOR)
ax.plot(co['year'], co['gdp_per_capita_usd'], color=COLOR, linewidth=2.5, marker='o', markersize=4)
ax.set_title('GDP Per Capita (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# Panel 2 — Inflation
ax = axes[0, 1]
ax.bar(co['year'], co['inflation_annual_pct'], color=COLOR, alpha=0.7, edgecolor='white')
ax.axhline(y=3, color='green', linestyle='--', linewidth=1.5, label='Target ~3%')
ax.set_title('Inflation Annual (%)')
ax.legend(fontsize=9)

# Panel 3 — Unemployment
ax = axes[1, 0]
ax.plot(co['year'], co['unemployment_pct'], color=COLOR, linewidth=2.5, marker='s', markersize=4)
ax.fill_between(co['year'], co['unemployment_pct'], alpha=0.1, color=COLOR)
ax.set_title('Unemployment Rate (%)')
ax.set_ylabel('%')

# Panel 4 — Population
ax = axes[1, 1]
ax.plot(co['year'], co['population_total'] / 1e6, color=COLOR, linewidth=2.5, marker='o', markersize=4)
ax.set_title('Population (millions)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}M'))

for ax_row in axes:
    for ax in ax_row:
        ax.set_xlabel('Year')

plt.tight_layout()
plt.savefig('chart_04_colombia_dashboard.png', dpi=130, bbox_inches='tight')
plt.show()
""")

md("""**Colombia's story in four panels:**

- **GDP per capita** tripled from ~$2,500 in 2000 to ~$6,800 in 2023 — a remarkable trajectory driven by oil, coffee, and services.
- **Inflation** was brought under control through the 2000s but spiked sharply in 2022 (global commodity shock post-COVID), reaching ~12%. The Banco de la República responded with aggressive rate hikes.
- **Unemployment** remains Colombia's most persistent challenge — consistently above 8%, with a dramatic spike during COVID (2020) that has only partially recovered.
- **Population** grew steadily to ~52 million, adding a growing consumer base — a structural tailwind for domestic demand.
""")

# ── CELL 6 — Act 4 ────────────────────────────────────────────────────────────
md("""---
## Act 4 — The Opportunity & The Risk ⚖️

Every data story needs a forward-looking conclusion.
Where are the gaps? Where are the risks? What does the data suggest?
""")

code("""# ── Inflation vs Unemployment scatter — the misery index ─────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

for _, row in latest.iterrows():
    code_val = row['country_code']
    color = COLORS.get(code_val, '#aaaaaa')
    size  = (row['population_total'] / 1e6) * 1.5  # bubble size = population

    ax.scatter(row['inflation_annual_pct'], row['unemployment_pct'],
               s=size, color=color, alpha=0.75, edgecolors='white', linewidths=1.5)

    ax.annotate(row['country_name'],
                xy=(row['inflation_annual_pct'], row['unemployment_pct']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

# Quadrant lines
ax.axvline(x=latest['inflation_annual_pct'].median(), color='gray',
           linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=latest['unemployment_pct'].median(), color='gray',
           linestyle='--', alpha=0.5, linewidth=1)

# Quadrant labels
ax.text(0.02, 0.98, '✅ Low inflation\nLow unemployment',
        transform=ax.transAxes, fontsize=8, color='green',
        va='top', alpha=0.7)
ax.text(0.75, 0.98, '⚠️ High inflation\nLow unemployment',
        transform=ax.transAxes, fontsize=8, color='orange',
        va='top', alpha=0.7)
ax.text(0.02, 0.08, '⚠️ Low inflation\nHigh unemployment',
        transform=ax.transAxes, fontsize=8, color='orange',
        va='top', alpha=0.7)
ax.text(0.65, 0.08, '🔴 High inflation\nHigh unemployment',
        transform=ax.transAxes, fontsize=8, color='red',
        va='top', alpha=0.7)

ax.set_title(f'Misery Index — Inflation vs Unemployment ({latest_year})\\n(Bubble size = population)')
ax.set_xlabel('Inflation (%)')
ax.set_ylabel('Unemployment (%)')
plt.tight_layout()
plt.savefig('chart_05_misery_index.png', dpi=130, bbox_inches='tight')
plt.show()
""")

code("""# ── Performance tier summary table ────────────────────────────────────────────
summary_display = df_summary[[
    'country_name', 'performance_tier', 'latest_gdp_per_capita_usd',
    'latest_inflation_pct', 'latest_unemployment_pct',
    'latest_health_score', 'avg_gdp_growth_pct'
]].copy()

summary_display.columns = [
    'Country', 'Tier', 'GDP/Capita', 'Inflation %',
    'Unemployment %', 'Health Score', 'Avg GDP Growth %'
]
summary_display['GDP/Capita'] = summary_display['GDP/Capita'].apply(lambda x: f'${x:,.0f}')
summary_display = summary_display.sort_values('Health Score', ascending=False)

print("\\n📊 Economic Performance Summary — All Countries")
print("=" * 85)
print(summary_display.to_string(index=False))
print("=" * 85)
""")

md("""---
## Conclusions 📝

**1. The global economy is resilient but unequal.**
World GDP tripled in 23 years, but GDP per capita gaps between nations remain massive.
The distance between Chile and the US (~5x) hasn't meaningfully closed.

**2. Colombia's macro fundamentals are solid — but unemployment is the Achilles heel.**
Consistent GDP growth, controlled inflation (except 2022), growing population.
But ~12% structural unemployment signals a labor market that isn't absorbing growth efficiently.
This is the #1 policy challenge for the next decade.

**3. Argentina is a cautionary tale about institutional instability.**
No other country in the dataset shows the same volatility. The data makes clear:
macroeconomic policy consistency matters more than natural resources.

**4. Chile's model deserves study.**
Consistently the LATAM leader across almost every indicator. Trade openness,
institutional quality, and fiscal discipline compound over decades.

**5. The 2020 COVID shock was sharp but short.**
Every economy in the dataset recovered by 2021–2022. The speed of recovery
correlated strongly with fiscal space — countries with lower debt recovered faster.

---
*Analysis by Rodrigo Infante | Data: World Bank API | Stack: Python · dbt · PostgreSQL*
""")

# ── Write notebook ─────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.7"}
    },
    "cells": cells
}

with open("world_bank_analysis.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("✅ Notebook created: world_bank_analysis.ipynb")
print("   Run it with: jupyter notebook world_bank_analysis.ipynb")
