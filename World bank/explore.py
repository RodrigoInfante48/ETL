# ============================================
# explore.py - EXPLORATORY ANALYSIS
# ============================================
# This script connects to PostgreSQL, pulls the
# clean data, and creates visualizations with seaborn.
# Run this AFTER running run_pipeline.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from config import DB_URL


def get_data():
    """Pulls data from PostgreSQL"""
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM economic_indicators", engine)
    engine.dispose()
    return df


def plot_gdp_comparison(df):
    """Bar chart: GDP per capita by country (latest year)"""
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year].sort_values("gdp_per_capita_usd", ascending=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=latest, x="gdp_per_capita_usd", y="country_name", palette="viridis")
    plt.title(f"GDP Per Capita by Country ({latest_year})", fontsize=14)
    plt.xlabel("GDP Per Capita (USD)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("gdp_per_capita.png", dpi=150)
    plt.show()
    print("💾 Saved: gdp_per_capita.png")


def plot_gdp_trend(df):
    """Line chart: GDP trend over time for all countries"""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="year", y="gdp_current_usd", hue="country_name", linewidth=2)
    plt.title("GDP Trend Over Time (2000-2023)", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("GDP (current USD)")
    plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("gdp_trend.png", dpi=150)
    plt.show()
    print("💾 Saved: gdp_trend.png")


def plot_inflation_heatmap(df):
    """Heatmap: Inflation by country and year"""
    pivot = df.pivot_table(
        index="country_name",
        columns="year",
        values="inflation_annual_pct"
    )
    
    plt.figure(figsize=(16, 6))
    sns.heatmap(pivot, cmap="RdYlGn_r", annot=False, fmt=".1f", linewidths=0.5)
    plt.title("Inflation Rate by Country and Year (%)", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("inflation_heatmap.png", dpi=150)
    plt.show()
    print("💾 Saved: inflation_heatmap.png")


def plot_unemployment_trend(df):
    """Line chart: Unemployment trend for LATAM countries"""
    latam = df[df["country_code"].isin(["CO", "BR", "MX", "AR", "CL", "PE"])]
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=latam, x="year", y="unemployment_pct", hue="country_name", linewidth=2)
    plt.title("Unemployment Rate - Latin America (2000-2023)", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Unemployment (%)")
    plt.legend(title="Country")
    plt.tight_layout()
    plt.savefig("unemployment_latam.png", dpi=150)
    plt.show()
    print("💾 Saved: unemployment_latam.png")


def run_exploration():
    """Runs all exploratory analysis"""
    print("📊 Loading data from PostgreSQL...")
    df = get_data()
    print(f"   Loaded {len(df)} rows\n")
    
    print("📊 Generating visualizations...\n")
    plot_gdp_comparison(df)
    plot_gdp_trend(df)
    plot_inflation_heatmap(df)
    plot_unemployment_trend(df)
    
    print("\n✅ All charts generated!")
    print("💡 Tip: These are for YOUR analysis.")
    print("   For stakeholders, connect Power BI to PostgreSQL.")


if __name__ == "__main__":
    run_exploration()
