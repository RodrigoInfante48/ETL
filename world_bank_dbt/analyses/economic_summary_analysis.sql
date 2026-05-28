-- =============================================================================
-- Analysis    : economic_summary_analysis
-- Purpose     : Cross-cutting analytical query that answers the five most
--               common questions stakeholders ask about this dataset.
--               Run with: dbt compile --select economic_summary_analysis
--               The compiled SQL in target/compiled/ can be pasted into any
--               SQL client connected to the marts schema.
--
-- Questions answered
-- ──────────────────
-- Q1. Which countries are "High Performers" and how do they compare?
-- Q2. How has world GDP evolved year over year, and when did it contract?
-- Q3. Which LATAM country improved the most in the last 5 years?
-- Q4. What is the inflation vs. unemployment trade-off per performance tier?
-- Q5. Which country/year combinations are outliers by health score?
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Q1 · Country performance snapshot
-- Ranks all 10 countries by latest health score and annotates each with
-- its performance tier, latest macro indicators, and 20-year averages.
-- ---------------------------------------------------------------------------
with country_snapshot as (
    select
        country_name,
        country_code,
        is_latam,
        performance_tier,
        rank_gdp_per_capita,
        -- Latest-year indicators
        latest_health_score,
        round(latest_gdp_per_capita_usd, 0)             as latest_gdp_pc_usd,
        round(latest_inflation_pct, 1)                  as latest_inflation_pct,
        round(latest_unemployment_pct, 1)               as latest_unemployment_pct,
        -- Historical baselines for comparison
        round(avg_gdp_per_capita_usd, 0)                as avg_gdp_pc_usd,
        round(avg_inflation_pct, 1)                     as avg_inflation_pct,
        round(avg_unemployment_pct, 1)                  as avg_unemployment_pct,
        round(avg_health_score, 1)                      as avg_health_score,
        years_reported
    from {{ ref('mart_country_summary') }}
    order by latest_health_score desc
),


-- ---------------------------------------------------------------------------
-- Q2 · World GDP trend + contraction flags
-- Marks years where world GDP contracted (growth < 0). Analysts typically
-- expect dips in 2009 (Global Financial Crisis) and 2020 (COVID-19).
-- ---------------------------------------------------------------------------
gdp_trend as (
    select
        year,
        round(world_gdp_usd / 1e12, 2)                 as world_gdp_trillion_usd,
        world_gdp_growth_pct,
        avg_health_score,
        countries_count,
        case
            when world_gdp_growth_pct < 0 then 'Contraction'
            when world_gdp_growth_pct < 2 then 'Slow Growth'
            when world_gdp_growth_pct < 5 then 'Moderate Growth'
            else 'Strong Growth'
        end                                             as growth_regime,
        -- Running total of contractions to date (useful in time-series charts)
        count(case when world_gdp_growth_pct < 0 then 1 end)
            over (order by year rows between unbounded preceding and current row)
                                                        as cumulative_contractions
    from {{ ref('mart_global_trend') }}
    order by year
),


-- ---------------------------------------------------------------------------
-- Q3 · LATAM health score improvement — last 5 vs. first 5 years
-- Computes average health score in the first 5 years (2000-2004) vs.
-- the last 5 years (2019-2023) to rank improvement trajectories.
-- Note: uses int_country_metrics directly for year-level granularity.
-- ---------------------------------------------------------------------------
latam_improvement as (
    select
        country_code,
        country_name,
        round(avg(case when year between 2000 and 2004 then economic_health_score end), 1)
                                                        as avg_score_2000_2004,
        round(avg(case when year between 2019 and 2023 then economic_health_score end), 1)
                                                        as avg_score_2019_2023,
        round(
            avg(case when year between 2019 and 2023 then economic_health_score end)
            - avg(case when year between 2000 and 2004 then economic_health_score end)
        , 1)                                            as score_improvement
    from {{ ref('int_country_metrics') }}
    where is_latam = true
    group by country_code, country_name
    order by score_improvement desc
),


-- ---------------------------------------------------------------------------
-- Q4 · Inflation vs. unemployment by performance tier
-- Aggregates macro trade-off metrics per tier to understand whether
-- "High Performers" achieve low inflation AND low unemployment simultaneously
-- (a macroeconomic ideal often in tension — the Phillips Curve).
-- ---------------------------------------------------------------------------
tier_tradeoffs as (
    select
        performance_tier,
        count(*)                                        as countries_in_tier,
        round(avg(avg_inflation_pct), 2)                as tier_avg_inflation,
        round(avg(avg_unemployment_pct), 2)             as tier_avg_unemployment,
        round(avg(avg_gdp_per_capita_usd), 0)           as tier_avg_gdp_pc,
        round(avg(latest_health_score), 1)              as tier_avg_latest_score,
        -- Flag tiers that achieve both low inflation (<4%) and low unemployment (<7%)
        case
            when avg(avg_inflation_pct) < 4
             and avg(avg_unemployment_pct) < 7
            then true
            else false
        end                                             as achieves_dual_mandate
    from {{ ref('mart_country_summary') }}
    group by performance_tier
    order by tier_avg_latest_score desc
),


-- ---------------------------------------------------------------------------
-- Q5 · Health score outliers (3-sigma rule applied to the full panel)
-- Identifies country/year observations where the health score deviates
-- more than 2 standard deviations from the panel mean — the same
-- 6-Sigma methodology used in the Seaborn reporting layer.
-- ---------------------------------------------------------------------------
panel_stats as (
    select
        avg(economic_health_score)                      as panel_mean,
        stddev(economic_health_score)                   as panel_stddev
    from {{ ref('int_country_metrics') }}
),

outliers as (
    select
        m.country_code,
        m.country_name,
        m.year,
        round(m.economic_health_score, 1)               as health_score,
        round(p.panel_mean, 1)                          as panel_mean,
        round(p.panel_stddev, 1)                        as panel_stddev,
        round((m.economic_health_score - p.panel_mean) / nullif(p.panel_stddev, 0), 2)
                                                        as z_score,
        case
            when abs((m.economic_health_score - p.panel_mean)
                 / nullif(p.panel_stddev, 0)) > 3 then '3σ outlier'
            when abs((m.economic_health_score - p.panel_mean)
                 / nullif(p.panel_stddev, 0)) > 2 then '2σ outlier'
            else 'within_normal'
        end                                             as sigma_band
    from {{ ref('int_country_metrics') }} m
    cross join panel_stats p
    where abs((m.economic_health_score - p.panel_mean)
          / nullif(p.panel_stddev, 0)) > 2
    order by abs(z_score) desc
)


-- ---------------------------------------------------------------------------
-- Final output — change the CTE name below to explore each question
-- ---------------------------------------------------------------------------
-- select * from country_snapshot;
-- select * from gdp_trend;
-- select * from latam_improvement;
-- select * from tier_tradeoffs;
select * from outliers;
