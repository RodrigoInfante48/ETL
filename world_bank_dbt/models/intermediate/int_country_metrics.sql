-- =============================================================================
-- Model       : int_country_metrics
-- Layer       : Intermediate
-- Description : Enriches staging data with derived metrics per country/year.
--               Adds GDP growth YoY, economic health score, and region flag.
-- Materialized: view
-- =============================================================================

with base as (
    select * from {{ ref('stg_economic_indicators') }}
),

with_growth as (
    select
        *,
        -- YoY GDP growth per country
        round(
            (gdp_usd - lag(gdp_usd) over (partition by country_code order by year))
            / nullif(lag(gdp_usd) over (partition by country_code order by year), 0)
            * 100
        , 2) as gdp_growth_pct,

        -- YoY GDP per capita growth
        round(
            (gdp_per_capita_usd - lag(gdp_per_capita_usd) over (partition by country_code order by year))
            / nullif(lag(gdp_per_capita_usd) over (partition by country_code order by year), 0)
            * 100
        , 2) as gdp_per_capita_growth_pct

    from base
),

with_scores as (
    select
        *,
        -- Economic health score (0-100):
        -- Higher GDP per capita, lower inflation, lower unemployment = better score
        round(
            (
                -- GDP per capita normalized (weight 50%)
                least(gdp_per_capita_usd / 80000.0 * 50, 50)
                -- Inflation penalty (weight 25%) — ideal is 2%, penalize deviation
                + greatest(25 - abs(coalesce(inflation_pct, 5) - 2) * 2, 0)
                -- Unemployment penalty (weight 25%) — ideal is under 5%
                + greatest(25 - coalesce(unemployment_pct, 10) * 2, 0)
            )::numeric
        , 1) as economic_health_score,

        -- LATAM flag
        case
            when country_code in ('CO','BR','MX','AR','CL','PE','EC')
            then true else false
        end as is_latam

    from with_growth
)

select * from with_scores
