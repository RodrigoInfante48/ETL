-- =============================================================================
-- Model       : mart_global_trend
-- Layer       : Marts
-- Description : Global yearly aggregates for trend analysis.
--               World GDP, avg indicators, YoY changes.
-- Materialized: table
-- =============================================================================

with metrics as (
    select * from {{ ref('int_country_metrics') }}
),

yearly as (
    select
        year,
        count(distinct country_code)                        as countries_count,
        round(sum(gdp_usd), 2)                              as world_gdp_usd,
        round(avg(gdp_per_capita_usd), 2)                   as avg_gdp_per_capita_usd,
        round(avg(inflation_pct), 2)                        as avg_inflation_pct,
        round(min(inflation_pct), 2)                        as min_inflation_pct,
        round(max(inflation_pct), 2)                        as max_inflation_pct,
        round(avg(unemployment_pct), 2)                     as avg_unemployment_pct,
        round(sum(population), 0)                           as world_population,
        round(avg(economic_health_score), 1)                as avg_health_score
    from metrics
    group by year
),

with_yoy as (
    select
        *,
        round(
            (world_gdp_usd - lag(world_gdp_usd) over (order by year))
            / nullif(lag(world_gdp_usd) over (order by year), 0)
            * 100
        , 2) as world_gdp_growth_pct
    from yearly
)

select * from with_yoy
order by year
