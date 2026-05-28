-- Version 2 — current stable contract (latest_version: 2 in schema.yml).
-- Added gdp_per_capita_growth_pct vs v1.
-- Cross-project consumers reach this via:
--   {{ ref('core_economics', 'int_country_metrics') }}          → v2 (latest)
--   {{ ref('core_economics', 'int_country_metrics', v=2) }}     → v2 (pinned)
--   {{ ref('core_economics', 'int_country_metrics', v=1) }}     → v1 (grace period)

with base as (
    select * from {{ ref('stg_economic_indicators') }}
),

with_growth as (
    select
        *,
        round(
            (gdp_usd - lag(gdp_usd) over (partition by country_code order by year))
            / nullif(lag(gdp_usd) over (partition by country_code order by year), 0)
            * 100
        , 2) as gdp_growth_pct,

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
        round(
            (
                least(gdp_per_capita_usd / 80000.0 * 50, 50)
                + greatest(25 - abs(coalesce(inflation_pct, 5) - 2) * 2, 0)
                + greatest(25 - coalesce(unemployment_pct, 10) * 2, 0)
            )::numeric
        , 1) as economic_health_score,

        case
            when country_code in ('CO','BR','MX','AR','CL','PE','EC')
            then true else false
        end as is_latam

    from with_growth
)

select * from with_scores
