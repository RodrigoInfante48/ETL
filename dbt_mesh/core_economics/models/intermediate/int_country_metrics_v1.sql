-- Version 1 — historical contract. Kept alive while consumers migrate to v2.
-- Breaking change introduced in v2: added gdp_per_capita_growth_pct column.
-- Consumers pinned to v1: {{ ref('int_country_metrics', v=1) }}
-- Consumers using latest:  {{ ref('int_country_metrics') }}  → resolves to v2

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
        , 2) as gdp_growth_pct
        -- gdp_per_capita_growth_pct not present in v1 — that is the breaking addition in v2

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
