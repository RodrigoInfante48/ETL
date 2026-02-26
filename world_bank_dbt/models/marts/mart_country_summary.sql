-- =============================================================================
-- Model       : mart_country_summary
-- Layer       : Marts
-- Description : Final analytical model — one row per country with historical
--               averages, latest year snapshot, and performance classification.
--               Ready for BI tools and reporting.
-- Materialized: table
-- =============================================================================

with metrics as (
    select * from {{ ref('int_country_metrics') }}
),

latest_year as (
    select max(year) as max_year from metrics
),

historical as (
    select
        country_code,
        country_name,
        is_latam,
        count(*)                                    as years_reported,
        min(year)                                   as first_year,
        max(year)                                   as last_year,
        round(avg(gdp_usd), 2)                      as avg_gdp_usd,
        round(avg(gdp_per_capita_usd), 2)           as avg_gdp_per_capita_usd,
        round(avg(inflation_pct), 2)                as avg_inflation_pct,
        round(avg(unemployment_pct), 2)             as avg_unemployment_pct,
        round(avg(population), 0)                   as avg_population,
        round(avg(gdp_growth_pct), 2)               as avg_gdp_growth_pct,
        round(avg(economic_health_score), 1)        as avg_health_score
    from metrics
    group by country_code, country_name, is_latam
),

latest_snapshot as (
    select
        m.country_code,
        m.gdp_usd                                   as latest_gdp_usd,
        m.gdp_per_capita_usd                        as latest_gdp_per_capita_usd,
        m.inflation_pct                             as latest_inflation_pct,
        m.unemployment_pct                          as latest_unemployment_pct,
        m.population                                as latest_population,
        m.gdp_growth_pct                            as latest_gdp_growth_pct,
        m.economic_health_score                     as latest_health_score
    from metrics m
    join latest_year ly on m.year = ly.max_year
),

final as (
    select
        h.*,
        s.latest_gdp_usd,
        s.latest_gdp_per_capita_usd,
        s.latest_inflation_pct,
        s.latest_unemployment_pct,
        s.latest_population,
        s.latest_gdp_growth_pct,
        s.latest_health_score,

        -- Performance tier based on latest health score
        case
            when s.latest_health_score >= 70 then 'High Performer'
            when s.latest_health_score >= 45 then 'Mid Performer'
            when s.latest_health_score >= 20 then 'Developing'
            else 'Needs Attention'
        end as performance_tier,

        -- GDP rank (latest year)
        rank() over (order by s.latest_gdp_per_capita_usd desc nulls last) as rank_gdp_per_capita

    from historical h
    left join latest_snapshot s using (country_code)
)

select * from final
order by rank_gdp_per_capita
