-- Cross-project ref into country_analytics (not core_economics — reporting_layer
-- never skips a layer). This preserves clear domain ownership: country_analytics
-- is responsible for the mart contract; reporting_layer only adds display logic.

with country_data as (
    select * from {{ ref('country_analytics', 'mart_country_summary') }}
),

formatted as (
    select
        country_code,
        country_name,
        is_latam,
        performance_tier,
        rank_gdp_per_capita,
        latest_health_score,
        latest_gdp_per_capita_usd,
        latest_inflation_pct,
        latest_unemployment_pct,
        avg_gdp_growth_pct,

        -- Email display columns
        case is_latam
            when true  then 'LATAM'
            else            'Global'
        end as region_label,

        case performance_tier
            when 'High Performer' then '🟢'
            when 'Mid Performer'  then '🟡'
            when 'Developing'     then '🟠'
            else                       '🔴'
        end as tier_indicator,

        round(latest_gdp_per_capita_usd / 1000.0, 1) as gdp_per_capita_k

    from country_data
)

select * from formatted
order by rank_gdp_per_capita
