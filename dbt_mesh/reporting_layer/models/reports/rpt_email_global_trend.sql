with trend_data as (
    select * from {{ ref('country_analytics', 'mart_global_trend') }}
),

with_annotations as (
    select
        year,
        world_gdp_usd,
        round(world_gdp_usd / 1e12, 2)             as world_gdp_trillions,
        world_gdp_growth_pct,
        avg_inflation_pct,
        avg_unemployment_pct,
        world_population,
        avg_health_score,

        -- Economic regime label for email narrative section
        case
            when world_gdp_growth_pct < 0      then 'Contraction'
            when world_gdp_growth_pct < 2.0    then 'Slow Growth'
            when world_gdp_growth_pct < 5.0    then 'Moderate Growth'
            else                                    'Strong Growth'
        end as economic_regime,

        -- Notable events for highlighted callout boxes in HTML email
        case
            when year = 2009                        then 'Global Financial Crisis'
            when year = 2020                        then 'COVID-19 Shock'
            when world_gdp_growth_pct > 7.0         then 'Exceptional Recovery Year'
            else null
        end as notable_event

    from trend_data
)

select * from with_annotations
order by year
