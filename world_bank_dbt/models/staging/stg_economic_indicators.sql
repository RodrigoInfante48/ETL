-- =============================================================================
-- Model       : stg_economic_indicators
-- Layer       : Staging
-- Description : Cleans and standardizes raw economic_indicators data.
-- Materialized: view
-- =============================================================================

with source as (
    select * from {{ source('world_bank', 'economic_indicators') }}
),

cleaned as (
    select
        country_code,
        country_name,
        cast(year as integer)                   as year,
        cast(gdp_current_usd as numeric)        as gdp_usd,
        cast(gdp_per_capita_usd as numeric)     as gdp_per_capita_usd,
        cast(inflation_annual_pct as numeric)   as inflation_pct,
        cast(unemployment_pct as numeric)       as unemployment_pct,
        cast(population_total as numeric)       as population
    from source
    where country_code is not null
      and year is not null
      and gdp_current_usd is not null
)

select * from cleaned
