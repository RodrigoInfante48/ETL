-- Singular test: LATAM countries must never have a null gdp_usd.
-- The 7 LATAM countries (CO, BR, MX, AR, CL, PE, EC) are the analytical core;
-- a null GDP for any of them would silently corrupt country summaries and
-- LATAM-vs-global comparisons.
-- Returns violating rows; dbt fails if any rows are returned.

select
    country_code,
    country_name,
    year,
    gdp_usd
from {{ ref('int_country_metrics') }}
where is_latam = true
  and gdp_usd is null
