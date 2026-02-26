-- =============================================================================
-- File        : mv_01_country_profile.sql
-- Project     : World Bank ETL – world_bank_db
-- Schema      : public
-- Author      : Rod
-- Created     : 2026-02-22
-- Description : Materialized view – macroeconomic profile aggregated by country.
--               Includes an optional temporal filter (start_year) to avoid
--               loading the full historical dataset when not required.
-- Standards   : PostgreSQL 14+, snake_case naming, explicit column aliases,
--               NULLS LAST on ORDER BY, no SELECT *.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Optional: set the earliest year to include in the analysis.
-- Adjust the value below before running (e.g. 2000, 2010, 2015).
-- To include all available data, set the value to 0.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_settings WHERE name = 'app.start_year'
    ) THEN
        PERFORM set_config('app.start_year', '2000', FALSE);
    END IF;
END $$;

-- -----------------------------------------------------------------------------
-- Drop and recreate the materialized view
-- -----------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS mv_country_profile;

CREATE MATERIALIZED VIEW mv_country_profile AS

SELECT
    country_code,
    country_name,

    -- Coverage
    COUNT(*)                                              AS years_reported,
    MIN(year)                                             AS first_year,
    MAX(year)                                             AS last_year,

    -- GDP (USD)
    ROUND(AVG(gdp_current_usd)::NUMERIC, 2)              AS avg_gdp_usd,
    ROUND(MIN(gdp_current_usd)::NUMERIC, 2)              AS min_gdp_usd,
    ROUND(MAX(gdp_current_usd)::NUMERIC, 2)              AS max_gdp_usd,

    -- GDP per capita (USD)
    ROUND(AVG(gdp_per_capita_usd)::NUMERIC, 2)           AS avg_gdp_per_capita_usd,
    ROUND(MIN(gdp_per_capita_usd)::NUMERIC, 2)           AS min_gdp_per_capita_usd,
    ROUND(MAX(gdp_per_capita_usd)::NUMERIC, 2)           AS max_gdp_per_capita_usd,

    -- Inflation (%)
    ROUND(AVG(inflation_annual_pct)::NUMERIC, 2)         AS avg_inflation_pct,
    ROUND(MIN(inflation_annual_pct)::NUMERIC, 2)         AS min_inflation_pct,
    ROUND(MAX(inflation_annual_pct)::NUMERIC, 2)         AS max_inflation_pct,

    -- Unemployment (%)
    ROUND(AVG(unemployment_pct)::NUMERIC, 2)             AS avg_unemployment_pct,
    ROUND(MIN(unemployment_pct)::NUMERIC, 2)             AS min_unemployment_pct,
    ROUND(MAX(unemployment_pct)::NUMERIC, 2)             AS max_unemployment_pct,

    -- Population
    ROUND(AVG(population_total)::NUMERIC, 0)             AS avg_population,
    ROUND(MAX(population_total)::NUMERIC, 0)             AS max_population

FROM economic_indicators

-- Temporal filter: exclude data before the configured start year
WHERE year >= COALESCE(
    NULLIF(current_setting('app.start_year', TRUE), '')::INTEGER,
    2000
)

GROUP BY
    country_code,
    country_name

ORDER BY
    avg_gdp_usd DESC NULLS LAST;

-- -----------------------------------------------------------------------------
-- Index for fast lookups by country
-- -----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uidx_mv_country_profile_code
    ON mv_country_profile (country_code);

-- -----------------------------------------------------------------------------
-- Refresh command (run after each ETL load)
-- REFRESH MATERIALIZED VIEW mv_country_profile;
-- =============================================================================
