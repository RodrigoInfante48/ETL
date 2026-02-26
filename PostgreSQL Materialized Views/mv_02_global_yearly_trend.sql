-- =============================================================================
-- File        : mv_02_global_yearly_trend.sql
-- Project     : World Bank ETL – world_bank_db
-- Schema      : public
-- Author      : Rod
-- Created     : 2026-02-22
-- Description : Materialized view – global macroeconomic trend aggregated by
--               year. Useful for time-series visualizations and YoY analysis.
--               Includes a temporal filter to restrict the analysis window.
-- Standards   : PostgreSQL 14+, snake_case naming, explicit column aliases,
--               NULLS LAST on ORDER BY, no SELECT *.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Drop and recreate the materialized view
-- -----------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS mv_global_yearly_trend;

CREATE MATERIALIZED VIEW mv_global_yearly_trend AS

SELECT
    year,

    -- Coverage per year
    COUNT(DISTINCT country_code)                          AS countries_count,

    -- GDP aggregates (USD)
    ROUND(SUM(gdp_current_usd)::NUMERIC, 2)              AS world_gdp_total_usd,
    ROUND(AVG(gdp_current_usd)::NUMERIC, 2)              AS avg_gdp_usd,
    ROUND(AVG(gdp_per_capita_usd)::NUMERIC, 2)           AS avg_gdp_per_capita_usd,

    -- Inflation (%)
    ROUND(AVG(inflation_annual_pct)::NUMERIC, 2)         AS avg_inflation_pct,
    ROUND(MIN(inflation_annual_pct)::NUMERIC, 2)         AS min_inflation_pct,
    ROUND(MAX(inflation_annual_pct)::NUMERIC, 2)         AS max_inflation_pct,

    -- Unemployment (%)
    ROUND(AVG(unemployment_pct)::NUMERIC, 2)             AS avg_unemployment_pct,
    ROUND(MIN(unemployment_pct)::NUMERIC, 2)             AS min_unemployment_pct,
    ROUND(MAX(unemployment_pct)::NUMERIC, 2)             AS max_unemployment_pct,

    -- Population
    ROUND(SUM(population_total)::NUMERIC, 0)             AS world_population_total,
    ROUND(AVG(population_total)::NUMERIC, 0)             AS avg_population_per_country

FROM economic_indicators

-- Temporal filter: restrict analysis to data from a given year onward
WHERE year >= COALESCE(
    NULLIF(current_setting('app.start_year', TRUE), '')::INTEGER,
    2000
)

GROUP BY
    year

ORDER BY
    year ASC;

-- -----------------------------------------------------------------------------
-- Index for fast time-series filtering
-- -----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uidx_mv_global_yearly_trend_year
    ON mv_global_yearly_trend (year);

-- -----------------------------------------------------------------------------
-- Refresh command (run after each ETL load)
-- REFRESH MATERIALIZED VIEW mv_global_yearly_trend;
-- =============================================================================
