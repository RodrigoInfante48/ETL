-- =============================================================================
-- File        : mv_03_latest_year_ranking.sql
-- Project     : World Bank ETL – world_bank_db
-- Schema      : public
-- Author      : Rod
-- Created     : 2026-02-22
-- Description : Materialized view – country rankings based on the most recent
--               year available in the dataset. Provides a snapshot of the
--               current macroeconomic standings across all indicators.
-- Standards   : PostgreSQL 14+, snake_case naming, CTEs for readability,
--               window functions with explicit PARTITION/ORDER, NULLS LAST.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Drop and recreate the materialized view
-- -----------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS mv_latest_year_ranking;

CREATE MATERIALIZED VIEW mv_latest_year_ranking AS

WITH latest_year AS (
    -- Resolve the most recent year available in the dataset
    SELECT MAX(year) AS max_year
    FROM economic_indicators
),

base AS (
    -- Filter to the latest year only
    SELECT
        ei.country_code,
        ei.country_name,
        ei.year,
        ei.gdp_current_usd,
        ei.gdp_per_capita_usd,
        ei.inflation_annual_pct,
        ei.unemployment_pct,
        ei.population_total
    FROM economic_indicators    ei
    JOIN latest_year            ly  ON ei.year = ly.max_year
)

SELECT
    country_code,
    country_name,
    year                                                              AS reference_year,

    -- Raw indicator values
    gdp_current_usd,
    gdp_per_capita_usd,
    inflation_annual_pct,
    unemployment_pct,
    population_total,

    -- Rankings (lower number = better position per convention below)
    RANK() OVER (ORDER BY gdp_current_usd      DESC NULLS LAST)     AS rank_gdp_total,
    RANK() OVER (ORDER BY gdp_per_capita_usd   DESC NULLS LAST)     AS rank_gdp_per_capita,
    RANK() OVER (ORDER BY inflation_annual_pct ASC  NULLS LAST)     AS rank_inflation,
    RANK() OVER (ORDER BY unemployment_pct     ASC  NULLS LAST)     AS rank_unemployment,
    RANK() OVER (ORDER BY population_total     DESC NULLS LAST)     AS rank_population

FROM base

ORDER BY
    rank_gdp_per_capita ASC NULLS LAST;

-- -----------------------------------------------------------------------------
-- Index for fast country lookups
-- -----------------------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uidx_mv_latest_year_ranking_code
    ON mv_latest_year_ranking (country_code);

-- -----------------------------------------------------------------------------
-- Refresh command (run after each ETL load)
-- REFRESH MATERIALIZED VIEW mv_latest_year_ranking;
-- =============================================================================
