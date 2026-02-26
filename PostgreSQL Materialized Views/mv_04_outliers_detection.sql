-- =============================================================================
-- File        : mv_04_outliers_detection.sql
-- Project     : World Bank ETL – world_bank_db
-- Schema      : public
-- Author      : Rod
-- Created     : 2026-02-22
-- Description : Materialized view – statistical outlier detection using the
--               Z-score method (threshold: |z| > 2). Flags countries/years
--               with extreme values across key macroeconomic indicators.
--               Includes temporal filter to focus on relevant time windows.
-- Standards   : PostgreSQL 14+, snake_case naming, CTEs for readability,
--               explicit CAST, NULLS LAST, no magic numbers in final SELECT.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Drop and recreate the materialized view
-- -----------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS mv_outliers_detection;

CREATE MATERIALIZED VIEW mv_outliers_detection AS

WITH temporal_filter AS (
    -- Apply start year filter to narrow the analysis window
    SELECT *
    FROM economic_indicators
    WHERE year >= COALESCE(
        NULLIF(current_setting('app.start_year', TRUE), '')::INTEGER,
        2000
    )
),

global_stats AS (
    -- Compute global mean and standard deviation per indicator
    SELECT
        AVG(gdp_per_capita_usd)         AS gdp_pc_mean,
        STDDEV(gdp_per_capita_usd)      AS gdp_pc_std,

        AVG(inflation_annual_pct)       AS inflation_mean,
        STDDEV(inflation_annual_pct)    AS inflation_std,

        AVG(unemployment_pct)           AS unemployment_mean,
        STDDEV(unemployment_pct)        AS unemployment_std,

        AVG(population_total)           AS population_mean,
        STDDEV(population_total)        AS population_std
    FROM temporal_filter
),

z_scores AS (
    -- Calculate Z-score for each record and indicator
    SELECT
        tf.country_code,
        tf.country_name,
        tf.year,
        tf.gdp_current_usd,
        tf.gdp_per_capita_usd,
        tf.inflation_annual_pct,
        tf.unemployment_pct,
        tf.population_total,

        -- Z-scores
        ROUND(
            ((tf.gdp_per_capita_usd - gs.gdp_pc_mean)
             / NULLIF(gs.gdp_pc_std, 0))::NUMERIC, 4
        )                                           AS z_gdp_per_capita,

        ROUND(
            ((tf.inflation_annual_pct - gs.inflation_mean)
             / NULLIF(gs.inflation_std, 0))::NUMERIC, 4
        )                                           AS z_inflation,

        ROUND(
            ((tf.unemployment_pct - gs.unemployment_mean)
             / NULLIF(gs.unemployment_std, 0))::NUMERIC, 4
        )                                           AS z_unemployment,

        ROUND(
            ((tf.population_total - gs.population_mean)
             / NULLIF(gs.population_std, 0))::NUMERIC, 4
        )                                           AS z_population

    FROM temporal_filter    tf
    CROSS JOIN global_stats gs
)

SELECT
    country_code,
    country_name,
    year,

    -- Raw values
    gdp_current_usd,
    gdp_per_capita_usd,
    inflation_annual_pct,
    unemployment_pct,
    population_total,

    -- Z-scores
    z_gdp_per_capita,
    z_inflation,
    z_unemployment,
    z_population,

    -- Outlier flags (|z| > 2 = outlier)
    CASE WHEN ABS(z_gdp_per_capita) > 2 THEN 'OUTLIER' ELSE 'NORMAL' END   AS flag_gdp_per_capita,
    CASE WHEN ABS(z_inflation)      > 2 THEN 'OUTLIER' ELSE 'NORMAL' END   AS flag_inflation,
    CASE WHEN ABS(z_unemployment)   > 2 THEN 'OUTLIER' ELSE 'NORMAL' END   AS flag_unemployment,
    CASE WHEN ABS(z_population)     > 2 THEN 'OUTLIER' ELSE 'NORMAL' END   AS flag_population,

    -- Summary: total outlier flags per row
    (
        (CASE WHEN ABS(z_gdp_per_capita) > 2 THEN 1 ELSE 0 END) +
        (CASE WHEN ABS(z_inflation)      > 2 THEN 1 ELSE 0 END) +
        (CASE WHEN ABS(z_unemployment)   > 2 THEN 1 ELSE 0 END) +
        (CASE WHEN ABS(z_population)     > 2 THEN 1 ELSE 0 END)
    )                                                                        AS total_outlier_flags

FROM z_scores

ORDER BY
    total_outlier_flags DESC,
    year                DESC,
    country_code        ASC;

-- -----------------------------------------------------------------------------
-- Indexes for filtered queries on outlier flags
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_mv_outliers_detection_country_year
    ON mv_outliers_detection (country_code, year);

CREATE INDEX IF NOT EXISTS idx_mv_outliers_detection_flags
    ON mv_outliers_detection (total_outlier_flags);

-- -----------------------------------------------------------------------------
-- Refresh command (run after each ETL load)
-- REFRESH MATERIALIZED VIEW mv_outliers_detection;
-- =============================================================================
