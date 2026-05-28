-- =============================================================================
-- Model       : mart_latam_comparison
-- Layer       : Marts
-- Description : LATAM-focused analytical model. Compares the 7 Latin American
--               countries against each other and against the non-LATAM benchmark
--               (USA, China, Germany) across key economic indicators.
--               One row per country × year — designed for time-series dashboards
--               and country-ranking tables scoped to the LATAM region.
-- Materialized: table
-- =============================================================================

with metrics as (
    select * from {{ ref('int_country_metrics') }}
),

-- ─── LATAM cohort ─────────────────────────────────────────────────────────────
latam as (
    select * from metrics where is_latam = true
),

-- ─── Non-LATAM benchmark (USA, China, Germany) ────────────────────────────────
benchmark as (
    select
        year,
        round(avg(gdp_per_capita_usd), 2)    as benchmark_gdp_per_capita,
        round(avg(inflation_pct), 2)          as benchmark_inflation_pct,
        round(avg(unemployment_pct), 2)       as benchmark_unemployment_pct,
        round(avg(economic_health_score), 1)  as benchmark_health_score
    from metrics
    where is_latam = false
    group by year
),

-- ─── Intra-LATAM rankings per year ────────────────────────────────────────────
ranked as (
    select
        l.*,
        -- Rankings within LATAM only (dense_rank so ties don't skip a rank)
        dense_rank() over (partition by l.year order by l.gdp_per_capita_usd  desc nulls last) as latam_rank_gdp_per_capita,
        dense_rank() over (partition by l.year order by l.economic_health_score desc nulls last) as latam_rank_health_score,
        dense_rank() over (partition by l.year order by l.inflation_pct         asc  nulls last) as latam_rank_inflation,     -- lower is better
        dense_rank() over (partition by l.year order by l.unemployment_pct      asc  nulls last) as latam_rank_unemployment   -- lower is better
    from latam l
),

-- ─── Gap vs. non-LATAM benchmark ──────────────────────────────────────────────
final as (
    select
        r.country_code,
        r.country_name,
        r.year,

        -- Core indicators
        r.gdp_usd,
        r.gdp_per_capita_usd,
        r.inflation_pct,
        r.unemployment_pct,
        r.population,

        -- Derived metrics from intermediate layer
        r.gdp_growth_pct,
        r.gdp_per_capita_growth_pct,
        r.economic_health_score,

        -- Intra-LATAM rankings
        r.latam_rank_gdp_per_capita,
        r.latam_rank_health_score,
        r.latam_rank_inflation,
        r.latam_rank_unemployment,

        -- Gap vs. non-LATAM benchmark (positive = LATAM is above benchmark)
        round(r.gdp_per_capita_usd   - b.benchmark_gdp_per_capita,   2) as gap_gdp_per_capita_vs_benchmark,
        round(r.inflation_pct        - b.benchmark_inflation_pct,    2) as gap_inflation_vs_benchmark,
        round(r.unemployment_pct     - b.benchmark_unemployment_pct, 2) as gap_unemployment_vs_benchmark,
        round(r.economic_health_score - b.benchmark_health_score,    1) as gap_health_score_vs_benchmark,

        -- Benchmark reference values (for BI label display)
        b.benchmark_gdp_per_capita,
        b.benchmark_inflation_pct,
        b.benchmark_unemployment_pct,
        b.benchmark_health_score,

        -- Convergence flag: is this country closing the health-score gap YoY?
        case
            when round(r.economic_health_score - b.benchmark_health_score, 1)
                 > lag(round(r.economic_health_score - b.benchmark_health_score, 1))
                      over (partition by r.country_code order by r.year)
            then true
            else false
        end as is_converging_to_benchmark

    from ranked r
    left join benchmark b using (year)
)

select * from final
order by year, latam_rank_health_score
