-- Singular test: economic_health_score must be within [0, 100].
-- Returns rows that violate the constraint; dbt fails if any rows are returned.
-- The score formula in int_country_metrics has three additive components
-- (GDP 50pts + inflation 25pts + unemployment 25pts) each clamped individually,
-- but floating-point arithmetic or edge inputs could theoretically exceed bounds.

select
    country_code,
    country_name,
    year,
    economic_health_score
from {{ ref('int_country_metrics') }}
where economic_health_score < 0
   or economic_health_score > 100
