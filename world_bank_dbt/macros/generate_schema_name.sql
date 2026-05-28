-- =============================================================================
-- Macro       : generate_schema_name
-- Purpose     : Separate dbt schemas by environment so dev runs never collide
--               with production tables.
--
-- Behaviour
-- ─────────
-- target.name = 'prod'  →  schema is the custom_schema_name as-is
--                           (staging, intermediate, marts)
-- target.name = anything else (dev, ci, …)
--                       →  schema is prefixed with the target schema
--                           (e.g. dbt_jsmith_staging, ci_staging)
--
-- This follows the official dbt recommendation:
-- https://docs.getdbt.com/docs/build/custom-schemas
--
-- Profile setup (profiles.yml)
-- ─────────────────────────────
-- production:
--   target: prod
--   outputs:
--     prod:
--       schema: analytics          # becomes staging / intermediate / marts
--
-- development:
--   target: dev
--   outputs:
--     dev:
--       schema: dbt_jsmith         # becomes dbt_jsmith_staging / etc.
--
-- CI:
--   target: ci
--   outputs:
--     ci:
--       schema: ci_{{ env_var('CI_BUILD_ID', 'local') }}
-- =============================================================================

{% macro generate_schema_name(custom_schema_name, node) -%}

    {%- set default_schema = target.schema -%}

    {%- if custom_schema_name is none -%}
        {# No custom schema configured — use the target schema unchanged #}
        {{ default_schema }}

    {%- elif target.name == 'prod' -%}
        {# Production: use the clean schema name without any prefix #}
        {{ custom_schema_name | trim }}

    {%- else -%}
        {# Non-production (dev, ci, staging, …): prefix to avoid collisions #}
        {{ default_schema }}_{{ custom_schema_name | trim }}

    {%- endif -%}

{%- endmacro %}
