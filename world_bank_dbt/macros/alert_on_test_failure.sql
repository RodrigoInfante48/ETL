{% macro alert_on_test_failure() %}
  {#
    on-run-end hook — runs after `dbt test` or `dbt build`.

    In dbt Cloud, you can configure email/Slack alerts in the UI under
    Account Settings → Notifications without writing any code. Use this macro
    when you need custom logic: filtering by severity, calling an external
    webhook, or enriching the message with business context.

    USAGE in dbt_project.yml:
        on-run-end:
          - "{{ alert_on_test_failure() }}"
  #}
  {% if execute %}

    {% set ns = namespace(error_tests=[], warn_tests=[]) %}

    {% for result in results %}
      {% if result.node.resource_type == 'test' %}
        {% if result.status in ('fail', 'error') %}
          {% set ns.error_tests = ns.error_tests + [result.node.name] %}
        {% elif result.status == 'warn' %}
          {% set ns.warn_tests = ns.warn_tests + [result.node.name] %}
        {% endif %}
      {% endif %}
    {% endfor %}

    {% if ns.error_tests | length > 0 %}
      {{ log("", info=True) }}
      {{ log("❌  ALERT — " ~ ns.error_tests | length ~ " test(s) FAILED:", info=True) }}
      {% for t in ns.error_tests %}
        {{ log("    • " ~ t, info=True) }}
      {% endfor %}
      {{ log("", info=True) }}
      {#
        To send a Slack notification, replace the log() calls above with an
        HTTP request via the `run_query` adapter or a Python script called
        via `post_hook`. Example (dbt Cloud env var: DBT_SLACK_WEBHOOK_URL):

        {% set payload = '{"text": "❌ dbt tests failed: ' ~ ns.error_tests | join(', ') ~ '"}' %}
        {% do run_query("select http_post('" ~ env_var('DBT_SLACK_WEBHOOK_URL') ~ "', '" ~ payload ~ "')") %}
      #}
    {% elif ns.warn_tests | length > 0 %}
      {{ log("", info=True) }}
      {{ log("⚠️   WARN — " ~ ns.warn_tests | length ~ " test(s) raised warnings:", info=True) }}
      {% for t in ns.warn_tests %}
        {{ log("    • " ~ t, info=True) }}
      {% endfor %}
      {{ log("", info=True) }}
    {% else %}
      {{ log("✅  All tests passed.", info=True) }}
    {% endif %}

  {% endif %}
{% endmacro %}
