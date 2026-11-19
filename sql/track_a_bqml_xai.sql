-- Track A: BigQuery ML Boosted Tree + Exact TreeSHAP Audit Pipeline
-- Bind named query parameters before running in BigQuery:
--   @core_project_id, @silver_dataset_id, @feature_group_source_table,
--   @ml_ops_project_id, @ml_audit_dataset_id, @ml_audit_table_id,
--   @ml_training_labels_table_id, @model_id, @model_version_id, @reason_rule_version

DECLARE feature_table_fqn STRING DEFAULT FORMAT(
  '%s.%s.%s',
  @core_project_id,
  @silver_dataset_id,
  @feature_group_source_table
);
DECLARE labels_table_fqn STRING DEFAULT FORMAT(
  '%s.%s.%s',
  @ml_ops_project_id,
  @ml_audit_dataset_id,
  @ml_training_labels_table_id
);
DECLARE model_fqn STRING DEFAULT FORMAT(
  '%s.%s.%s',
  @ml_ops_project_id,
  @ml_audit_dataset_id,
  @model_id
);
DECLARE ledger_fqn STRING DEFAULT FORMAT(
  '%s.%s.%s',
  @ml_ops_project_id,
  @ml_audit_dataset_id,
  @ml_audit_table_id
);

-- Step 1: Materialize point-in-time training features without future leakage.
EXECUTE IMMEDIATE FORMAT(
  '''
  CREATE OR REPLACE MODEL `%s`
  OPTIONS (
    model_type = 'BOOSTED_TREE_CLASSIFIER',
    input_label_cols = ['approved_flag'],
    enable_global_explain = TRUE,
    model_registry = 'VERTEX_AI',
    vertex_ai_model_id = @model_id,
    max_iterations = 50,
    learn_rate = 0.1
  ) AS
  SELECT
    f.tenure_months,
    f.avg_monthly_spend,
    f.rolling_order_count,
    f.days_since_last_order,
    f.cancellation_rate_30d,
    f.discount_utilization_ratio,
    f.approved_flag
  FROM ML.ENTITY_FEATURES_AT_TIME(
    (
      SELECT
        entity_id,
        feature_timestamp,
        tenure_months,
        avg_monthly_spend,
        rolling_order_count,
        days_since_last_order,
        cancellation_rate_30d,
        discount_utilization_ratio
      FROM `%s`
      WHERE feature_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY)
    ),
    (
      SELECT
        entity_id,
        label_timestamp AS time,
        approved_flag
      FROM `%s`
      WHERE DATE(label_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY)
    ),
    num_rows => 1,
    ignore_feature_nulls => FALSE
  ) AS f
  ''',
  model_fqn,
  feature_table_fqn,
  labels_table_fqn
)
USING @model_id AS model_id;

-- Step 2: Score and write exact TreeSHAP (APPROX_FEATURE_CONTRIB = FALSE) to ml_xai_audit_ledger.
EXECUTE IMMEDIATE FORMAT(
  '''
  INSERT INTO `%s` (
    decision_id,
    entity_id,
    decision_timestamp,
    serving_track,
    model_id,
    model_version_id,
    model_artifact_sha256,
    container_image_digest,
    schema_version,
    explained_class,
    approve_prob,
    raw_margin,
    base_logit,
    shap_sum_logit,
    shap_logit_map,
    top_adverse_reasons,
    feature_payload,
    reason_rule_version
  )
  SELECT
    GENERATE_UUID() AS decision_id,
    CAST(p.entity_id AS STRING) AS entity_id,
    CURRENT_TIMESTAMP() AS decision_timestamp,
    'bqml_batch' AS serving_track,
    @model_id AS model_id,
    @model_version_id AS model_version_id,
    'bqml-managed' AS model_artifact_sha256,
    'bqml-native' AS container_image_digest,
    'v1' AS schema_version,
    IF(CAST(p.predicted_approved_flag AS INT64) = 1, 'APPROVE', 'REJECT') AS explained_class,
    IF(
      CAST(p.predicted_approved_flag AS INT64) = 1,
      CAST(p.probability AS FLOAT64),
      1.0 - CAST(p.probability AS FLOAT64)
    ) AS approve_prob,
    CAST(p.prediction_value AS FLOAT64) AS raw_margin,
    CAST(p.baseline_prediction_value AS FLOAT64) AS base_logit,
    CAST(
      COALESCE(
        (SELECT SUM(attr.attribution) FROM UNNEST(p.top_feature_attributions) AS attr),
        0.0
      ) AS FLOAT64
    ) AS shap_sum_logit,
    COALESCE(
      (
        SELECT JSON_OBJECT(
          ARRAY_AGG(attr.feature ORDER BY attr.feature),
          ARRAY_AGG(attr.attribution ORDER BY attr.feature)
        )
        FROM UNNEST(p.top_feature_attributions) AS attr
      ),
      JSON '{}'
    ) AS shap_logit_map,
    TO_JSON(ARRAY(
      SELECT AS STRUCT
        CASE attr.feature
          WHEN 'tenure_months' THEN 'RC01'
          WHEN 'avg_monthly_spend' THEN 'RC02'
          WHEN 'rolling_order_count' THEN 'RC03'
          WHEN 'days_since_last_order' THEN 'RC04'
          WHEN 'cancellation_rate_30d' THEN 'RC05'
          WHEN 'discount_utilization_ratio' THEN 'RC06'
          ELSE 'RC99'
        END AS reason_code,
        CASE attr.feature
          WHEN 'tenure_months' THEN 'Short customer relationship tenure relative to underwriting policy'
          WHEN 'avg_monthly_spend' THEN 'Insufficient rolling 30-day net transaction volume'
          WHEN 'rolling_order_count' THEN 'Low verified order frequency in trailing 30-day window'
          WHEN 'days_since_last_order' THEN 'Extended account inactivity since last completed transaction'
          WHEN 'cancellation_rate_30d' THEN 'Elevated 30-day order cancellation or reversal rate'
          WHEN 'discount_utilization_ratio' THEN 'High promotional discount dependency relative to gross spend'
          ELSE CONCAT('Adverse model attribution on ', attr.feature)
        END AS reason_text,
        attr.feature AS feature,
        attr.attribution AS attribution
      FROM UNNEST(p.top_feature_attributions) AS attr
      WHERE IF(CAST(p.predicted_approved_flag AS INT64) = 1, attr.attribution, -attr.attribution) < 0
      ORDER BY IF(CAST(p.predicted_approved_flag AS INT64) = 1, attr.attribution, -attr.attribution) ASC, attr.feature ASC
      LIMIT 4
    )) AS top_adverse_reasons,
    TO_JSON(STRUCT(
      p.tenure_months,
      p.avg_monthly_spend,
      p.rolling_order_count,
      p.days_since_last_order,
      p.cancellation_rate_30d,
      p.discount_utilization_ratio
    )) AS feature_payload,
    @reason_rule_version AS reason_rule_version
  FROM ML.EXPLAIN_PREDICT(
    MODEL `%s`,
    (
      SELECT
        entity_id,
        tenure_months,
        avg_monthly_spend,
        rolling_order_count,
        days_since_last_order,
        cancellation_rate_30d,
        discount_utilization_ratio
      FROM ML.FEATURES_AT_TIME(
        (
          SELECT
            entity_id,
            feature_timestamp,
            tenure_months,
            avg_monthly_spend,
            rolling_order_count,
            days_since_last_order,
            cancellation_rate_30d,
            discount_utilization_ratio
          FROM `%s`
          WHERE feature_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY)
        ),
        time => CURRENT_TIMESTAMP(),
        num_rows => 1,
        ignore_feature_nulls => FALSE
      )
    ),
    STRUCT(100 AS top_k_features, FALSE AS approx_feature_contrib)
  ) AS p
  ''',
  ledger_fqn,
  model_fqn,
  feature_table_fqn
)
USING
  @model_id AS model_id,
  @model_version_id AS model_version_id,
  @reason_rule_version AS reason_rule_version;
