-- Track A: BigQuery ML Boosted Tree + Exact TreeSHAP Audit Pipeline
-- Bind named query parameters before running in BigQuery:
--   @core_project_id, @silver_dataset_id, @feature_group_source_table,
--   @ml_ops_project_id, @ml_audit_dataset_id, @ml_audit_table_id,
--   @model_id, @model_version_id, @reason_rule_version

DECLARE feature_table_fqn STRING DEFAULT FORMAT(
  '%s.%s.%s',
  @core_project_id,
  @silver_dataset_id,
  @feature_group_source_table
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
    f.delinquency_count_12m,
    f.credit_utilization_ratio,
    f.debt_to_income_ratio,
    f.verified_income_band,
    f.approved_flag
  FROM ML.FEATURES_AT_TIME(
    TABLE `%s`,
    time => CURRENT_TIMESTAMP(),
    num_rows => 1,
    ignore_feature_nulls => FALSE
  ) AS f
  ''',
  model_fqn,
  feature_table_fqn
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
    CAST(p.customer_id AS STRING) AS entity_id,
    CURRENT_TIMESTAMP() AS decision_timestamp,
    'bqml_batch' AS serving_track,
    @model_id AS model_id,
    @model_version_id AS model_version_id,
    'bqml-managed' AS model_artifact_sha256,
    'bqml-native' AS container_image_digest,
    'v1' AS schema_version,
    'APPROVE' AS explained_class,
    CAST((SELECT prob FROM UNNEST(p.predicted_approved_flag_probs) WHERE label = 1) AS FLOAT64) AS approve_prob,
    LN(SAFE_DIVIDE(
      GREATEST(CAST((SELECT prob FROM UNNEST(p.predicted_approved_flag_probs) WHERE label = 1) AS FLOAT64), 1e-12),
      GREATEST(1.0 - CAST((SELECT prob FROM UNNEST(p.predicted_approved_flag_probs) WHERE label = 1) AS FLOAT64), 1e-12)
    )) AS raw_margin,
    CAST(p.baseline_prediction_value AS FLOAT64) AS base_logit,
    CAST((SELECT SUM(attr.attribution) FROM UNNEST(p.top_feature_attributions) AS attr) AS FLOAT64) AS shap_sum_logit,
    TO_JSON(ARRAY(
      SELECT AS STRUCT attr.feature, attr.attribution
      FROM UNNEST(p.top_feature_attributions) AS attr
    )) AS shap_logit_map,
    TO_JSON(ARRAY(
      SELECT AS STRUCT attr.feature, attr.attribution
      FROM UNNEST(p.top_feature_attributions) AS attr
      WHERE attr.attribution < 0
      ORDER BY attr.attribution ASC, attr.feature ASC
      LIMIT 4
    )) AS top_adverse_reasons,
    TO_JSON(STRUCT(
      p.tenure_months,
      p.avg_monthly_spend,
      p.delinquency_count_12m,
      p.credit_utilization_ratio,
      p.debt_to_income_ratio,
      p.verified_income_band
    )) AS feature_payload,
    @reason_rule_version AS reason_rule_version
  FROM ML.EXPLAIN_PREDICT(
    MODEL `%s`,
    (
      SELECT
        customer_id,
        tenure_months,
        avg_monthly_spend,
        delinquency_count_12m,
        credit_utilization_ratio,
        debt_to_income_ratio,
        verified_income_band
      FROM ML.FEATURES_AT_TIME(
        TABLE `%s`,
        time => CURRENT_TIMESTAMP(),
        num_rows => 1,
        ignore_feature_nulls => FALSE
      )
    ),
    STRUCT(27 AS top_k_features, FALSE AS approx_feature_contrib)
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
