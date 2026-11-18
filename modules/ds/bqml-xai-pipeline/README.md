# BQML XAI Pipeline Module

Provisions the Track A BigQuery ML workflow and Cloud Scheduler trigger.

## Pipeline Behavior

- Trains `BOOSTED_TREE_CLASSIFIER` with `ENABLE_GLOBAL_EXPLAIN = TRUE` and `MODEL_REGISTRY = 'VERTEX_AI'`.
- Executes `ML.EXPLAIN_PREDICT` with `APPROX_FEATURE_CONTRIB = FALSE` for exact TreeSHAP values.
- Inserts 18-column audit rows directly into `ml_xai_audit_ledger`.
- Publishes failure notifications to a regional Pub/Sub topic and Cloud Monitoring policy.

## Inputs and Outputs

See [variables.tf](variables.tf) and [outputs.tf](outputs.tf) for all module settings.
