# Regulated XAI and Model Risk Management Architecture

This reference documents architectural guardrails for Malaysian FSI and enterprise ML deployments on Google Cloud.

## Deprecation Avoidance Matrix

| Deprecated Pattern | Sunset / Risk | Replacement in `gcp-data-science` |
| --- | --- | --- |
| Agent Platform Explainable AI (`:explain`) (formerly Vertex AI) | Scheduled March 2027 deprecation | Agent Platform `:predict` + Custom inference routine TreeSHAP |
| Agent Platform Feature Store `optimized` online store | Scheduled February 2027 sunset | Bigtable-backed Feature Online Store (`auto_scaling`) |
| `google_notebooks_instance` | Legacy Agent Platform Workbench (formerly Vertex AI Workbench) | Managed Agent Platform Pipelines v2 and Cloud Run services |
| `asia-southeast2` Online Feature Store | Online serving unsupported in Jakarta | Precondition blocks Jakarta online store; use `asia-southeast1` |

## Dual Explainability Tracks (Track A and Track B)

### Track A: Batch BigQuery ML (`bqml_batch`)

- Uses `ML.FEATURES_AT_TIME` to build point-in-time training sets without target leakage.
- Trains `BOOSTED_TREE_CLASSIFIER` with `ENABLE_GLOBAL_EXPLAIN = TRUE` and `MODEL_REGISTRY = 'VERTEX_AI'`.
- Scores via `ML.EXPLAIN_PREDICT` with `APPROX_FEATURE_CONTRIB = FALSE` for exact TreeSHAP attributions.
- Writes `18` locked columns to `ml_xai_audit_ledger` and exports Track A JSON receipts to WORM GCS.

### Track B: Online Agent Platform CPR TreeSHAP (`vertex_cpr_online`)

- Serves private `:predict` endpoints over Private Service Connect (`ml_ops_project_id` and `ds_exp_project_id`).
- Locks `OMP_NUM_THREADS=1` inside `src/cpr_treeshap/predictor.py` for deterministic floating-point reduction.
- Verifies model artifact SHA-256 against the manifest before loading `xgboost.Booster`.
- Casts XGBoost `float32` margins and contributions to Python `float` (`FLOAT64`) before math checks.
- Enforces `abs((base_logit + shap_sum_logit) - raw_margin) <= tolerance * max(1.0, abs(raw_margin))`.
- Breaks equal-magnitude adverse ties alphabetically by feature name (`reason_rule_version = bnm-fair-lending-v1.0`).

## Seven-Year Immutable Audit Ledger Pattern (BNM RMiT and Act 873)

Regulated lending decisions under BNM RMiT and Financial Services Act 2013 (Act 873) require full auditability.

- `ml_xai_audit_ledger` partitions daily on `decision_timestamp` and uses CMEK encryption when `kms_key_id` is set.
- `FailClosedLedgerWriter` writes GCS WORM receipts first, then BigQuery Storage Write API `_default` streams before HTTP `200`.
- If WORM archival or BigQuery Storage Write fails, Cloud Run returns HTTP `503`.
- Enable `require_worm` in `prod` so empty `ml_worm_bucket_name` fails closed before any BigQuery commit.
- `vertex_predict_raw_tap` records 100% of raw `:predict` traffic for secondary reconciliation.
- Restricted PII keys (`customer_ic_number`, `nric`, `passport_number`) are rejected before inference and ledger writes.

## Locked 18-Column `ml_xai_audit_ledger` Contract

| Column | Type | Purpose |
| --- | --- | --- |
| `decision_id` | `STRING` | Unique decision UUID |
| `entity_id` | `STRING` | Scored customer identifier |
| `decision_timestamp` | `TIMESTAMP` | UTC decision timestamp |
| `serving_track` | `STRING` | `bqml_batch` or `vertex_cpr_online` |
| `model_id` | `STRING` | Registered model identifier |
| `model_version_id` | `STRING` | Deployed model version identifier |
| `model_artifact_sha256` | `STRING` | SHA-256 digest of model artifact |
| `container_image_digest` | `STRING` | Immutable container image digest |
| `schema_version` | `STRING` | Ledger payload schema version |
| `explained_class` | `STRING` | Target class explained (`APPROVE`) |
| `approve_prob` | `FLOAT64` | Calibrated approval probability |
| `raw_margin` | `FLOAT64` | Raw model log-odds margin |
| `base_logit` | `FLOAT64` | TreeSHAP baseline expected log-odds |
| `shap_sum_logit` | `FLOAT64` | Sum of exact TreeSHAP contributions |
| `shap_logit_map` | `JSON` | Per-feature exact TreeSHAP log-odds map |
| `top_adverse_reasons` | `JSON` | Ranked deterministic adverse-action codes |
| `feature_payload` | `JSON` | Non-PII feature vector evaluated at scoring |
| `reason_rule_version` | `STRING` | Versioned reason-code mapping rule ID |

## GenAI Deterministic Reason-Code Boundary

Downstream web portals and ADK agents must never recalculate credit scores or invent explanations.

- `format_adverse_reasons_for_agent` strips `feature_payload`, `shap_logit_map`, and `raw_margin` before agent exposure.
- Agents receive only `decision_id`, `explained_class`, `reason_rule_version`, and pre-approved `reason_code` text.
