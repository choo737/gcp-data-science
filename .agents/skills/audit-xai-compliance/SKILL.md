---
name: audit-xai-compliance
description: Audits regulated XAI invariants including TreeSHAP additivity, 18-column ledger schema, MRM fairness gate, and zero deprecated Vertex configs.
---

# Audit XAI Compliance

Run this audit before approving the `model-validator` GitHub environment for `prod` deployment.

## Step 1: Verify Zero Deprecated Services

Confirm no Terraform or Python file uses deprecated Vertex AI configurations:

- Reject any occurrence of `explanation_spec` across `.tf` and `.py` files.
- Reject any occurrence of `google_notebooks_instance` across `.tf` and `.py` files.
- Reject any `optimized` block inside `google_vertex_ai_feature_online_store`.
- Confirm `enable_online_feature_store` is blocked in `asia-southeast2`.

## Step 2: Verify Fail-Closed Audit Persistence

Confirm Track B online scoring cannot release an unaudited credit decision:

1. Verify `src/cpr_treeshap/predictor.py` locks `OMP_NUM_THREADS=1` and casts `float32` outputs to Python `float`.
2. Verify `TreeShapCprPredictor` raises `ValueError` when `abs((base_logit + shap_sum_logit) - raw_margin)` breaches tolerance.
3. Verify `FailClosedLedgerWriter.commit_row_sync` writes to a `COMMITTED` Storage Write API stream before returning.
4. Verify `ScoringService.score_applicant` returns HTTP `503` when `LedgerCommitError` occurs.
5. Verify restricted PII keys (`customer_ic_number`, `nric`, `passport_number`) trigger HTTP `400` or `ValueError`.

## Step 3: Verify 18-Column Ledger and Deterministic Agent Boundary

Check `stages/enterprise/main.tf` and `sql/track_a_bqml_xai.sql` against the locked 18 columns:

- `decision_id`, `entity_id`, `decision_timestamp`, `serving_track`, `model_id`, `model_version_id`
- `model_artifact_sha256`, `container_image_digest`, `schema_version`, `explained_class`
- `approve_prob`, `raw_margin`, `base_logit`, `shap_sum_logit`
- `shap_logit_map`, `top_adverse_reasons`, `feature_payload`, `reason_rule_version`

Verify downstream webapps and ADK agents call `format_adverse_reasons_for_agent` so raw logits remain hidden.

## Step 4: Run Hermetic Suite and MRM Gate

```bash
make fmt-check
make validate
make test
python3 -m src.mrm_validation.evaluate_gate --input-json <eval_payload.json> --output-card model_card.json
```

Confirm `model_card.json` reports `"approved": true` and `disparate_impact_ratio >= 0.80` before approving `prod`.
