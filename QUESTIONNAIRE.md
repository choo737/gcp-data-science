# Data Science Intake Questionnaire

Complete this questionnaire before deploying Layer 04 for Greenfield, Brownfield, or Sandbox targets.

You can import `platform-context.json` from [gcp-launchpad](https://github.com/choo737/gcp-launchpad) instead of entering values manually.

Never commit real project IDs, KMS keys, bucket names, or model secrets to tracked files.

Record answers in `stages/enterprise/envs/<env>.tfvars`, `stages/sandbox/terraform.tfvars`, or gitignored `00_linked.auto.tfvars.json`.

Valid enterprise environments are `dev`, `uat`, and `prod`.

## Handoff From Launchpad

If `platform-context.json` exists from [gcp-launchpad](https://github.com/choo737/gcp-launchpad), run the Brownfield bridge to write `00_linked.auto.tfvars.json` (`0600`).

```bash
python3 ../gcp-launchpad/scripts/generate-brownfield-link.py \
  --context ../gcp-launchpad/platform-context.json \
  --target data-science \
  --env dev \
  --output stages/enterprise/00_linked.auto.tfvars.json
```

| `platform-context.json` Field | Target `00_linked.auto.tfvars.json` Key | Purpose |
| --- | --- | --- |
| `existing_landing_zone.core_project_id` | `core_project_id` | Project hosting BigQuery Silver feature tables |
| `existing_landing_zone.ml_ops_project_id` | `ml_ops_project_id` | Project hosting Agent Platform Feature Store and Scorer |
| `existing_landing_zone.silver_dataset_id` | `silver_dataset_id` | Silver dataset hosting `vw_ml_customer_features_online` |
| `existing_landing_zone.ml_audit_dataset_id` | `ml_audit_dataset_id` | Dataset hosting `ml_xai_audit_ledger` |

## Deterministic Agent Output Routing

Map each questionnaire section to the exact Terraform file below:

| Questionnaire Section | Target File | Mode |
| --- | --- | --- |
| Section 1 (Greenfield / Brownfield) | `stages/enterprise/00_linked.auto.tfvars.json` | Linked upstream outputs (`0600`) |
| Section 1 (Sandbox) | `stages/sandbox/terraform.tfvars` | Single-project sandbox inputs |
| Sections 2 and 3 | `stages/enterprise/envs/<env>.tfvars` | Tracked `dev`, `uat`, `prod` |

### Company Acronym Rule (`prefix`)

Derive `prefix` as a `3-6` lowercase char company acronym from the organization name.

- Strip legal suffixes (`Bhd`, `Sdn`, `Pte`, `Ltd`, `Inc`, `Corp`) when three core words remain.
- Example: `'Example Global Retail Ltd'` -> `egr`, `'Sample Financial Group'` -> `sfg`.
- Confirm the suggested `prefix` with the user before writing `.tfvars` files.

## Section 1: Upstream Landing Zone Mode

Select one deployment mode before linking upstream projects or running `terraform plan`.

| Mode | Stage Directory | Upstream Source | Output File |
| --- | --- | --- | --- |
| Greenfield Enterprise | `stages/enterprise/` | `gcp-foundation` local state or GCS export | `stages/enterprise/00_linked.auto.tfvars.json` |
| Brownfield Enterprise | `stages/enterprise/` | Existing customer projects, KMS, and VPC | `stages/enterprise/00_linked.auto.tfvars.json` |
| Single-Project Sandbox | `stages/sandbox/` | Existing sandbox project | `stages/sandbox/terraform.tfvars` |

### Greenfield Enterprise Intake

Run `scripts/link-stages.sh` to import non-sensitive outputs from `gcp-foundation` and `gcp-data-medallion`.

```bash
./scripts/link-stages.sh --env dev \
  --from-local ../gcp-foundation/data/enterprise/03-data-science \
  ../gcp-foundation/foundation/02-security-network \
  ../gcp-foundation/data/enterprise/01-platform \
  ../gcp-foundation/data/enterprise/02-governance-ops \
  stages/enterprise

# Or import outputs exported earlier to GCS:
./scripts/link-stages.sh --env dev --from-gcs gs://<outputs-bucket>/foundation-outputs.json stages/enterprise
```

### Brownfield Enterprise Intake (Without `gcp-foundation`)

Populate `stages/enterprise/00_linked.auto.tfvars.json` directly or via `scripts/generate-brownfield-link.py --target data-science`.

| Variable Name | Requirement | Description | Stage Alias |
| --- | --- | --- | --- |
| `core_project_id` | Required | Project hosting BigQuery Silver feature tables | `core_storage_project_id` |
| `ds_exp_project_id` | Required | Data science experimentation project allowlisted for PSC | — |
| `ml_ops_project_id` | Required | Project hosting Agent Platform (formerly Vertex AI) Feature Store, Endpoint, and Scorer | — |
| `silver_dataset_id` | Required | Silver dataset hosting `vw_ml_customer_features_online` | — |
| `ml_audit_dataset_id` | Required | BigQuery dataset hosting `ml_xai_audit_ledger` | — |
| `ml_artifact_repo_id` | Required | Artifact Registry repository storing CPR and scorer images | — |
| `kms_key_id` | Optional | Regional Cloud KMS CryptoKey ID for CMEK | `kms_key_name` |
| `vpc_id` | Optional | Existing VPC network resource ID | `network_id` |
| `egress_connector_subnet_id` | Optional | Existing subnet ID for Cloud Run Direct VPC Egress | `subnet_id` |
| `ml_worm_bucket_name` | Optional | Immutable GCS WORM bucket for audit receipts | — |

## Section 2: Serving Track and Agent Platform Feature Store

Choose Track A, Track B, or both inside `stages/enterprise/envs/<env>.tfvars`.

| Setting | Variable Name | Default | Guidance |
| --- | --- | --- | --- |
| Serving Track | `ds_serving_track` | `"both"` | Choose `bqml_batch`, `vertex_cpr_online`, or `both` |
| Online Store Toggle | `enable_online_feature_store` | `false` | Uses Bigtable autoscaling; blocked in `asia-southeast2` |
| Online Store Nodes | `feature_online_store_min_nodes` | `1` | Scale up to `feature_online_store_max_nodes` (`2`) |
| Feature Sync Cron | `feature_sync_cron` | `"TZ=Asia/Singapore 0 * * * *"` | Cron schedule syncing FeatureView from BigQuery |
| Source Feature Table | `feature_group_source_table` | `"vw_ml_customer_features_online"` | Silver view with `entity_id` key |

## Section 3: Regulated XAI, Scorer, and MRM Thresholds

Confirm model governance thresholds before deploying to `uat` and `prod`.

| Control | Variable Name | Default | Guidance |
| --- | --- | --- | --- |
| Scorer Service | `enable_scorer_service` | `true` | Deploys internal Cloud Run v2 fail-closed scorer |
| Scorer Image | `scorer_container_image` | `""` | Must use `@sha256` digest in `prod` |
| CPR Model Toggle | `deploy_cpr_model` | `false` | Set `true` once `cpr_container_image` is built |
| CPR Image | `cpr_container_image` | `""` | Must use `@sha256` digest in `prod` |
| SHAP Tolerance | `shap_additivity_tolerance` | `0.00001` | Fail-closed threshold for `base_logit + shap_sum_logit` drift |
| Fairness Floor | `fairness_min_disparate_impact` | `0.80` | Minimum subgroup disparate-impact ratio for MRM gate |
| BQML Schedule | `enable_bqml_schedule` | `true` | Triggers Track A `ML.EXPLAIN_PREDICT` workflow |
| BQML Cron | `bqml_schedule_cron` | `"0 2 * * *"` | Daily batch training and scoring schedule |
