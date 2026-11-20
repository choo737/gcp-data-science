# Repository Agent Instructions

This repository holds Terraform modules, stages, SQL scripts, and Python services for regulated machine learning.

It pairs with [gcp-launchpad](https://github.com/choo737/gcp-launchpad).
It deploys onto landing zones from [gcp-foundation](https://github.com/choo737/gcp-foundation).
It reads Silver data from [gcp-data-ingestion](https://github.com/choo737/gcp-data-ingestion).
It consumes conformed tables from [gcp-data-medallion](https://github.com/choo737/gcp-data-medallion).

## Repository Layout

- `modules/ds/`: Reusable modules (`feature-store`, `xai-serving-endpoint`, `bqml-xai-pipeline`).
- `stages/sandbox/`: Single-project stage with GCS backend and `.terraform.lock.hcl`.
- `stages/enterprise/`: Multi-project stage with `envs/{dev,uat,staging,prod}.tfvars.example` and `.terraform.lock.hcl`.
- `src/cpr_treeshap/`: Custom Prediction Routine container computing exact `OMP_NUM_THREADS=1` TreeSHAP attributions.
- `src/scorer_service/`: Fail-closed Cloud Run v2 scorer committing rows before decision release.
- `src/mrm_validation/`: Mathematical MRM challenger gate calculating ROC-AUC, KS, fairness, and TreeSHAP parity.
- `src/pipelines/`: Vertex AI Pipelines v2 training and registry DAG.
- `sql/`: Parameterized Track A BigQuery ML training and `ML.EXPLAIN_PREDICT` SQL.
- `workflows/`: Cloud Workflows GA `v1` definition for Track A runs.
- `scripts/link-stages.sh`: Links local or GCS upstream outputs into `00_linked.auto.tfvars.json`.
- `QUESTIONNAIRE.md`: Greenfield, Brownfield, and Sandbox intake questionnaire.
- `docs/`: Six-step onboarding guide and regulated XAI architecture reference.

## Interactive Setup Protocol

Load `.agents/skills/setup-data-science/SKILL.md` before deploying this repository for a customer.

1. **Create a private repo first**: Never fork publicly; run `gh repo create --private` and verify `isPrivate`.
2. **Collect intake inputs**: Check `platform-context.json` from [gcp-launchpad](https://github.com/choo737/gcp-launchpad) or complete `QUESTIONNAIRE.md`.
3. **Suggest company acronym**: Derive a `3-6` char `prefix` from core org words (`'Example Global Retail Ltd'` -> `egr`).
4. Confirm the suggested `prefix` with the user before writing `.tfvars` files.
3. **Verify GCP auth**: Check `gcloud config get-value account` or `GOOGLE_OAUTH_ACCESS_TOKEN`.
4. **Link Greenfield or Brownfield outputs**: Run `scripts/link-stages.sh --env <env>` or populate `00_linked.auto.tfvars.json`.
5. **Audit XAI compliance**: Run `.agents/skills/audit-xai-compliance/SKILL.md` before promoting to `prod`.

## Hard Contracts

1. Every directory under `modules/ds/*` must contain only `README.md`, `main.tf`, `variables.tf`, `outputs.tf`, and `versions.tf`.
2. Every stage under `stages/*` must contain only those five files plus `terraform.tfvars.example` and `.terraform.lock.hcl`.
3. `stages/enterprise` supports `dev`, `uat`, `staging`, and `prod` with `35` variables.
4. Map `QUESTIONNAIRE.md` Section 1 to `00_linked.auto.tfvars.json`.
5. Map `QUESTIONNAIRE.md` Sections 2 and 3 to `envs/<env>.tfvars`.
6. Keep `envs/{dev,uat,staging,prod}.tfvars.example` synchronized with `variables.tf`.
7. Pin `required_version = ">= 1.8.0"` and `hashicorp/google` `version = ">= 8.0.0, < 9.0.0"` in `versions.tf`.
8. Never use deprecated `explanation_spec`, `google_notebooks_instance`, or `optimized` Feature Online Store blocks.
9. Never write `coalesce(..., "")` or `var.x != null && trimspace(var.x) != ""` in Terraform.
10. Never commit real GCP project IDs, org IDs, secrets, `*.tfstate*`, `custom.auto.tfvars`, or `00_linked.auto.tfvars.json`.
11. Keep prose sentences within `20` words without emojis, section symbols, or banned filler words.

## Verification Commands

```bash
make fmt-check
make validate
make test
```

## Available Repository Skills

- `.agents/skills/setup-data-science/SKILL.md`
- `.agents/skills/audit-xai-compliance/SKILL.md`
