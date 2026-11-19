# Six-Step Enterprise Operationalization Guide

Use this guide to take `gcp-data-science` from Day-0 setup to Day-2 regulated operations.

## Day 0: Step 1 — Private Repository and MRM Environments

1. Create a private repository from `choo737/gcp-data-science` (`gh repo create --private`).
2. Protect the `main` branch and require `CODEOWNERS` reviews across `modules/`, `src/`, and `stages/`.
3. Configure GitHub Environments `dev`, `staging`, `model-validator`, and `prod`.
4. Require manual reviewer approval from model risk validators on `model-validator` and `prod`.
5. Configure WIF variables (`GCP_WIF_PROVIDER`, `GCP_DS_WIF_SA`, `GCP_TFSTATE_BUCKET`, `IAC_OUTPUTS_BUCKET`, `ARTIFACT_REGISTRY_REPOSITORY`).

## Day 1: Step 2 — Link Upstream Landing Zone Outputs

Link upstream `gcp-foundation` outputs into `stages/enterprise/` before running Terraform.

```bash
./scripts/link-stages.sh --env dev \
  --from-gcs gs://<outputs-bucket>/foundation-outputs.json \
  stages/enterprise
```

For Brownfield estates, generate `stages/enterprise/00_linked.auto.tfvars.json` via `gcp-launchpad/scripts/generate-brownfield-link.py`.

## Day 1: Step 3 — Build Signed CPR and Scorer Containers

1. Run `.github/workflows/build-cpr-and-scorer.yml` to build `cpr-treeshap` and `xai-scorer`.
2. Both Dockerfiles enforce `--require-hashes --no-deps` against locked `requirements.lock` files.
3. Record the resulting `@sha256` immutable digests in `stages/enterprise/envs/prod.tfvars`.

## Day 1: Step 4 — Deploy Baseline (`dev` and `staging`)

1. Copy `stages/enterprise/envs/dev.tfvars.example` to `stages/enterprise/envs/dev.tfvars`.
2. Deploy Track A (`ds_serving_track = "bqml_batch"`, `enable_bqml_schedule = false`) first in `dev`.
3. Promote `staging` with `ds_serving_track = "both"` and `enable_online_feature_store = false` to avoid idle Bigtable cost.
4. Opt into `enable_online_feature_store = true` in `asia-southeast1` only when `<25 ms` online serving is required.

## Day 2: Step 5 — Execute Mathematical MRM Gate Before `prod`

1. Run `python3 -m src.mrm_validation.evaluate_gate --input-json <eval.json> --output-card model_card.json`.
2. Confirm ROC-AUC and KS beat thresholds without regressing against the incumbent model.
3. Confirm subgroup disparate impact is `>= 0.80` and false-positive-rate gap is `<= 0.05`.
4. Confirm TreeSHAP vs BQML log-odds parity drift stays `<= 0.00001`.
5. Approve the `model-validator` GitHub environment gate to trigger `apply-prod`.

## Day 2: Step 6 — Operate, Reconcile, and Audit

- Monitor `ml_xai_audit_ledger` daily and reconcile against `vertex_predict_raw_tap`.
- Verify Cloud Monitoring alerts on Track A Cloud Workflows failures.
- Rotate models through challenger evaluation rather than editing production endpoints manually.
