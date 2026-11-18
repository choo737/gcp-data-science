---
name: setup-data-science
description: Interactive Day-0 to Day-2 workflow covering private repo creation, stage linking, container builds, Terraform deploy, and verification.
---

# Setup Data Science

Run these steps in order whenever deploying this repository for a customer or sandbox.

## Step 0: Create a Private Repository

GitHub forces standard `gh repo fork` clones to remain public.

Never commit customer project IDs or regulated model configs to a public fork.

```bash
gh repo create <owner>/<customer-repo> --private --template choo737/gcp-data-science --clone
cd <customer-repo>
gh repo view --json isPrivate --jq '.isPrivate'
```

Confirm the command prints `true`.

## Step 1: Collect Customer Inputs

Check for `platform-context.json` from [gcp-launchpad](https://github.com/choo737/gcp-launchpad) first.

Complete [QUESTIONNAIRE.md](../../../QUESTIONNAIRE.md) before editing `stages/enterprise/envs/<env>.tfvars` or `stages/sandbox/terraform.tfvars`.

Confirm these required decisions:

- `environment` (`dev`, `staging`, `prod`), `region` (`asia-southeast1`), and `prefix`.
- `ds_serving_track`: `bqml_batch`, `vertex_cpr_online`, or `both`.
- `enable_online_feature_store`: keep `false` in `asia-southeast2` because online serving is unsupported there.
- `fairness_min_disparate_impact` (`0.80`) and `shap_additivity_tolerance` (`0.00001`).

## Step 2: Link Upstream Foundation Outputs

Verify active `gcloud` credentials before running Terraform:

```bash
gcloud config get-value account
```

For Greenfield deployments, link upstream `gcp-foundation` outputs into `00_linked.auto.tfvars.json`:

```bash
./scripts/link-stages.sh --env dev \
  ../gcp-foundation/foundation/02-security-network \
  ../gcp-foundation/data/enterprise/01-platform \
  ../gcp-foundation/data/enterprise/02-governance-ops \
  stages/enterprise

# Or pull outputs exported earlier to GCS:
./scripts/link-stages.sh --env dev --from-gcs gs://<outputs-bucket>/foundation-outputs.json stages/enterprise
```

For Brownfield deployments, run `gcp-launchpad/scripts/generate-brownfield-link.py` or populate `00_linked.auto.tfvars.json` directly.

## Step 3: Build Immutable Container Digests

Run `.github/workflows/build-cpr-and-scorer.yml` to push `@sha256` digests for `cpr-treeshap` and `xai-scorer`.

Pin both `cpr_container_image` and `scorer_container_image` to `@sha256:...` digests in `stages/enterprise/envs/prod.tfvars`.

## Step 4: Deploy with Terraform

```bash
make fmt-check
make validate
make test
make plan STAGE=enterprise ENV=dev TFSTATE_BUCKET=<bucket>
make apply STAGE=enterprise ENV=dev TFSTATE_BUCKET=<bucket>
```

## Step 5: Verify Live Tracks

- Run the Track A Cloud Workflow and check rows in `ml_xai_audit_ledger`.
- Call `/score` on the internal Cloud Run scorer and verify `ledger_committed: true`.
- Run `python3 -m src.mrm_validation.evaluate_gate` and review `model_card.json` before `prod` approval.
