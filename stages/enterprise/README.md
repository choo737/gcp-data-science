# Enterprise Data Science Stage

Multi-project Terraform root coordinating Feature Store, Vertex AI Endpoint, Cloud Run scorer, and Track A BQML workflows.

## Usage

1. Link upstream outputs using `scripts/link-stages.sh --env <env> ... stages/enterprise`.
2. Copy `envs/<env>.tfvars.example` to `envs/<env>.tfvars` and set image digests.
3. Run `make plan STAGE=enterprise ENV=<env> TFSTATE_BUCKET=<bucket>`.

## Inputs and Outputs

See [variables.tf](variables.tf) and [outputs.tf](outputs.tf) for the 35-variable interface contract.
