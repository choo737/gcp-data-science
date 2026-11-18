# Sandbox Data Science Stage

Single-project Terraform root for prototyping Feature Store, BQML Track A, and CPR Track B workflows.

## Usage

1. Copy `terraform.tfvars.example` to `terraform.tfvars` or `custom.auto.tfvars`.
2. Run `make plan STAGE=sandbox ENV=dev TFSTATE_BUCKET=<bucket>`.
3. Run `make apply STAGE=sandbox ENV=dev TFSTATE_BUCKET=<bucket>`.

## Inputs and Outputs

See [variables.tf](variables.tf) and [outputs.tf](outputs.tf) for all configuration options.
