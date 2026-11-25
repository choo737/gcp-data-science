# Feature Store Module

Provisions a zero-copy Agent Platform (formerly Vertex AI) Feature Group backed by BigQuery Silver tables.

Optionally provisions a Bigtable-backed Agent Platform Feature Online Store and synced FeatureView.

## Security and Compliance

- Enforces CMEK on the Feature Online Store when `kms_key_id` is supplied.
- Rejects online serving in `asia-southeast2` through a Terraform precondition.
- Uses Bigtable autoscaling instead of deprecated `optimized` online stores.

## Inputs and Outputs

See [variables.tf](variables.tf) and [outputs.tf](outputs.tf) for all module settings.
