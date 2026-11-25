# XAI Serving Endpoint Module

Provisions a private Agent Platform (formerly Vertex AI) Endpoint and a fail-closed Cloud Run scorer.

## Architecture Notes

- Configures Private Service Connect allowlisting for `ml_ops_project_id` and `ds_exp_project_id`.
- Enables 100% BigQuery `:predict` request-response logging as a secondary reconciliation tap.
- Omits `explanation_spec` because online explanations are computed via exact TreeSHAP inside CPR.
- Requires `@sha256` container image digests in `prod`.

## Inputs and Outputs

See [variables.tf](variables.tf) and [outputs.tf](outputs.tf) for all module settings.
