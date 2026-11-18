# CLAUDE.md

Read [AGENTS.md](AGENTS.md) and [QUESTIONNAIRE.md](QUESTIONNAIRE.md) first.
Also read [the six-step guide](docs/01-six-step-operationalization.md).
Load [the setup skill](.agents/skills/setup-data-science/SKILL.md) before running commands.

## Quick Rules

- Create a private repository (`gh repo create --private`) before storing customer configuration.
- Pair with [gcp-launchpad](https://github.com/choo737/gcp-launchpad) for multi-repo routing.
- Link landing zones from [gcp-foundation](https://github.com/choo737/gcp-foundation).
- Read Bronze feeds from [gcp-data-ingestion](https://github.com/choo737/gcp-data-ingestion).
- Read Silver features from [gcp-data-medallion](https://github.com/choo737/gcp-data-medallion).
- Check optional `platform-context.json` from [gcp-launchpad](https://github.com/choo737/gcp-launchpad) or complete `QUESTIONNAIRE.md` first.
- Link Greenfield outputs via `scripts/link-stages.sh --env <env>` or populate Brownfield `00_linked.auto.tfvars.json`.
- Keep all three `modules/ds/` directories at exact five-file symmetry.
- Keep `stages/enterprise/variables.tf` locked at exactly `32` variables.
- Never use `explanation_spec`, `google_notebooks_instance`, or `optimized` Feature Online Store blocks.
- Pin Terraform `>= 1.8.0` and `hashicorp/google` `>= 8.0.0, < 9.0.0`.
- Run `make fmt-check`, `make validate`, and `make test` after every edit.
