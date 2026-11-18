provider "google" {
  region = var.region
}

provider "google-beta" {
  region = var.region
}

locals {
  propagation_wait_seconds   = "20s"
  delete_contents_on_destroy = var.environment == "dev"

  trimmed_domain_name   = var.domain_name != null ? trimspace(var.domain_name) : ""
  workflow_prefix       = local.trimmed_domain_name != "" ? "${var.prefix}-${local.trimmed_domain_name}" : var.prefix
  effective_kms_key_id  = var.kms_key_id != null ? trimspace(var.kms_key_id) : ""
  use_cmek              = local.effective_kms_key_id != ""
  effective_vpc_id      = var.vpc_id != null ? trimspace(var.vpc_id) : ""
  effective_subnet_id   = var.egress_connector_subnet_id != null ? trimspace(var.egress_connector_subnet_id) : ""
  effective_worm_bucket = var.ml_worm_bucket_name != null ? trimspace(var.ml_worm_bucket_name) : ""
  trimmed_repo_id       = var.ml_artifact_repo_id != null ? trimspace(var.ml_artifact_repo_id) : ""
  artifact_repo_name    = local.trimmed_repo_id != "" ? element(split("/", local.trimmed_repo_id), length(split("/", local.trimmed_repo_id)) - 1) : ""

  enable_online_track = contains(["vertex_cpr_online", "both"], var.ds_serving_track)
  enable_bqml_track   = contains(["bqml_batch", "both"], var.ds_serving_track)

  effective_labels = merge({
    environment = var.environment
    managed_by  = "terraform"
    layer       = "data-science"
  }, var.labels)

  ml_ops_required_apis = toset(concat(
    [
      "aiplatform.googleapis.com",
      "bigquery.googleapis.com",
      "bigquerystorage.googleapis.com",
      "iam.googleapis.com",
    ],
    local.enable_online_track && var.enable_scorer_service ? ["run.googleapis.com"] : [],
    local.enable_bqml_track ? ["workflows.googleapis.com", "workflowexecutions.googleapis.com"] : [],
    local.enable_bqml_track && var.enable_bqml_schedule ? ["cloudscheduler.googleapis.com"] : [],
    length(var.alert_notification_channel_ids) > 0 ? ["monitoring.googleapis.com"] : [],
  ))

  ml_xai_audit_ledger_schema = jsonencode([
    { name = "decision_id", type = "STRING", mode = "REQUIRED", description = "Unique decision UUID." },
    { name = "entity_id", type = "STRING", mode = "REQUIRED", description = "Scored customer or entity identifier." },
    { name = "decision_timestamp", type = "TIMESTAMP", mode = "REQUIRED", description = "UTC timestamp when decision was recorded." },
    { name = "serving_track", type = "STRING", mode = "REQUIRED", description = "Serving track (bqml_batch or vertex_cpr_online)." },
    { name = "model_id", type = "STRING", mode = "REQUIRED", description = "Registered Vertex AI or BQML model ID." },
    { name = "model_version_id", type = "STRING", mode = "REQUIRED", description = "Deployed model version identifier." },
    { name = "model_artifact_sha256", type = "STRING", mode = "REQUIRED", description = "SHA-256 digest of the XGBoost or model artifact." },
    { name = "container_image_digest", type = "STRING", mode = "REQUIRED", description = "Immutable container image digest serving the decision." },
    { name = "schema_version", type = "STRING", mode = "REQUIRED", description = "Audit ledger payload schema version." },
    { name = "explained_class", type = "STRING", mode = "REQUIRED", description = "Target class explained by TreeSHAP attributions." },
    { name = "approve_prob", type = "FLOAT", mode = "REQUIRED", description = "FLOAT64 calibrated approval probability." },
    { name = "raw_margin", type = "FLOAT", mode = "REQUIRED", description = "FLOAT64 raw model log-odds margin." },
    { name = "base_logit", type = "FLOAT", mode = "REQUIRED", description = "FLOAT64 TreeSHAP expected baseline log-odds." },
    { name = "shap_sum_logit", type = "FLOAT", mode = "REQUIRED", description = "FLOAT64 sum of per-feature TreeSHAP log-odds." },
    { name = "shap_logit_map", type = "JSON", mode = "REQUIRED", description = "JSON map of feature names to exact TreeSHAP log-odds." },
    { name = "top_adverse_reasons", type = "JSON", mode = "REQUIRED", description = "Deterministic ranked adverse-action reason codes." },
    { name = "feature_payload", type = "JSON", mode = "REQUIRED", description = "Non-PII input feature vector evaluated at scoring time." },
    { name = "reason_rule_version", type = "STRING", mode = "REQUIRED", description = "Adverse-action reason mapping rule version." },
  ])
}

data "google_project" "ml_ops" {
  project_id = var.ml_ops_project_id
}

resource "terraform_data" "validate_enterprise_inputs" {
  lifecycle {
    precondition {
      condition     = trimspace(var.core_project_id) != "" && trimspace(var.ml_ops_project_id) != "" && trimspace(var.ds_exp_project_id) != ""
      error_message = "core_project_id, ml_ops_project_id, and ds_exp_project_id must be non-empty."
    }
    precondition {
      condition     = trimspace(var.silver_dataset_id) != "" && trimspace(var.ml_audit_dataset_id) != "" && trimspace(var.ml_artifact_repo_id) != ""
      error_message = "silver_dataset_id, ml_audit_dataset_id, and ml_artifact_repo_id must be non-empty."
    }
    precondition {
      condition     = var.fairness_min_disparate_impact > 0 && var.fairness_min_disparate_impact <= 1.0
      error_message = "fairness_min_disparate_impact must be in (0, 1.0]."
    }
    precondition {
      condition     = var.shap_additivity_tolerance > 0 && var.shap_additivity_tolerance <= 0.01
      error_message = "shap_additivity_tolerance must be positive and <= 0.01."
    }
  }
}

resource "google_project_service" "ml_ops_apis" {
  for_each = local.ml_ops_required_apis

  project            = var.ml_ops_project_id
  service            = each.value
  disable_on_destroy = false
}

resource "time_sleep" "api_propagation" {
  create_duration = local.propagation_wait_seconds

  depends_on = [google_project_service.ml_ops_apis]
}

resource "google_project_service_identity" "aiplatform_agent" {
  provider = google-beta
  project  = var.ml_ops_project_id
  service  = "aiplatform.googleapis.com"

  depends_on = [time_sleep.api_propagation]
}

resource "google_project_service_identity" "run_agent" {
  count    = local.enable_online_track && var.enable_scorer_service ? 1 : 0
  provider = google-beta
  project  = var.ml_ops_project_id
  service  = "run.googleapis.com"

  depends_on = [time_sleep.api_propagation]
}

resource "google_project_service_identity" "workflows_agent" {
  count    = local.enable_bqml_track ? 1 : 0
  provider = google-beta
  project  = var.ml_ops_project_id
  service  = "workflows.googleapis.com"

  depends_on = [time_sleep.api_propagation]
}

resource "time_sleep" "service_identity_propagation" {
  create_duration = local.propagation_wait_seconds

  depends_on = [
    google_project_service_identity.aiplatform_agent,
    google_project_service_identity.run_agent,
    google_project_service_identity.workflows_agent,
  ]
}

resource "google_kms_crypto_key_iam_member" "aiplatform_cmek" {
  count = local.use_cmek && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_project_service_identity.aiplatform_agent.member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "google_kms_crypto_key_iam_member" "run_cmek" {
  count = local.use_cmek && var.manage_kms_iam && local.enable_online_track && var.enable_scorer_service ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_project_service_identity.run_agent[0].member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "google_kms_crypto_key_iam_member" "workflows_cmek" {
  count = local.use_cmek && var.manage_kms_iam && local.enable_bqml_track ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_project_service_identity.workflows_agent[0].member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "time_sleep" "cmek_propagation" {
  count = local.use_cmek ? 1 : 0

  create_duration = local.propagation_wait_seconds

  depends_on = [
    google_kms_crypto_key_iam_member.aiplatform_cmek,
    google_kms_crypto_key_iam_member.run_cmek,
    google_kms_crypto_key_iam_member.workflows_cmek,
  ]
}

resource "google_bigquery_table" "ml_xai_audit_ledger" {
  project             = var.ml_ops_project_id
  dataset_id          = var.ml_audit_dataset_id
  table_id            = var.ml_audit_table_id
  description         = "Immutable 18-column XAI decision and TreeSHAP audit ledger."
  deletion_protection = !local.delete_contents_on_destroy
  clustering          = ["serving_track", "model_id", "entity_id"]
  schema              = local.ml_xai_audit_ledger_schema
  labels              = local.effective_labels

  time_partitioning {
    type  = "DAY"
    field = "decision_timestamp"
  }

  dynamic "encryption_configuration" {
    for_each = local.use_cmek ? [local.effective_kms_key_id] : []
    content {
      kms_key_name = encryption_configuration.value
    }
  }

  depends_on = [
    terraform_data.validate_enterprise_inputs,
    time_sleep.api_propagation,
    time_sleep.cmek_propagation,
  ]
}

resource "google_bigquery_dataset_iam_member" "aiplatform_silver_viewer" {
  project    = var.core_project_id
  dataset_id = var.silver_dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = google_project_service_identity.aiplatform_agent.member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "google_artifact_registry_repository_iam_member" "aiplatform_reader" {
  count = local.artifact_repo_name != "" ? 1 : 0

  project    = var.ml_ops_project_id
  location   = var.region
  repository = local.artifact_repo_name
  role       = "roles/artifactregistry.reader"
  member     = google_project_service_identity.aiplatform_agent.member

  depends_on = [time_sleep.service_identity_propagation]
}

module "feature_store" {
  source = "../../modules/ds/feature-store"

  ml_ops_project_id              = var.ml_ops_project_id
  ml_ops_project_number          = data.google_project.ml_ops.number
  core_project_id                = var.core_project_id
  silver_dataset_id              = var.silver_dataset_id
  feature_group_source_table     = var.feature_group_source_table
  region                         = var.region
  prefix                         = local.workflow_prefix
  enable_online_feature_store    = var.enable_online_feature_store
  feature_online_store_min_nodes = var.feature_online_store_min_nodes
  feature_online_store_max_nodes = var.feature_online_store_max_nodes
  feature_sync_cron              = var.feature_sync_cron
  kms_key_id                     = local.effective_kms_key_id
  delete_contents_on_destroy     = local.delete_contents_on_destroy
  labels                         = local.effective_labels

  depends_on = [
    terraform_data.validate_enterprise_inputs,
    time_sleep.service_identity_propagation,
    time_sleep.cmek_propagation,
    google_bigquery_dataset_iam_member.aiplatform_silver_viewer,
  ]
}

module "xai_serving_endpoint" {
  count  = local.enable_online_track ? 1 : 0
  source = "../../modules/ds/xai-serving-endpoint"

  ml_ops_project_id          = var.ml_ops_project_id
  ds_exp_project_id          = var.ds_exp_project_id
  ml_audit_dataset_id        = var.ml_audit_dataset_id
  ml_audit_table_id          = google_bigquery_table.ml_xai_audit_ledger.table_id
  ml_worm_bucket_name        = local.effective_worm_bucket
  region                     = var.region
  prefix                     = local.workflow_prefix
  environment                = var.environment
  enable_scorer_service      = var.enable_scorer_service
  scorer_container_image     = var.scorer_container_image
  deploy_cpr_model           = var.deploy_cpr_model
  cpr_container_image        = var.cpr_container_image
  serving_track              = var.ds_serving_track
  shap_additivity_tolerance  = var.shap_additivity_tolerance
  kms_key_id                 = local.effective_kms_key_id
  manage_kms_iam             = var.manage_kms_iam
  vpc_id                     = local.effective_vpc_id
  egress_connector_subnet_id = local.effective_subnet_id
  delete_contents_on_destroy = local.delete_contents_on_destroy
  labels                     = local.effective_labels

  depends_on = [
    terraform_data.validate_enterprise_inputs,
    time_sleep.service_identity_propagation,
    time_sleep.cmek_propagation,
    google_bigquery_table.ml_xai_audit_ledger,
  ]
}

module "bqml_xai_pipeline" {
  count  = local.enable_bqml_track ? 1 : 0
  source = "../../modules/ds/bqml-xai-pipeline"

  ml_ops_project_id              = var.ml_ops_project_id
  core_project_id                = var.core_project_id
  silver_dataset_id              = var.silver_dataset_id
  feature_group_source_table     = var.feature_group_source_table
  ml_audit_dataset_id            = var.ml_audit_dataset_id
  ml_audit_table_id              = google_bigquery_table.ml_xai_audit_ledger.table_id
  region                         = var.region
  prefix                         = local.workflow_prefix
  enable_bqml_schedule           = var.enable_bqml_schedule
  bqml_schedule_cron             = var.bqml_schedule_cron
  kms_key_id                     = local.effective_kms_key_id
  manage_kms_iam                 = var.manage_kms_iam
  alert_notification_channel_ids = var.alert_notification_channel_ids
  delete_contents_on_destroy     = local.delete_contents_on_destroy
  labels                         = local.effective_labels

  depends_on = [
    terraform_data.validate_enterprise_inputs,
    time_sleep.service_identity_propagation,
    time_sleep.cmek_propagation,
    google_bigquery_table.ml_xai_audit_ledger,
  ]
}
