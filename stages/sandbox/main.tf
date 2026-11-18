provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

data "google_project" "current" {
  project_id = var.project_id
}

locals {
  propagation_wait_seconds = "20s"
  dataset_prefix           = replace(lower(var.prefix), "-", "_")
  raw_silver_ds            = var.silver_dataset_id != null ? trimspace(var.silver_dataset_id) : ""
  effective_silver_ds      = local.raw_silver_ds != "" ? local.raw_silver_ds : "${local.dataset_prefix}_silver_curated"
  raw_audit_ds             = var.ml_audit_dataset_id != null ? trimspace(var.ml_audit_dataset_id) : ""
  effective_audit_ds       = local.raw_audit_ds != "" ? local.raw_audit_ds : "${local.dataset_prefix}_ml_xai_audit"
  effective_kms_key_id     = var.kms_key_id != null ? trimspace(var.kms_key_id) : ""
  use_cmek                 = local.effective_kms_key_id != ""
  enable_online_track      = contains(["vertex_cpr_online", "both"], var.ds_serving_track)
  enable_bqml_track        = contains(["bqml_batch", "both"], var.ds_serving_track)

  effective_labels = merge({
    environment = "sandbox"
    managed_by  = "terraform"
    layer       = "data-science"
  }, var.labels)

  required_apis = toset(concat(
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

resource "google_project_service" "apis" {
  for_each = local.required_apis

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "time_sleep" "api_propagation" {
  create_duration = local.propagation_wait_seconds

  depends_on = [google_project_service.apis]
}

resource "google_bigquery_dataset" "silver" {
  count = var.create_silver_dataset ? 1 : 0

  project                    = var.project_id
  dataset_id                 = local.effective_silver_ds
  friendly_name              = "${var.prefix} Silver Features"
  description                = "Sandbox Silver dataset created when standalone mode is enabled."
  location                   = var.region
  delete_contents_on_destroy = var.delete_contents_on_destroy
  labels                     = local.effective_labels

  dynamic "default_encryption_configuration" {
    for_each = local.use_cmek ? [local.effective_kms_key_id] : []
    content {
      kms_key_name = default_encryption_configuration.value
    }
  }

  depends_on = [time_sleep.api_propagation]
}

resource "google_bigquery_dataset" "ml_audit" {
  count = var.create_ml_audit_dataset && local.raw_audit_ds == "" ? 1 : 0

  project                    = var.project_id
  dataset_id                 = local.effective_audit_ds
  friendly_name              = "${var.prefix} ML XAI Audit"
  description                = "Sandbox dataset storing ml_xai_audit_ledger and BQML models."
  location                   = var.region
  delete_contents_on_destroy = var.delete_contents_on_destroy
  labels                     = local.effective_labels

  dynamic "default_encryption_configuration" {
    for_each = local.use_cmek ? [local.effective_kms_key_id] : []
    content {
      kms_key_name = default_encryption_configuration.value
    }
  }

  depends_on = [time_sleep.api_propagation]
}

resource "google_bigquery_table" "ml_xai_audit_ledger" {
  project             = var.project_id
  dataset_id          = local.effective_audit_ds
  table_id            = var.ml_audit_table_id
  description         = "Sandbox 18-column XAI decision and TreeSHAP audit ledger."
  deletion_protection = !var.delete_contents_on_destroy
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
    time_sleep.api_propagation,
    google_bigquery_dataset.ml_audit,
  ]
}

module "feature_store" {
  source = "../../modules/ds/feature-store"

  ml_ops_project_id              = var.project_id
  ml_ops_project_number          = data.google_project.current.number
  core_project_id                = var.project_id
  silver_dataset_id              = local.effective_silver_ds
  feature_group_source_table     = var.feature_group_source_table
  region                         = var.region
  prefix                         = var.prefix
  enable_online_feature_store    = var.enable_online_feature_store
  feature_online_store_min_nodes = var.feature_online_store_min_nodes
  feature_online_store_max_nodes = var.feature_online_store_max_nodes
  feature_sync_cron              = var.feature_sync_cron
  kms_key_id                     = local.effective_kms_key_id
  delete_contents_on_destroy     = var.delete_contents_on_destroy
  labels                         = local.effective_labels

  depends_on = [
    time_sleep.api_propagation,
    google_bigquery_dataset.silver,
  ]
}

module "xai_serving_endpoint" {
  count  = local.enable_online_track ? 1 : 0
  source = "../../modules/ds/xai-serving-endpoint"

  ml_ops_project_id          = var.project_id
  ds_exp_project_id          = var.project_id
  ml_audit_dataset_id        = local.effective_audit_ds
  ml_audit_table_id          = google_bigquery_table.ml_xai_audit_ledger.table_id
  region                     = var.region
  prefix                     = var.prefix
  environment                = "dev"
  enable_scorer_service      = var.enable_scorer_service
  scorer_container_image     = var.scorer_container_image
  deploy_cpr_model           = var.deploy_cpr_model
  cpr_container_image        = var.cpr_container_image
  serving_track              = var.ds_serving_track
  shap_additivity_tolerance  = var.shap_additivity_tolerance
  kms_key_id                 = local.effective_kms_key_id
  vpc_id                     = var.vpc_id
  egress_connector_subnet_id = var.egress_connector_subnet_id
  delete_contents_on_destroy = var.delete_contents_on_destroy
  labels                     = local.effective_labels

  depends_on = [
    time_sleep.api_propagation,
    google_bigquery_table.ml_xai_audit_ledger,
  ]
}

module "bqml_xai_pipeline" {
  count  = local.enable_bqml_track ? 1 : 0
  source = "../../modules/ds/bqml-xai-pipeline"

  ml_ops_project_id              = var.project_id
  core_project_id                = var.project_id
  silver_dataset_id              = local.effective_silver_ds
  feature_group_source_table     = var.feature_group_source_table
  ml_audit_dataset_id            = local.effective_audit_ds
  ml_audit_table_id              = google_bigquery_table.ml_xai_audit_ledger.table_id
  region                         = var.region
  prefix                         = var.prefix
  enable_bqml_schedule           = var.enable_bqml_schedule
  bqml_schedule_cron             = var.bqml_schedule_cron
  kms_key_id                     = local.effective_kms_key_id
  alert_notification_channel_ids = var.alert_notification_channel_ids
  delete_contents_on_destroy     = var.delete_contents_on_destroy
  labels                         = local.effective_labels

  depends_on = [
    time_sleep.api_propagation,
    google_bigquery_table.ml_xai_audit_ledger,
  ]
}
