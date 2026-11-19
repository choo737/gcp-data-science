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

data "google_storage_project_service_account" "gcs_account" {
  count   = local.use_cmek && var.manage_kms_iam ? 1 : 0
  project = var.project_id

  depends_on = [time_sleep.api_propagation]
}

data "google_bigquery_default_service_account" "bq_account" {
  count   = local.use_cmek && var.manage_kms_iam ? 1 : 0
  project = var.project_id

  depends_on = [time_sleep.api_propagation]
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
  effective_vpc_id         = var.vpc_id != null ? trimspace(var.vpc_id) : ""
  effective_subnet_id      = var.egress_connector_subnet_id != null ? trimspace(var.egress_connector_subnet_id) : ""
  use_direct_vpc           = local.effective_vpc_id != "" && local.effective_subnet_id != ""
  network_project_id       = local.effective_vpc_id != "" ? try(regex("^projects/([^/]+)/global/networks/", local.effective_vpc_id)[0], var.project_id) : var.project_id
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
      "artifactregistry.googleapis.com",
      "bigquery.googleapis.com",
      "bigquerystorage.googleapis.com",
      "iam.googleapis.com",
    ],
    local.enable_online_track && (var.enable_scorer_service || var.deploy_cpr_model) ? ["run.googleapis.com"] : [],
    local.enable_bqml_track ? ["workflows.googleapis.com", "workflowexecutions.googleapis.com", "pubsub.googleapis.com"] : [],
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

  ml_training_labels_schema = jsonencode([
    { name = "application_id", type = "STRING", mode = "REQUIRED", description = "Unique credit application identifier." },
    { name = "entity_id", type = "STRING", mode = "REQUIRED", description = "Scored customer or entity identifier." },
    { name = "label_timestamp", type = "TIMESTAMP", mode = "REQUIRED", description = "UTC timestamp when ground-truth outcome was observed." },
    { name = "approved_flag", type = "INT64", mode = "REQUIRED", description = "Binary ground-truth approval outcome (1 = approved, 0 = rejected)." },
    { name = "demographic_group", type = "STRING", mode = "NULLABLE", description = "Protected fairness evaluation subgroup attribute." },
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

resource "google_project_service" "network_connectivity_api" {
  count = local.use_direct_vpc ? 1 : 0

  project            = local.network_project_id
  service            = "networkconnectivity.googleapis.com"
  disable_on_destroy = false
}

resource "time_sleep" "network_api_propagation" {
  count           = local.use_direct_vpc ? 1 : 0
  create_duration = local.propagation_wait_seconds

  depends_on = [google_project_service.network_connectivity_api]
}

resource "google_project_service_identity" "aiplatform_agent" {
  provider = google-beta
  project  = var.project_id
  service  = "aiplatform.googleapis.com"

  depends_on = [time_sleep.api_propagation]
}

resource "google_project_service_identity" "run_agent" {
  count    = local.use_cmek && var.manage_kms_iam && local.enable_online_track && (var.enable_scorer_service || var.deploy_cpr_model) ? 1 : 0
  provider = google-beta
  project  = var.project_id
  service  = "run.googleapis.com"

  depends_on = [time_sleep.api_propagation]
}

resource "google_project_service_identity" "workflows_agent" {
  count    = local.use_cmek && var.manage_kms_iam && local.enable_bqml_track ? 1 : 0
  provider = google-beta
  project  = var.project_id
  service  = "workflows.googleapis.com"

  depends_on = [time_sleep.api_propagation]
}

resource "google_project_service_identity" "pubsub_agent" {
  count    = local.use_cmek && var.manage_kms_iam && local.enable_bqml_track ? 1 : 0
  provider = google-beta
  project  = var.project_id
  service  = "pubsub.googleapis.com"

  depends_on = [time_sleep.api_propagation]
}

resource "google_project_service_identity" "artifactregistry_agent" {
  count    = local.use_cmek && var.manage_kms_iam ? 1 : 0
  provider = google-beta
  project  = var.project_id
  service  = "artifactregistry.googleapis.com"

  depends_on = [time_sleep.api_propagation]
}

resource "time_sleep" "service_identity_propagation" {
  create_duration = local.propagation_wait_seconds

  depends_on = [
    google_project_service_identity.aiplatform_agent,
    google_project_service_identity.run_agent,
    google_project_service_identity.workflows_agent,
    google_project_service_identity.pubsub_agent,
    google_project_service_identity.artifactregistry_agent,
    data.google_storage_project_service_account.gcs_account,
    data.google_bigquery_default_service_account.bq_account,
  ]
}

resource "google_project_iam_member" "aiplatform_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = google_project_service_identity.aiplatform_agent.member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "google_kms_crypto_key_iam_member" "aiplatform_cmek" {
  count = local.use_cmek && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_project_service_identity.aiplatform_agent.member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "google_kms_crypto_key_iam_member" "run_cmek" {
  count = local.use_cmek && var.manage_kms_iam && local.enable_online_track && (var.enable_scorer_service || var.deploy_cpr_model) ? 1 : 0

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

resource "google_kms_crypto_key_iam_member" "pubsub_cmek" {
  count = local.use_cmek && var.manage_kms_iam && local.enable_bqml_track ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_project_service_identity.pubsub_agent[0].member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "google_kms_crypto_key_iam_member" "artifactregistry_cmek" {
  count = local.use_cmek && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_project_service_identity.artifactregistry_agent[0].member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "google_kms_crypto_key_iam_member" "gcs_cmek" {
  count = local.use_cmek && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = data.google_storage_project_service_account.gcs_account[0].member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "google_kms_crypto_key_iam_member" "bq_cmek" {
  count = local.use_cmek && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = data.google_bigquery_default_service_account.bq_account[0].member

  depends_on = [time_sleep.service_identity_propagation]
}

resource "time_sleep" "cmek_propagation" {
  count = local.use_cmek ? 1 : 0

  create_duration = local.propagation_wait_seconds

  depends_on = [
    google_kms_crypto_key_iam_member.aiplatform_cmek,
    google_kms_crypto_key_iam_member.run_cmek,
    google_kms_crypto_key_iam_member.workflows_cmek,
    google_kms_crypto_key_iam_member.pubsub_cmek,
    google_kms_crypto_key_iam_member.artifactregistry_cmek,
    google_kms_crypto_key_iam_member.gcs_cmek,
    google_kms_crypto_key_iam_member.bq_cmek,
  ]
}

resource "google_network_connectivity_service_connection_policy" "vertex_psc" {
  count = local.use_direct_vpc ? 1 : 0

  project       = local.network_project_id
  name          = "${replace(lower(var.prefix), "_", "-")}-vertex-psc-policy"
  location      = var.region
  service_class = "gcp-vertexai"
  network       = local.effective_vpc_id
  labels        = local.effective_labels

  psc_config {
    subnetworks = [local.effective_subnet_id]
  }

  depends_on = [time_sleep.network_api_propagation]
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

  depends_on = [
    time_sleep.api_propagation,
    time_sleep.cmek_propagation,
  ]
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

  depends_on = [
    time_sleep.api_propagation,
    time_sleep.cmek_propagation,
  ]
}

resource "google_bigquery_table" "ml_xai_audit_ledger" {
  project                  = var.project_id
  dataset_id               = local.effective_audit_ds
  table_id                 = var.ml_audit_table_id
  description              = "Sandbox 18-column XAI decision and TreeSHAP audit ledger."
  require_partition_filter = true
  deletion_protection      = !var.delete_contents_on_destroy
  clustering               = ["serving_track", "model_id", "explained_class", "entity_id"]
  schema                   = local.ml_xai_audit_ledger_schema
  labels                   = local.effective_labels

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
    time_sleep.cmek_propagation,
    google_bigquery_dataset.ml_audit,
  ]
}

resource "google_bigquery_table" "ml_training_labels" {
  project                  = var.project_id
  dataset_id               = local.effective_audit_ds
  table_id                 = "ml_training_labels"
  description              = "Point-in-time ground-truth credit approval labels for Track A and Track B training."
  require_partition_filter = true
  deletion_protection      = !var.delete_contents_on_destroy
  clustering               = ["entity_id", "approved_flag"]
  schema                   = local.ml_training_labels_schema
  labels                   = local.effective_labels

  time_partitioning {
    type  = "DAY"
    field = "label_timestamp"
  }

  dynamic "encryption_configuration" {
    for_each = local.use_cmek ? [local.effective_kms_key_id] : []
    content {
      kms_key_name = encryption_configuration.value
    }
  }

  depends_on = [
    time_sleep.api_propagation,
    time_sleep.cmek_propagation,
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
    time_sleep.service_identity_propagation,
    time_sleep.cmek_propagation,
    google_bigquery_dataset.silver,
    google_project_iam_member.aiplatform_bq_job_user,
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
  cpr_model_artifact_uri     = var.cpr_model_artifact_uri
  cpr_model_sha256           = var.cpr_model_sha256
  serving_track              = var.ds_serving_track
  shap_additivity_tolerance  = var.shap_additivity_tolerance
  kms_key_id                 = local.effective_kms_key_id
  manage_kms_iam             = var.manage_kms_iam
  vpc_id                     = local.effective_vpc_id
  egress_connector_subnet_id = local.effective_subnet_id
  delete_contents_on_destroy = var.delete_contents_on_destroy
  labels                     = local.effective_labels

  depends_on = [
    time_sleep.api_propagation,
    time_sleep.cmek_propagation,
    google_bigquery_table.ml_xai_audit_ledger,
    google_network_connectivity_service_connection_policy.vertex_psc,
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
  manage_kms_iam                 = var.manage_kms_iam
  alert_notification_channel_ids = var.alert_notification_channel_ids
  delete_contents_on_destroy     = var.delete_contents_on_destroy
  labels                         = local.effective_labels

  depends_on = [
    time_sleep.api_propagation,
    time_sleep.cmek_propagation,
    google_bigquery_table.ml_xai_audit_ledger,
    google_bigquery_table.ml_training_labels,
  ]
}