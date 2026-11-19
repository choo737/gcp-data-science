locals {
  sanitized_prefix      = replace(lower(var.prefix), "_", "-")
  workflow_name         = trimsuffix(substr("${local.sanitized_prefix}-bqml-xai-pipeline", 0, 64), "-")
  scheduler_name        = trimsuffix(substr("${local.sanitized_prefix}-bqml-xai-schedule", 0, 64), "-")
  alert_topic_name      = trimsuffix(substr("${local.sanitized_prefix}-bqml-xai-alerts", 0, 64), "-")
  workflow_sa_id        = trimsuffix(substr("${local.sanitized_prefix}-bqml-wf", 0, 28), "-")
  trigger_sa_id         = trimsuffix(substr("${local.sanitized_prefix}-bqml-trig", 0, 28), "-")
  effective_kms_key_id  = var.kms_key_id != null ? trimspace(var.kms_key_id) : ""
  same_project_channels = [for ch in var.alert_notification_channel_ids : trimspace(ch) if startswith(trimspace(ch), "projects/${var.ml_ops_project_id}/")]
  workflow_source       = file("${path.module}/../../../workflows/bqml_xai_pipeline.workflows.yaml")
}

resource "google_project_service_identity" "pubsub_agent" {
  count    = local.effective_kms_key_id != "" && var.manage_kms_iam ? 1 : 0
  provider = google-beta
  project  = var.ml_ops_project_id
  service  = "pubsub.googleapis.com"
}

resource "google_kms_crypto_key_iam_member" "pubsub_cmek" {
  count = local.effective_kms_key_id != "" && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_project_service_identity.pubsub_agent[0].member
}

resource "google_pubsub_topic" "alerts" {
  project      = var.ml_ops_project_id
  name         = local.alert_topic_name
  kms_key_name = local.effective_kms_key_id != "" ? local.effective_kms_key_id : null
  labels       = var.labels

  message_storage_policy {
    allowed_persistence_regions = [var.region]
  }

  depends_on = [google_kms_crypto_key_iam_member.pubsub_cmek]
}

resource "google_service_account" "workflow" {
  project      = var.ml_ops_project_id
  account_id   = local.workflow_sa_id
  display_name = "${var.prefix} BQML XAI Workflow"
  description  = "Least-privilege identity for Track A BQML training and ML.EXPLAIN_PREDICT jobs."
}

resource "google_service_account" "trigger" {
  count = var.enable_bqml_schedule ? 1 : 0

  project      = var.ml_ops_project_id
  account_id   = local.trigger_sa_id
  display_name = "${var.prefix} BQML XAI Trigger"
  description  = "Minimal invoker identity for Cloud Scheduler Track A runs."
}

resource "google_project_iam_member" "workflow_bq_job_user" {
  project = var.ml_ops_project_id
  role    = "roles/bigquery.jobUser"
  member  = google_service_account.workflow.member
}

resource "google_bigquery_dataset_iam_member" "workflow_silver_viewer" {
  project    = var.core_project_id
  dataset_id = var.silver_dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = google_service_account.workflow.member
}

resource "google_bigquery_dataset_iam_member" "workflow_audit_editor" {
  project    = var.ml_ops_project_id
  dataset_id = var.ml_audit_dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_service_account.workflow.member
}

resource "google_project_iam_member" "workflow_aiplatform_user" {
  project = var.ml_ops_project_id
  role    = "roles/aiplatform.user"
  member  = google_service_account.workflow.member
}

resource "google_project_iam_member" "workflow_log_writer" {
  project = var.ml_ops_project_id
  role    = "roles/logging.logWriter"
  member  = google_service_account.workflow.member
}

resource "google_pubsub_topic_iam_member" "workflow_alert_publisher" {
  project = var.ml_ops_project_id
  topic   = google_pubsub_topic.alerts.name
  role    = "roles/pubsub.publisher"
  member  = google_service_account.workflow.member
}

resource "google_kms_crypto_key_iam_member" "workflow_cmek" {
  count = local.effective_kms_key_id != "" && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_service_account.workflow.member
}

resource "google_project_iam_member" "trigger_workflows_invoker" {
  count = var.enable_bqml_schedule ? 1 : 0

  project = var.ml_ops_project_id
  role    = "roles/workflows.invoker"
  member  = google_service_account.trigger[0].member
}

resource "google_workflows_workflow" "bqml_xai" {
  project             = var.ml_ops_project_id
  region              = var.region
  name                = local.workflow_name
  description         = "Track A BQML BOOSTED_TREE_CLASSIFIER training and exact ML.EXPLAIN_PREDICT audit run."
  service_account     = google_service_account.workflow.id
  crypto_key_name     = local.effective_kms_key_id != "" ? local.effective_kms_key_id : null
  call_log_level      = "LOG_ERRORS_ONLY"
  deletion_protection = !var.delete_contents_on_destroy
  labels              = var.labels
  source_contents     = local.workflow_source

  user_env_vars = {
    ML_OPS_PROJECT_ID           = var.ml_ops_project_id
    CORE_PROJECT_ID             = var.core_project_id
    SILVER_DATASET_ID           = var.silver_dataset_id
    FEATURE_GROUP_SOURCE_TABLE  = var.feature_group_source_table
    ML_AUDIT_DATASET_ID         = var.ml_audit_dataset_id
    ML_AUDIT_TABLE_ID           = var.ml_audit_table_id
    ML_TRAINING_LABELS_TABLE_ID = var.ml_training_labels_table_id
    BQ_LOCATION                 = var.region
    MODEL_ID                    = "${replace(lower(var.prefix), "-", "_")}_credit_bqml_v1"
    REASON_RULE_VERSION         = "bnm-fair-lending-v1.0"
    ALERT_TOPIC_PATH            = google_pubsub_topic.alerts.id
  }

  depends_on = [
    google_project_iam_member.workflow_bq_job_user,
    google_bigquery_dataset_iam_member.workflow_silver_viewer,
    google_bigquery_dataset_iam_member.workflow_audit_editor,
    google_project_iam_member.workflow_aiplatform_user,
    google_pubsub_topic_iam_member.workflow_alert_publisher,
    google_kms_crypto_key_iam_member.workflow_cmek,
  ]
}

resource "google_cloud_scheduler_job" "bqml_schedule" {
  count = var.enable_bqml_schedule ? 1 : 0

  project          = var.ml_ops_project_id
  region           = var.region
  name             = local.scheduler_name
  description      = "Scheduled Track A BQML training and ML.EXPLAIN_PREDICT batch run."
  schedule         = var.bqml_schedule_cron
  time_zone        = var.schedule_time_zone
  attempt_deadline = "320s"

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/${google_workflows_workflow.bqml_xai.id}/executions"
    body        = base64encode(jsonencode({ argument = jsonencode({ train_model = true }) }))

    oauth_token {
      service_account_email = google_service_account.trigger[0].email
      scope                 = "https://www.googleapis.com/auth/cloud-platform"
    }
  }

  depends_on = [google_project_iam_member.trigger_workflows_invoker]
}

resource "google_monitoring_alert_policy" "bqml_failure" {
  count = length(local.same_project_channels) > 0 ? 1 : 0

  project               = var.ml_ops_project_id
  display_name          = "${var.prefix} BQML XAI Workflow Failure"
  combiner              = "OR"
  notification_channels = local.same_project_channels

  conditions {
    display_name = "Workflow execution failed"
    condition_threshold {
      filter          = "resource.type=\"workflows.googleapis.com/Workflow\" AND resource.labels.workflow_id=\"${google_workflows_workflow.bqml_xai.name}\" AND metric.type=\"workflows.googleapis.com/finished_execution_count\" AND metric.labels.status!=\"SUCCEEDED\""
      duration        = "0s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }
}
