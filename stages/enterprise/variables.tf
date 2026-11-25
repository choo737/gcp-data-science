variable "alert_notification_channel_ids" {
  type        = list(string)
  default     = []
  description = "Existing Cloud Monitoring notification channel IDs for pipeline failure alerts."
}

variable "bqml_schedule_cron" {
  type        = string
  default     = "0 2 * * *"
  description = "Cron schedule for Track A BQML training and batch ML.EXPLAIN_PREDICT scoring."
}

variable "core_project_id" {
  type        = string
  description = "Upstream GCP project ID hosting Silver feature tables."
}

variable "cpr_container_image" {
  type        = string
  default     = ""
  description = "Artifact Registry container image URI for the Custom inference routine (formerly Custom Prediction Routine) TreeSHAP predictor."
}

variable "cpr_model_artifact_uri" {
  type        = string
  default     = ""
  description = "GCS directory URI (gs://bucket/dir) or direct .ubj file URI (gs://bucket/dir/model.ubj) when deploy_cpr_model is true."

  validation {
    condition     = var.cpr_model_artifact_uri == null || trimspace(var.cpr_model_artifact_uri) == "" || startswith(trimspace(var.cpr_model_artifact_uri), "gs://")
    error_message = "cpr_model_artifact_uri must be empty or a gs:// directory or .ubj artifact URI."
  }
}

variable "cpr_model_sha256" {
  type        = string
  default     = ""
  description = "Expected lowercase hex SHA-256 digest of the CPR XGBoost model artifact."
}

variable "create_ml_audit_ledger_table" {
  type        = bool
  default     = false
  description = "Create ml_xai_audit_ledger and ml_training_labels in Brownfield deployments without Greenfield 03-data-science."
}

variable "deploy_cpr_model" {
  type        = bool
  default     = false
  description = "Validate and bind the Custom inference routine TreeSHAP container image."
}

variable "domain_name" {
  type        = string
  default     = ""
  description = "Optional business-unit domain slug for multi-BU state namespacing."
}

variable "ds_exp_project_id" {
  type        = string
  description = "Data science experimentation project ID allowlisted for Agent Platform Private Service Connect."
}

variable "ds_serving_track" {
  type        = string
  default     = "both"
  description = "Active explainable serving track: bqml_batch, vertex_cpr_online, or both."

  validation {
    condition     = contains(["bqml_batch", "vertex_cpr_online", "both"], var.ds_serving_track)
    error_message = "ds_serving_track must be one of bqml_batch, vertex_cpr_online, or both."
  }
}

variable "egress_connector_subnet_id" {
  type        = string
  default     = ""
  description = "Existing subnetwork ID for Cloud Run Direct VPC Egress."
}

variable "enable_bqml_schedule" {
  type        = bool
  default     = true
  description = "Provision Cloud Scheduler trigger for the Track A BQML XAI workflow."
}

variable "enable_online_feature_store" {
  type        = bool
  default     = false
  description = "Provision Bigtable-backed Agent Platform Feature Online Store and FeatureView."
}

variable "enable_scorer_service" {
  type        = bool
  default     = true
  description = "Provision the fail-closed Cloud Run v2 XAI scorer service."
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Target deployment environment: dev, uat, or prod."

  validation {
    condition     = contains(["dev", "uat", "prod"], var.environment)
    error_message = "environment must be one of dev, uat, or prod."
  }
}

variable "fairness_min_disparate_impact" {
  type        = number
  default     = 0.80
  description = "Minimum subgroup disparate-impact ratio enforced by the MRM evaluation gate."
}

variable "feature_group_source_table" {
  type        = string
  default     = "vw_ml_customer_features_online"
  description = "BigQuery Silver view or table backing the Agent Platform Feature Group."
}

variable "feature_online_store_max_nodes" {
  type        = number
  default     = 2
  description = "Maximum Bigtable autoscaling node count for Feature Online Store."
}

variable "feature_online_store_min_nodes" {
  type        = number
  default     = 1
  description = "Minimum Bigtable autoscaling node count for Feature Online Store."
}

variable "feature_sync_cron" {
  type        = string
  default     = "TZ=Asia/Singapore 0 * * * *"
  description = "Cron schedule for syncing FeatureView from BigQuery to Bigtable."
}

variable "kms_key_id" {
  type        = string
  default     = ""
  description = "Regional Cloud KMS CryptoKey resource ID for CMEK encryption."
}

variable "labels" {
  type        = map(string)
  default     = {}
  description = "Additional GCP labels merged onto provisioned resources."
}

variable "manage_kms_iam" {
  type        = bool
  default     = true
  description = "Grant KMS CryptoKey encrypter/decrypter IAM bindings for service identities and SAs."
}

variable "ml_artifact_repo_id" {
  type        = string
  description = "Artifact Registry repository ID or resource path storing CPR and scorer images."
}

variable "ml_audit_dataset_id" {
  type        = string
  description = "BigQuery dataset ID hosting ml_xai_audit_ledger and BQML models."
}

variable "ml_audit_table_id" {
  type        = string
  default     = "ml_xai_audit_ledger"
  description = "BigQuery table ID for the immutable 18-column XAI audit ledger."
}

variable "ml_ops_project_id" {
  type        = string
  description = "GCP project ID hosting Agent Platform (formerly Vertex AI) Feature Store, Endpoint, Scorer, and BQML ops."
}

variable "ml_worm_bucket_name" {
  type        = string
  default     = ""
  description = "Optional immutable WORM GCS bucket name for decision audit receipts."
}

variable "prefix" {
  type        = string
  default     = "acme"
  description = "Short organizational resource naming prefix."
}

variable "region" {
  type        = string
  default     = "asia-southeast1"
  description = "GCP region for ML Ops, Agent Platform, Cloud Run, and BigQuery resources."
}

variable "scorer_container_image" {
  type        = string
  default     = ""
  description = "Artifact Registry container image URI for the Cloud Run XAI scorer service."
}

variable "shap_additivity_tolerance" {
  type        = number
  default     = 0.00001
  description = "Maximum allowed TreeSHAP log-odds additivity drift before fail-closed rejection."
}

variable "silver_dataset_id" {
  type        = string
  description = "Upstream BigQuery Silver dataset ID hosting vw_ml_customer_features_online."
}

variable "vpc_id" {
  type        = string
  default     = ""
  description = "Existing VPC network ID for Cloud Run Direct VPC Egress."
}
