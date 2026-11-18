variable "project_id" {
  type        = string
  description = "Existing single sandbox GCP project ID."
}

variable "region" {
  type        = string
  default     = "asia-southeast1"
  description = "GCP region for Vertex AI, Cloud Run, Workflows, and BigQuery."
}

variable "prefix" {
  type        = string
  default     = "bqfnd"
  description = "Short sandbox resource naming prefix."
}

variable "silver_dataset_id" {
  type        = string
  default     = ""
  description = "Existing Silver dataset ID (defaults to <prefix>_silver_curated)."
}

variable "create_silver_dataset" {
  type        = bool
  default     = false
  description = "Create the sandbox Silver dataset when not provisioned upstream."
}

variable "ml_audit_dataset_id" {
  type        = string
  default     = ""
  description = "Existing ML XAI audit dataset ID (defaults to <prefix>_ml_xai_audit)."
}

variable "create_ml_audit_dataset" {
  type        = bool
  default     = true
  description = "Create the sandbox ML XAI audit dataset when not supplied."
}

variable "ml_audit_table_id" {
  type        = string
  default     = "ml_xai_audit_ledger"
  description = "BigQuery table ID for the 18-column XAI audit ledger."
}

variable "feature_group_source_table" {
  type        = string
  default     = "ml_customer_features"
  description = "BigQuery Silver table backing the Vertex AI Feature Group."
}

variable "ds_serving_track" {
  type        = string
  default     = "bqml_batch"
  description = "Active serving track: bqml_batch, vertex_cpr_online, or both."

  validation {
    condition     = contains(["bqml_batch", "vertex_cpr_online", "both"], var.ds_serving_track)
    error_message = "ds_serving_track must be one of bqml_batch, vertex_cpr_online, or both."
  }
}

variable "enable_online_feature_store" {
  type        = bool
  default     = false
  description = "Provision Bigtable Feature Online Store and FeatureView."
}

variable "feature_online_store_min_nodes" {
  type        = number
  default     = 1
  description = "Minimum Bigtable autoscaling node count."
}

variable "feature_online_store_max_nodes" {
  type        = number
  default     = 2
  description = "Maximum Bigtable autoscaling node count."
}

variable "feature_sync_cron" {
  type        = string
  default     = "TZ=Asia/Singapore 0 * * * *"
  description = "Cron schedule for FeatureView online sync."
}

variable "enable_scorer_service" {
  type        = bool
  default     = false
  description = "Provision the Cloud Run v2 XAI scorer service."
}

variable "scorer_container_image" {
  type        = string
  default     = ""
  description = "Container image URI for the Cloud Run scorer service."
}

variable "deploy_cpr_model" {
  type        = bool
  default     = false
  description = "Validate and bind the CPR TreeSHAP container image."
}

variable "cpr_container_image" {
  type        = string
  default     = ""
  description = "Container image URI for the CPR TreeSHAP predictor."
}

variable "enable_bqml_schedule" {
  type        = bool
  default     = false
  description = "Provision Cloud Scheduler trigger for Track A BQML XAI runs."
}

variable "bqml_schedule_cron" {
  type        = string
  default     = "0 2 * * *"
  description = "Cron schedule for Track A BQML XAI runs."
}

variable "shap_additivity_tolerance" {
  type        = number
  default     = 0.00001
  description = "Maximum allowed TreeSHAP log-odds additivity drift."
}

variable "kms_key_id" {
  type        = string
  default     = ""
  description = "Optional regional Cloud KMS CryptoKey ID for CMEK."
}

variable "vpc_id" {
  type        = string
  default     = ""
  description = "Optional VPC network ID for Cloud Run Direct VPC Egress."
}

variable "egress_connector_subnet_id" {
  type        = string
  default     = ""
  description = "Optional subnetwork ID for Cloud Run Direct VPC Egress."
}

variable "alert_notification_channel_ids" {
  type        = list(string)
  default     = []
  description = "Optional Cloud Monitoring notification channel IDs."
}

variable "delete_contents_on_destroy" {
  type        = bool
  default     = false
  description = "Allow deletion of sandbox datasets and tables on destroy."
}

variable "labels" {
  type        = map(string)
  default     = {}
  description = "Additional GCP labels applied to sandbox resources."
}
