variable "ml_ops_project_id" {
  type        = string
  description = "GCP project ID running BigQuery ML jobs, Workflows, and Scheduler."
}

variable "core_project_id" {
  type        = string
  description = "GCP project ID hosting Silver feature tables."
}

variable "silver_dataset_id" {
  type        = string
  description = "BigQuery Silver dataset ID hosting vw_ml_customer_features_online."
}

variable "feature_group_source_table" {
  type        = string
  default     = "vw_ml_customer_features_online"
  description = "BigQuery Silver feature view or table name."
}

variable "ml_audit_dataset_id" {
  type        = string
  description = "BigQuery dataset ID storing BQML models and ml_xai_audit_ledger."
}

variable "ml_audit_table_id" {
  type        = string
  default     = "ml_xai_audit_ledger"
  description = "BigQuery table ID for immutable XAI audit records."
}

variable "ml_training_labels_table_id" {
  type        = string
  default     = "ml_training_labels"
  description = "BigQuery table ID storing point-in-time ground-truth training labels."
}

variable "region" {
  type        = string
  default     = "asia-southeast1"
  description = "GCP region for Workflows, Scheduler, and BigQuery job execution."
}

variable "prefix" {
  type        = string
  default     = "acme"
  description = "Naming prefix for BQML workflow and scheduler resources."
}

variable "enable_bqml_schedule" {
  type        = bool
  default     = true
  description = "Provision Cloud Scheduler job to trigger the Track A BQML XAI workflow."
}

variable "bqml_schedule_cron" {
  type        = string
  default     = "0 2 * * *"
  description = "Cron schedule for Track A BQML training and batch XAI scoring."
}

variable "schedule_time_zone" {
  type        = string
  default     = "Asia/Kuala_Lumpur"
  description = "IANA time zone for the Cloud Scheduler job."
}

variable "kms_key_id" {
  type        = string
  default     = ""
  description = "Regional Cloud KMS CryptoKey ID for Workflows and Pub/Sub alerts CMEK."
}

variable "manage_kms_iam" {
  type        = bool
  default     = true
  description = "Grant KMS CryptoKey encrypter/decrypter IAM bindings to module service accounts."
}

variable "alert_notification_channel_ids" {
  type        = list(string)
  default     = []
  description = "Existing Cloud Monitoring notification channel IDs for pipeline failure alerts."
}

variable "delete_contents_on_destroy" {
  type        = bool
  default     = false
  description = "Allow deletion of protected workflow resources when true."
}

variable "labels" {
  type        = map(string)
  default     = {}
  description = "Resource labels applied to Track A workflow resources."
}
