variable "ml_ops_project_id" {
  type        = string
  description = "GCP project ID hosting Vertex AI Feature Store resources."
}

variable "ml_ops_project_number" {
  type        = string
  description = "Numeric project number for ml_ops_project_id."
}

variable "core_project_id" {
  type        = string
  description = "GCP project ID hosting the BigQuery Silver feature table."
}

variable "silver_dataset_id" {
  type        = string
  description = "BigQuery Silver dataset ID hosting the feature source table."
}

variable "feature_group_source_table" {
  type        = string
  default     = "vw_ml_customer_features_online"
  description = "BigQuery Silver view or table backing the Feature Group."
}

variable "entity_id_columns" {
  type        = list(string)
  default     = ["entity_id"]
  description = "Entity ID columns used to key the Vertex AI Feature Group."
}

variable "region" {
  type        = string
  default     = "asia-southeast1"
  description = "Vertex AI Feature Store region."
}

variable "prefix" {
  type        = string
  default     = "acme"
  description = "Naming prefix for Feature Store resources."
}

variable "enable_online_feature_store" {
  type        = bool
  default     = false
  description = "Provision Bigtable-backed Feature Online Store and FeatureView when true."
}

variable "feature_online_store_min_nodes" {
  type        = number
  default     = 1
  description = "Minimum Bigtable autoscaling node count for Feature Online Store."
}

variable "feature_online_store_max_nodes" {
  type        = number
  default     = 2
  description = "Maximum Bigtable autoscaling node count for Feature Online Store."
}

variable "feature_sync_cron" {
  type        = string
  default     = "TZ=Asia/Singapore 0 * * * *"
  description = "Cron schedule for FeatureView online sync."
}

variable "kms_key_id" {
  type        = string
  default     = ""
  description = "Regional Cloud KMS CryptoKey ID for Feature Online Store CMEK."
}

variable "delete_contents_on_destroy" {
  type        = bool
  default     = false
  description = "Allow force destruction of Feature Online Store on terraform destroy."
}

variable "labels" {
  type        = map(string)
  default     = {}
  description = "Resource labels applied to Feature Store resources."
}

variable "feature_definitions" {
  type = map(object({
    description         = string
    version_column_name = optional(string, "")
  }))
  default = {
    tenure_months = {
      description = "Customer relationship tenure in completed calendar months."
    }
    avg_monthly_spend = {
      description = "Trailing 30-day rolling net transaction volume in reporting currency."
    }
    rolling_order_count = {
      description = "Verified completed order count in the trailing 30-day window."
    }
    days_since_last_order = {
      description = "Calendar days elapsed since the last completed transaction."
    }
    cancellation_rate_30d = {
      description = "Share of trailing 30-day orders cancelled or reversed."
    }
    discount_utilization_ratio = {
      description = "Promotional discount amount divided by gross trailing 30-day spend."
    }
  }
  description = "Map of feature IDs registered inside the Vertex AI Feature Group."
}
