variable "ml_ops_project_id" {
  type        = string
  description = "GCP project ID hosting the Agent Platform (formerly Vertex AI) Endpoint and Cloud Run scorer."
}

variable "ds_exp_project_id" {
  type        = string
  description = "Data science experimentation project ID allowlisted for Private Service Connect."
}

variable "ml_audit_dataset_id" {
  type        = string
  description = "BigQuery dataset ID storing ml_xai_audit_ledger and raw prediction tap."
}

variable "ml_audit_table_id" {
  type        = string
  default     = "ml_xai_audit_ledger"
  description = "BigQuery table ID for synchronous XAI audit logging."
}

variable "ml_worm_bucket_name" {
  type        = string
  default     = ""
  description = "Optional immutable WORM GCS bucket name for decision audit receipts."
}

variable "region" {
  type        = string
  default     = "asia-southeast1"
  description = "GCP region for Agent Platform Endpoint and Cloud Run scorer service."
}

variable "prefix" {
  type        = string
  default     = "acme"
  description = "Resource naming prefix."
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Deployment environment (dev, uat, or prod)."
}

variable "enable_scorer_service" {
  type        = bool
  default     = true
  description = "Provision the fail-closed Cloud Run v2 XAI scorer service."
}

variable "scorer_container_image" {
  type        = string
  default     = ""
  description = "Container image URI (pinned to @sha256 in prod) for the XAI scorer service."
}

variable "deploy_cpr_model" {
  type        = bool
  default     = false
  description = "Validate and bind the Custom inference routine (formerly Custom Prediction Routine) TreeSHAP container image."
}

variable "cpr_container_image" {
  type        = string
  default     = ""
  description = "Container image URI (pinned to @sha256 in prod) for the CPR TreeSHAP predictor."
}

variable "cpr_model_artifact_uri" {
  type        = string
  default     = ""
  description = "GCS directory or artifact URI hosting the trained XGBoost model.ubj when deploy_cpr_model is true."
}

variable "cpr_model_sha256" {
  type        = string
  default     = ""
  description = "Expected lowercase hex SHA-256 digest of the CPR XGBoost model artifact."
}

variable "cpr_deployer_image" {
  type        = string
  default     = ""
  description = "Optional container image URI for the Cloud Run v2 CPR deployment job."
}

variable "feature_names" {
  type = list(string)
  default = [
    "tenure_months",
    "avg_monthly_spend",
    "rolling_order_count",
    "days_since_last_order",
    "cancellation_rate_30d",
    "discount_utilization_ratio",
  ]
  description = "Ordered feature list configured on the deployed CPR predictor container."
}

variable "psc_service_attachment" {
  type        = string
  default     = ""
  description = "Optional Private Service Connect service attachment URI for manual consumer forwarding rules."
}

variable "psc_endpoint_ip_override" {
  type        = string
  default     = ""
  description = "Optional explicit Private Service Connect endpoint IP or private prediction host."
}

variable "serving_track" {
  type        = string
  default     = "both"
  description = "Active serving track (bqml_batch, vertex_cpr_online, or both)."
}

variable "shap_additivity_tolerance" {
  type        = number
  default     = 0.00001
  description = "Maximum allowed TreeSHAP log-odds additivity drift before fail-closed rejection."
}

variable "kms_key_id" {
  type        = string
  default     = ""
  description = "Regional Cloud KMS CryptoKey ID for endpoint and Cloud Run CMEK."
}

variable "manage_kms_iam" {
  type        = bool
  default     = true
  description = "Grant KMS CryptoKey encrypter/decrypter IAM bindings to module service accounts."
}

variable "vpc_id" {
  type        = string
  default     = ""
  description = "VPC network ID for Cloud Run Direct VPC Egress."
}

variable "egress_connector_subnet_id" {
  type        = string
  default     = ""
  description = "Subnet ID for Cloud Run Direct VPC Egress."
}

variable "delete_contents_on_destroy" {
  type        = bool
  default     = false
  description = "Allow deletion of protected resources when true."
}

variable "labels" {
  type        = map(string)
  default     = {}
  description = "Resource labels applied to endpoint and Cloud Run resources."
}
