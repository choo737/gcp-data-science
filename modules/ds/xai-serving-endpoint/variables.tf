variable "ml_ops_project_id" {
  type        = string
  description = "GCP project ID hosting the Vertex AI Endpoint and Cloud Run scorer."
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
  description = "GCP region for Vertex AI Endpoint and Cloud Run scorer service."
}

variable "prefix" {
  type        = string
  default     = "acme"
  description = "Resource naming prefix."
}

variable "environment" {
  type        = string
  default     = "dev"
  description = "Deployment environment (dev, staging, or prod)."
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
  description = "Validate and bind the Custom Prediction Routine (CPR) TreeSHAP container image."
}

variable "cpr_container_image" {
  type        = string
  default     = ""
  description = "Container image URI (pinned to @sha256 in prod) for the CPR TreeSHAP predictor."
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
