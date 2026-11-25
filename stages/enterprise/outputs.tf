output "environment" {
  value       = var.environment
  description = "Active deployment environment."
}

output "feature_group_id" {
  value       = module.feature_store.feature_group_id
  description = "Agent Platform Feature Group resource ID."
}

output "feature_online_store_id" {
  value       = module.feature_store.feature_online_store_id
  description = "Agent Platform Bigtable Feature Online Store resource ID when enabled."
}

output "feature_view_id" {
  value       = module.feature_store.feature_view_id
  description = "Agent Platform FeatureView resource ID when online serving is enabled."
}

output "ml_ops_project_id" {
  value       = var.ml_ops_project_id
  description = "GCP project ID hosting ML Ops resources."
}

output "region" {
  value       = var.region
  description = "Active deployment region."
}

output "ml_xai_audit_ledger_id" {
  value       = try(google_bigquery_table.ml_xai_audit_ledger[0].id, "projects/${var.ml_ops_project_id}/datasets/${var.ml_audit_dataset_id}/tables/${var.ml_audit_table_id}")
  description = "Qualified BigQuery resource ID for ml_xai_audit_ledger."
}

output "vertex_endpoint_id" {
  value       = try(module.xai_serving_endpoint[0].vertex_endpoint_id, "")
  description = "Agent Platform Endpoint resource ID when online serving track is active."
}

output "vertex_endpoint_name" {
  value       = try(module.xai_serving_endpoint[0].vertex_endpoint_name, "")
  description = "Numeric Agent Platform Endpoint ID when online serving track is active."
}

output "psc_endpoint_ip" {
  value       = try(module.xai_serving_endpoint[0].psc_endpoint_ip, "")
  description = "Private Service Connect endpoint IP when online serving track is active."
}

output "cpr_deployer_job_name" {
  value       = try(module.xai_serving_endpoint[0].cpr_deployer_job_name, "")
  description = "Cloud Run v2 job name that uploads and deploys the CPR model when enabled."
}

output "scorer_service_name" {
  value       = try(module.xai_serving_endpoint[0].scorer_service_name, "")
  description = "Internal Cloud Run v2 XAI scorer service name when enabled."
}

output "scorer_service_uri" {
  value       = try(module.xai_serving_endpoint[0].scorer_service_uri, "")
  description = "Internal Cloud Run v2 XAI scorer service URI when enabled."
}

output "scorer_service_account_email" {
  value       = try(module.xai_serving_endpoint[0].scorer_service_account_email, "")
  description = "Service account email assigned to the XAI scorer service."
}

output "bqml_workflow_id" {
  value       = try(module.bqml_xai_pipeline[0].workflow_id, "")
  description = "Workflows resource ID for Track A BQML XAI runs."
}

output "bqml_workflow_name" {
  value       = try(module.bqml_xai_pipeline[0].workflow_name, "")
  description = "Workflows short name for Track A BQML XAI runs."
}

output "bqml_alert_topic_id" {
  value       = try(module.bqml_xai_pipeline[0].alert_topic_id, "")
  description = "Pub/Sub topic ID receiving Track A pipeline failure alerts."
}
