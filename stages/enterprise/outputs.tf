output "environment" {
  value       = var.environment
  description = "Active deployment environment."
}

output "feature_group_id" {
  value       = module.feature_store.feature_group_id
  description = "Vertex AI Feature Group resource ID."
}

output "feature_online_store_id" {
  value       = module.feature_store.feature_online_store_id
  description = "Vertex AI Bigtable Feature Online Store resource ID when enabled."
}

output "feature_view_id" {
  value       = module.feature_store.feature_view_id
  description = "Vertex AI FeatureView resource ID when online serving is enabled."
}

output "ml_xai_audit_ledger_id" {
  value       = google_bigquery_table.ml_xai_audit_ledger.id
  description = "Qualified BigQuery resource ID for ml_xai_audit_ledger."
}

output "vertex_endpoint_id" {
  value       = try(module.xai_serving_endpoint[0].vertex_endpoint_id, "")
  description = "Vertex AI Endpoint resource ID when online serving track is active."
}

output "vertex_endpoint_name" {
  value       = try(module.xai_serving_endpoint[0].vertex_endpoint_name, "")
  description = "Numeric Vertex AI Endpoint ID when online serving track is active."
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
  description = "Cloud Workflows resource ID for Track A BQML XAI runs."
}

output "bqml_workflow_name" {
  value       = try(module.bqml_xai_pipeline[0].workflow_name, "")
  description = "Cloud Workflows short name for Track A BQML XAI runs."
}

output "bqml_alert_topic_id" {
  value       = try(module.bqml_xai_pipeline[0].alert_topic_id, "")
  description = "Pub/Sub topic ID receiving Track A pipeline failure alerts."
}
