output "feature_group_id" {
  value       = module.feature_store.feature_group_id
  description = "Sandbox Vertex AI Feature Group resource ID."
}

output "feature_online_store_id" {
  value       = module.feature_store.feature_online_store_id
  description = "Sandbox Bigtable Feature Online Store resource ID when enabled."
}

output "ml_xai_audit_ledger_id" {
  value       = google_bigquery_table.ml_xai_audit_ledger.id
  description = "Qualified BigQuery resource ID for ml_xai_audit_ledger."
}

output "vertex_endpoint_id" {
  value       = try(module.xai_serving_endpoint[0].vertex_endpoint_id, "")
  description = "Sandbox Vertex AI Endpoint resource ID when online track is active."
}

output "scorer_service_uri" {
  value       = try(module.xai_serving_endpoint[0].scorer_service_uri, "")
  description = "Sandbox Cloud Run XAI scorer URI when enabled."
}

output "bqml_workflow_id" {
  value       = try(module.bqml_xai_pipeline[0].workflow_id, "")
  description = "Sandbox Cloud Workflows resource ID for Track A BQML XAI runs."
}

output "bqml_workflow_name" {
  value       = try(module.bqml_xai_pipeline[0].workflow_name, "")
  description = "Sandbox Cloud Workflows short name for Track A BQML XAI runs."
}
