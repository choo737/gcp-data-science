output "feature_group_id" {
  value       = google_vertex_ai_feature_group.customer_features.id
  description = "Full resource ID of the Agent Platform Feature Group."
}

output "feature_group_name" {
  value       = google_vertex_ai_feature_group.customer_features.name
  description = "Short name of the Agent Platform Feature Group."
}

output "feature_online_store_id" {
  value       = try(google_vertex_ai_feature_online_store.online[0].id, "")
  description = "Full resource ID of the Bigtable Feature Online Store when enabled."
}

output "feature_online_store_name" {
  value       = try(google_vertex_ai_feature_online_store.online[0].name, "")
  description = "Short name of the Bigtable Feature Online Store when enabled."
}

output "feature_view_id" {
  value       = try(google_vertex_ai_feature_online_store_featureview.customer_view[0].id, "")
  description = "Full resource ID of the synced FeatureView when online serving is enabled."
}
