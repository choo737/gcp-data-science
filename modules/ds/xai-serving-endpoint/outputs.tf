output "vertex_endpoint_id" {
  value       = google_vertex_ai_endpoint.this.id
  description = "Full resource ID of the Vertex AI Endpoint."
}

output "vertex_endpoint_name" {
  value       = google_vertex_ai_endpoint.this.name
  description = "Numeric endpoint ID used in :predict requests."
}

output "scorer_service_name" {
  value       = try(google_cloud_run_v2_service.scorer[0].name, "")
  description = "Cloud Run v2 XAI scorer service name when enabled."
}

output "scorer_service_uri" {
  value       = try(google_cloud_run_v2_service.scorer[0].uri, "")
  description = "Internal Cloud Run v2 XAI scorer service URI when enabled."
}

output "scorer_service_account_email" {
  value       = try(google_service_account.scorer[0].email, "")
  description = "Dedicated service account email used by the XAI scorer service."
}

output "psc_endpoint_ip" {
  value       = local.psc_endpoint_ip
  description = "Private Service Connect endpoint IP when automated or manually configured."
}

output "cpr_deployer_job_name" {
  value       = try(google_cloud_run_v2_job.cpr_deployer[0].name, "")
  description = "Cloud Run v2 job name that uploads and deploys the CPR model when enabled."
}
