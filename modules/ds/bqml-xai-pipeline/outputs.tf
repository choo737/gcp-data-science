output "workflow_id" {
  value       = google_workflows_workflow.bqml_xai.id
  description = "Full resource ID of the Track A BQML XAI Cloud Workflow."
}

output "workflow_name" {
  value       = google_workflows_workflow.bqml_xai.name
  description = "Short name of the Track A BQML XAI Cloud Workflow."
}

output "scheduler_job_name" {
  value       = try(google_cloud_scheduler_job.bqml_schedule[0].name, "")
  description = "Cloud Scheduler job name when Track A scheduling is enabled."
}

output "alert_topic_id" {
  value       = google_pubsub_topic.alerts.id
  description = "Pub/Sub topic ID receiving Track A pipeline failure notifications."
}

output "workflow_service_account_email" {
  value       = google_service_account.workflow.email
  description = "Dedicated service account email running Track A BigQuery ML jobs."
}
