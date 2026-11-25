locals {
  sanitized_prefix     = replace(lower(var.prefix), "-", "_")
  feature_group_name   = trimsuffix(substr("${local.sanitized_prefix}_customer_features", 0, 64), "_")
  online_store_name    = trimsuffix(substr("${local.sanitized_prefix}_online_store", 0, 64), "_")
  effective_kms_key_id = var.kms_key_id != null ? trimspace(var.kms_key_id) : ""
  feature_source_uri   = "bq://${var.core_project_id}.${var.silver_dataset_id}.${var.feature_group_source_table}"
}

resource "terraform_data" "validate_online_store_region" {
  lifecycle {
    precondition {
      condition     = !var.enable_online_feature_store || var.region != "asia-southeast2"
      error_message = "Agent Platform Feature Store online serving is not supported in asia-southeast2. Use asia-southeast1 or set enable_online_feature_store = false."
    }
    precondition {
      condition     = var.feature_online_store_min_nodes >= 1 && var.feature_online_store_max_nodes >= var.feature_online_store_min_nodes
      error_message = "feature_online_store_min_nodes must be >= 1 and <= feature_online_store_max_nodes."
    }
  }
}

resource "google_vertex_ai_feature_group" "customer_features" {
  project     = var.ml_ops_project_id
  region      = var.region
  name        = local.feature_group_name
  description = "Zero-copy BigQuery Feature Group backed by Silver customer features."
  labels      = var.labels

  big_query {
    entity_id_columns = var.entity_id_columns
    big_query_source {
      input_uri = local.feature_source_uri
    }
  }
}

resource "google_vertex_ai_feature_group_feature" "features" {
  for_each = var.feature_definitions

  project             = var.ml_ops_project_id
  region              = var.region
  feature_group       = google_vertex_ai_feature_group.customer_features.name
  name                = each.key
  description         = each.value.description
  version_column_name = trimspace(each.value.version_column_name) != "" ? trimspace(each.value.version_column_name) : null
  labels              = var.labels
}

resource "google_vertex_ai_feature_online_store" "online" {
  count = var.enable_online_feature_store ? 1 : 0

  project       = var.ml_ops_project_id
  region        = var.region
  name          = local.online_store_name
  force_destroy = var.delete_contents_on_destroy
  labels        = var.labels

  bigtable {
    auto_scaling {
      min_node_count         = var.feature_online_store_min_nodes
      max_node_count         = var.feature_online_store_max_nodes
      cpu_utilization_target = 50
    }
  }

  dynamic "encryption_spec" {
    for_each = local.effective_kms_key_id != "" ? [local.effective_kms_key_id] : []
    content {
      kms_key_name = encryption_spec.value
    }
  }

  depends_on = [terraform_data.validate_online_store_region]
}

resource "google_vertex_ai_feature_online_store_featureview" "customer_view" {
  count = var.enable_online_feature_store ? 1 : 0

  project              = var.ml_ops_project_id
  region               = var.region
  feature_online_store = google_vertex_ai_feature_online_store.online[0].name
  name                 = "customer_features_view"
  labels               = var.labels

  sync_config {
    cron = var.feature_sync_cron
  }

  feature_registry_source {
    project_number = var.ml_ops_project_number
    feature_groups {
      feature_group_id = google_vertex_ai_feature_group.customer_features.name
      feature_ids      = [for k in sort(keys(var.feature_definitions)) : k]
    }
  }

  depends_on = [google_vertex_ai_feature_group_feature.features]
}
