locals {
  sanitized_prefix             = replace(lower(var.prefix), "_", "-")
  raw_digits                   = join("", regexall("[0-9]", md5("${var.ml_ops_project_id}-${var.prefix}-${var.environment}")))
  endpoint_numeric_id          = "1${substr("${local.raw_digits}000000000", 0, 9)}"
  scorer_service_name          = trimsuffix(substr("${local.sanitized_prefix}-xai-scorer", 0, 63), "-")
  scorer_sa_account_id         = trimsuffix(substr("${local.sanitized_prefix}-xai-scorer", 0, 28), "-")
  cpr_deployer_job_name        = trimsuffix(substr("${local.sanitized_prefix}-cpr-deployer", 0, 63), "-")
  cpr_deployer_sa_id           = trimsuffix(substr("${local.sanitized_prefix}-cpr-dep", 0, 28), "-")
  psc_address_name             = trimsuffix(substr("${local.sanitized_prefix}-vertex-psc-ip", 0, 63), "-")
  psc_forwarding_rule_name     = trimsuffix(substr("${local.sanitized_prefix}-vertex-psc-fr", 0, 63), "-")
  effective_kms_key_id         = var.kms_key_id != null ? trimspace(var.kms_key_id) : ""
  effective_worm_bucket        = var.ml_worm_bucket_name != null ? trimspace(var.ml_worm_bucket_name) : ""
  effective_scorer_image       = var.scorer_container_image != null ? trimspace(var.scorer_container_image) : ""
  effective_cpr_image          = var.cpr_container_image != null ? trimspace(var.cpr_container_image) : ""
  effective_cpr_artifact_uri   = var.cpr_model_artifact_uri != null ? trimspace(var.cpr_model_artifact_uri) : ""
  effective_cpr_model_sha256   = var.cpr_model_sha256 != null ? trimspace(var.cpr_model_sha256) : ""
  effective_cpr_deployer_image = var.cpr_deployer_image != null ? trimspace(var.cpr_deployer_image) : ""
  resolved_cpr_deployer_image  = local.effective_cpr_deployer_image != "" ? local.effective_cpr_deployer_image : local.effective_cpr_image
  cpr_model_bucket             = local.effective_cpr_artifact_uri != "" ? try(regex("^gs://([^/]+)", local.effective_cpr_artifact_uri)[0], "") : ""
  effective_vpc_id             = var.vpc_id != null ? trimspace(var.vpc_id) : ""
  effective_subnet_id          = var.egress_connector_subnet_id != null ? trimspace(var.egress_connector_subnet_id) : ""
  effective_psc_attachment     = var.psc_service_attachment != null ? trimspace(var.psc_service_attachment) : ""
  effective_psc_ip_override    = var.psc_endpoint_ip_override != null ? trimspace(var.psc_endpoint_ip_override) : ""
  use_direct_vpc               = local.effective_vpc_id != "" && local.effective_subnet_id != ""
  network_project_id           = local.effective_vpc_id != "" ? try(regex("^projects/([^/]+)/global/networks/", local.effective_vpc_id)[0], var.ml_ops_project_id) : var.ml_ops_project_id
  psc_project_allowlist        = distinct(compact([var.ml_ops_project_id, var.ds_exp_project_id, local.network_project_id]))
  raw_tap_bq_uri               = "bq://${var.ml_ops_project_id}.${var.ml_audit_dataset_id}.vertex_predict_raw_tap"
  psc_endpoint_ip_candidates = compact([
    local.effective_psc_ip_override,
    try(google_compute_address.psc_endpoint[0].address, ""),
    try(google_vertex_ai_endpoint.this.private_service_connect_config[0].psc_automation_configs[0].ip_address, ""),
  ])
  psc_endpoint_ip = length(local.psc_endpoint_ip_candidates) > 0 ? local.psc_endpoint_ip_candidates[0] : ""
}

resource "terraform_data" "validate_serving_inputs" {
  lifecycle {
    precondition {
      condition     = !var.enable_scorer_service || local.effective_scorer_image != ""
      error_message = "scorer_container_image is required when enable_scorer_service is true."
    }
    precondition {
      condition     = !var.enable_scorer_service || var.environment != "prod" || strcontains(local.effective_scorer_image, "@sha256:")
      error_message = "scorer_container_image must be pinned to an immutable @sha256 digest in prod."
    }
    precondition {
      condition     = !var.deploy_cpr_model || local.effective_cpr_image != ""
      error_message = "cpr_container_image is required when deploy_cpr_model is true."
    }
    precondition {
      condition     = !var.deploy_cpr_model || var.environment != "prod" || strcontains(local.effective_cpr_image, "@sha256:")
      error_message = "cpr_container_image must be pinned to an immutable @sha256 digest in prod."
    }
    precondition {
      condition     = !var.deploy_cpr_model || (local.effective_cpr_artifact_uri != "" && startswith(local.effective_cpr_artifact_uri, "gs://"))
      error_message = "cpr_model_artifact_uri must be a non-empty gs:// URI when deploy_cpr_model is true."
    }
    precondition {
      condition     = !var.deploy_cpr_model || can(regex("^[0-9a-fA-F]{64}$", local.effective_cpr_model_sha256))
      error_message = "cpr_model_sha256 must be a 64-character hex SHA-256 digest when deploy_cpr_model is true."
    }
  }
}

resource "google_vertex_ai_endpoint" "this" {
  project      = var.ml_ops_project_id
  region       = var.region
  location     = var.region
  name         = local.endpoint_numeric_id
  display_name = "${var.prefix}-credit-xai-endpoint"
  description  = "Private Service Connect Agent Platform endpoint for CPR TreeSHAP :predict serving."
  labels       = var.labels

  dynamic "encryption_spec" {
    for_each = local.effective_kms_key_id != "" ? [local.effective_kms_key_id] : []
    content {
      kms_key_name = encryption_spec.value
    }
  }

  private_service_connect_config {
    enable_private_service_connect = true
    project_allowlist              = local.psc_project_allowlist

    dynamic "psc_automation_configs" {
      for_each = local.use_direct_vpc ? [1] : []
      content {
        project_id = local.network_project_id
        network    = local.effective_vpc_id
      }
    }
  }

  predict_request_response_logging_config {
    enabled       = true
    sampling_rate = 1.0
    bigquery_destination {
      output_uri = local.raw_tap_bq_uri
    }
  }

  depends_on = [terraform_data.validate_serving_inputs]
}

resource "google_compute_address" "psc_endpoint" {
  count = local.use_direct_vpc && local.effective_psc_attachment != "" ? 1 : 0

  project      = local.network_project_id
  region       = var.region
  name         = local.psc_address_name
  address_type = "INTERNAL"
  purpose      = "GCE_ENDPOINT"
  subnetwork   = local.effective_subnet_id
}

resource "google_compute_forwarding_rule" "psc_endpoint" {
  count = local.use_direct_vpc && local.effective_psc_attachment != "" ? 1 : 0

  project                 = local.network_project_id
  region                  = var.region
  name                    = local.psc_forwarding_rule_name
  network                 = local.effective_vpc_id
  ip_address              = google_compute_address.psc_endpoint[0].id
  load_balancing_scheme   = ""
  target                  = local.effective_psc_attachment
  allow_psc_global_access = false
}

resource "google_service_account" "scorer" {
  count = var.enable_scorer_service ? 1 : 0

  project      = var.ml_ops_project_id
  account_id   = local.scorer_sa_account_id
  display_name = "${var.prefix} XAI Scorer"
  description  = "Least-privilege identity for fail-closed scoring and XAI ledger writes."
}

resource "google_project_iam_member" "scorer_aiplatform_user" {
  count = var.enable_scorer_service ? 1 : 0

  project = var.ml_ops_project_id
  role    = "roles/aiplatform.user"
  member  = google_service_account.scorer[0].member
}

resource "google_bigquery_dataset_iam_member" "scorer_audit_editor" {
  count = var.enable_scorer_service ? 1 : 0

  project    = var.ml_ops_project_id
  dataset_id = var.ml_audit_dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_service_account.scorer[0].member
}

resource "google_project_iam_member" "scorer_bq_job_user" {
  count = var.enable_scorer_service ? 1 : 0

  project = var.ml_ops_project_id
  role    = "roles/bigquery.jobUser"
  member  = google_service_account.scorer[0].member
}

resource "google_storage_bucket_iam_member" "scorer_worm_creator" {
  count = var.enable_scorer_service && local.effective_worm_bucket != "" ? 1 : 0

  bucket = local.effective_worm_bucket
  role   = "roles/storage.objectCreator"
  member = google_service_account.scorer[0].member
}

resource "google_kms_crypto_key_iam_member" "scorer_cmek" {
  count = var.enable_scorer_service && local.effective_kms_key_id != "" && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_service_account.scorer[0].member
}

resource "google_cloud_run_v2_service" "scorer" {
  count = var.enable_scorer_service ? 1 : 0

  project             = var.ml_ops_project_id
  location            = var.region
  name                = local.scorer_service_name
  ingress             = "INGRESS_TRAFFIC_INTERNAL_ONLY"
  deletion_protection = !var.delete_contents_on_destroy
  labels              = var.labels

  template {
    service_account = google_service_account.scorer[0].email
    encryption_key  = local.effective_kms_key_id != "" ? local.effective_kms_key_id : null

    dynamic "vpc_access" {
      for_each = local.use_direct_vpc ? [1] : []
      content {
        egress = var.environment == "prod" ? "ALL_TRAFFIC" : "PRIVATE_RANGES_ONLY"
        network_interfaces {
          network    = local.effective_vpc_id
          subnetwork = local.effective_subnet_id
        }
      }
    }

    containers {
      image = local.effective_scorer_image

      ports {
        container_port = 8080
      }

      env {
        name  = "ML_OPS_PROJECT_ID"
        value = var.ml_ops_project_id
      }
      env {
        name  = "REGION"
        value = var.region
      }
      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
      env {
        name  = "VERTEX_ENDPOINT_ID"
        value = google_vertex_ai_endpoint.this.name
      }
      env {
        name  = "VERTEX_PSC_ENDPOINT_IP"
        value = local.psc_endpoint_ip
      }
      env {
        name  = "ML_AUDIT_DATASET_ID"
        value = var.ml_audit_dataset_id
      }
      env {
        name  = "ML_AUDIT_TABLE_ID"
        value = var.ml_audit_table_id
      }
      env {
        name  = "ML_WORM_BUCKET_NAME"
        value = local.effective_worm_bucket
      }
      env {
        name  = "SHAP_ADDITIVITY_TOLERANCE"
        value = tostring(var.shap_additivity_tolerance)
      }
      env {
        name  = "SERVING_TRACK"
        value = var.serving_track
      }
      env {
        name  = "DEPLOY_CPR_MODEL"
        value = tostring(var.deploy_cpr_model)
      }
    }
  }

  depends_on = [
    terraform_data.validate_serving_inputs,
    google_project_iam_member.scorer_aiplatform_user,
    google_bigquery_dataset_iam_member.scorer_audit_editor,
    google_kms_crypto_key_iam_member.scorer_cmek,
  ]
}

resource "google_service_account" "cpr_deployer" {
  count = var.deploy_cpr_model ? 1 : 0

  project      = var.ml_ops_project_id
  account_id   = local.cpr_deployer_sa_id
  display_name = "${var.prefix} CPR Deployer"
  description  = "Least-privilege identity for uploading and deploying CPR models to Agent Platform."
}

resource "google_project_iam_member" "cpr_deployer_aiplatform_user" {
  count = var.deploy_cpr_model ? 1 : 0

  project = var.ml_ops_project_id
  role    = "roles/aiplatform.user"
  member  = google_service_account.cpr_deployer[0].member
}

resource "google_project_service_identity" "aiplatform_agent" {
  count    = var.deploy_cpr_model && local.cpr_model_bucket != "" ? 1 : 0
  provider = google-beta
  project  = var.ml_ops_project_id
  service  = "aiplatform.googleapis.com"
}

resource "google_storage_bucket_iam_member" "cpr_deployer_model_viewer" {
  count = var.deploy_cpr_model && local.cpr_model_bucket != "" ? 1 : 0

  bucket = local.cpr_model_bucket
  role   = "roles/storage.objectViewer"
  member = google_service_account.cpr_deployer[0].member
}

resource "google_storage_bucket_iam_member" "aiplatform_model_viewer" {
  count = var.deploy_cpr_model && local.cpr_model_bucket != "" ? 1 : 0

  bucket = local.cpr_model_bucket
  role   = "roles/storage.objectViewer"
  member = google_project_service_identity.aiplatform_agent[0].member
}

resource "google_kms_crypto_key_iam_member" "cpr_deployer_cmek" {
  count = var.deploy_cpr_model && local.effective_kms_key_id != "" && var.manage_kms_iam ? 1 : 0

  crypto_key_id = local.effective_kms_key_id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = google_service_account.cpr_deployer[0].member
}

resource "google_cloud_run_v2_job" "cpr_deployer" {
  count = var.deploy_cpr_model ? 1 : 0

  project             = var.ml_ops_project_id
  location            = var.region
  name                = local.cpr_deployer_job_name
  deletion_protection = !var.delete_contents_on_destroy
  labels              = var.labels

  template {
    template {
      service_account = google_service_account.cpr_deployer[0].email
      encryption_key  = local.effective_kms_key_id != "" ? local.effective_kms_key_id : null
      max_retries     = 1
      timeout         = "900s"

      dynamic "vpc_access" {
        for_each = local.use_direct_vpc ? [1] : []
        content {
          egress = var.environment == "prod" ? "ALL_TRAFFIC" : "PRIVATE_RANGES_ONLY"
          network_interfaces {
            network    = local.effective_vpc_id
            subnetwork = local.effective_subnet_id
          }
        }
      }

      containers {
        image   = local.resolved_cpr_deployer_image
        command = ["python3", "/app/deploy_cpr.py"]

        env {
          name  = "ML_OPS_PROJECT_ID"
          value = var.ml_ops_project_id
        }
        env {
          name  = "REGION"
          value = var.region
        }
        env {
          name  = "VERTEX_ENDPOINT_ID"
          value = google_vertex_ai_endpoint.this.name
        }
        env {
          name  = "CPR_CONTAINER_IMAGE"
          value = local.effective_cpr_image
        }
        env {
          name  = "CPR_MODEL_ARTIFACT_URI"
          value = local.effective_cpr_artifact_uri
        }
        env {
          name  = "EXPECTED_MODEL_SHA256"
          value = local.effective_cpr_model_sha256
        }
        env {
          name  = "FEATURE_NAMES"
          value = jsonencode(var.feature_names)
        }
        env {
          name  = "MODEL_DISPLAY_NAME"
          value = "${var.prefix}-credit-cpr-treeshap"
        }
        env {
          name  = "MODEL_FILENAME"
          value = "model.ubj"
        }
        env {
          name  = "REASON_RULE_VERSION"
          value = "bnm-fair-lending-v1.0"
        }
        env {
          name  = "SHAP_ADDITIVITY_TOLERANCE"
          value = tostring(var.shap_additivity_tolerance)
        }
        env {
          name  = "MODEL_VERSION_ID"
          value = substr(local.effective_cpr_model_sha256, 0, 12)
        }
        env {
          name  = "KMS_KEY_ID"
          value = local.effective_kms_key_id
        }
      }
    }
  }

  depends_on = [
    terraform_data.validate_serving_inputs,
    google_project_iam_member.cpr_deployer_aiplatform_user,
    google_storage_bucket_iam_member.cpr_deployer_model_viewer,
    google_storage_bucket_iam_member.aiplatform_model_viewer,
    google_kms_crypto_key_iam_member.cpr_deployer_cmek,
  ]
}