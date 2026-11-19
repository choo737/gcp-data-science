"""Idempotent Vertex AI REST v1 uploader and deployer for CPR TreeSHAP models."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Mapping

_AIP_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def parse_deploy_config(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Parse and validate required CPR deployment environment variables."""
    env = os.environ if environ is None else environ
    project_id = env.get("ML_OPS_PROJECT_ID", "").strip()
    region = env.get("REGION", "asia-southeast1").strip()
    endpoint_id = env.get("VERTEX_ENDPOINT_ID", "").strip()
    cpr_image = env.get("CPR_CONTAINER_IMAGE", "").strip()
    artifact_uri = env.get("CPR_MODEL_ARTIFACT_URI", "").strip()
    expected_sha256 = env.get("EXPECTED_MODEL_SHA256", "").strip().lower()
    raw_features = env.get("FEATURE_NAMES", "").strip()
    display_name = env.get("MODEL_DISPLAY_NAME", "credit-scoring-cpr-treeshap").strip()
    model_filename = env.get("MODEL_FILENAME", "model.ubj").strip() or "model.ubj"
    reason_rule_version = env.get("REASON_RULE_VERSION", "bnm-fair-lending-v1.0").strip()
    additivity_tolerance = env.get("SHAP_ADDITIVITY_TOLERANCE", "0.00001").strip()
    machine_type = env.get("DEPLOY_MACHINE_TYPE", "n2-standard-4").strip()

    if not project_id or not endpoint_id or not cpr_image:
        raise ValueError("ML_OPS_PROJECT_ID, VERTEX_ENDPOINT_ID, and CPR_CONTAINER_IMAGE are required.")
    if not artifact_uri.startswith("gs://"):
        raise ValueError(f"CPR_MODEL_ARTIFACT_URI must be a gs:// URI, got: {artifact_uri!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("EXPECTED_MODEL_SHA256 must be a 64-character hex SHA-256 digest.")
    if not raw_features:
        raise ValueError("FEATURE_NAMES must be non-empty.")

    if raw_features.startswith("["):
        features = [str(f).strip() for f in json.loads(raw_features) if str(f).strip()]
    else:
        features = [f.strip() for f in raw_features.split(",") if f.strip()]
    if not features:
        raise ValueError("FEATURE_NAMES parsed to an empty list.")

    return {
        "project_id": project_id,
        "region": region,
        "endpoint_id": endpoint_id,
        "cpr_container_image": cpr_image,
        "cpr_model_artifact_uri": artifact_uri.rstrip("/"),
        "expected_sha256": expected_sha256,
        "feature_names": features,
        "display_name": display_name,
        "model_filename": model_filename,
        "reason_rule_version": reason_rule_version,
        "additivity_tolerance": additivity_tolerance,
        "machine_type": machine_type,
    }


def _default_authorized_session() -> Any:
    import google.auth  # type: ignore
    from google.auth.transport.requests import AuthorizedSession  # type: ignore

    credentials, _ = google.auth.default(scopes=[_AIP_CLOUD_PLATFORM_SCOPE])
    return AuthorizedSession(credentials)


def _poll_lro(
    session: Any,
    base_url: str,
    initial_op: Mapping[str, Any],
    sleep_fn: Callable[[float], None],
    poll_interval_seconds: float,
    max_polls: int,
) -> dict[str, Any]:
    op = dict(initial_op)
    polls = 0
    while not op.get("done", False):
        if polls >= max_polls:
            raise RuntimeError(f"Vertex AI operation timed out after {max_polls} polls: {op.get('name', '')}")
        sleep_fn(poll_interval_seconds)
        polls += 1
        lro_name = str(op["name"]).lstrip("/")
        resp = session.get(f"{base_url}/{lro_name}")
        resp.raise_for_status()
        op = dict(resp.json())

    if "error" in op and op["error"]:
        raise RuntimeError(f"Vertex AI operation failed: {op['error']}")
    return op


def _find_matching_deployed_model(
    session: Any,
    base_url: str,
    endpoint_path: str,
    display_name: str,
    expected_sha256: str,
) -> str | None:
    resp = session.get(f"{base_url}/{endpoint_path}")
    resp.raise_for_status()
    endpoint_doc = resp.json()
    for dm in endpoint_doc.get("deployedModels", []):
        if dm.get("displayName") != display_name:
            continue
        model_name = str(dm.get("model", "")).lstrip("/")
        if not model_name:
            continue
        model_resp = session.get(f"{base_url}/{model_name}")
        model_resp.raise_for_status()
        model_doc = model_resp.json()
        env_list = model_doc.get("containerSpec", {}).get("env", [])
        env_map = {str(item.get("name", "")): str(item.get("value", "")) for item in env_list if isinstance(item, Mapping)}
        if env_map.get("EXPECTED_MODEL_SHA256", "").strip().lower() == expected_sha256.lower():
            return model_name
    return None


def deploy_cpr_model(
    config: Mapping[str, Any],
    session: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 120,
) -> str:
    """Idempotently upload and deploy the CPR model via Vertex AI v1 REST API."""
    http = session if session is not None else _default_authorized_session()
    region = str(config["region"])
    project_id = str(config["project_id"])
    endpoint_id = str(config["endpoint_id"])
    display_name = str(config["display_name"])
    expected_sha256 = str(config["expected_sha256"]).lower()

    base_url = f"https://{region}-aiplatform.googleapis.com/v1"
    parent = f"projects/{project_id}/locations/{region}"
    endpoint_path = f"{parent}/endpoints/{endpoint_id}"

    existing_model = _find_matching_deployed_model(
        session=http,
        base_url=base_url,
        endpoint_path=endpoint_path,
        display_name=display_name,
        expected_sha256=expected_sha256,
    )
    if existing_model is not None:
        print(f"Idempotent no-op: {display_name} ({existing_model}) with SHA-256 {expected_sha256} already deployed.")
        return existing_model

    env_vars = {
        "MODEL_FILENAME": str(config["model_filename"]),
        "EXPECTED_MODEL_SHA256": expected_sha256,
        "FEATURE_NAMES": json.dumps(list(config["feature_names"])),
        "MODEL_ID": display_name,
        "CONTAINER_IMAGE_DIGEST": str(config["cpr_container_image"]),
        "REASON_RULE_VERSION": str(config["reason_rule_version"]),
        "SHAP_ADDITIVITY_TOLERANCE": str(config["additivity_tolerance"]),
    }
    upload_payload = {
        "model": {
            "displayName": display_name,
            "artifactUri": str(config["cpr_model_artifact_uri"]),
            "containerSpec": {
                "imageUri": str(config["cpr_container_image"]),
                "predictRoute": "/predict",
                "healthRoute": "/health",
                "env": [{"name": k, "value": v} for k, v in env_vars.items()],
            },
        }
    }
    upload_resp = http.post(f"{base_url}/{parent}/models:upload", json=upload_payload)
    upload_resp.raise_for_status()
    upload_op = _poll_lro(
        session=http,
        base_url=base_url,
        initial_op=upload_resp.json(),
        sleep_fn=sleep_fn,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
    )
    model_resource_name = str(upload_op["response"]["model"])

    deploy_payload = {
        "deployedModel": {
            "model": model_resource_name,
            "displayName": display_name,
            "dedicatedResources": {
                "machineSpec": {"machineType": str(config["machine_type"])},
                "minReplicaCount": 1,
                "maxReplicaCount": 3,
            },
        },
        "trafficSplit": {"0": 100},
    }
    deploy_resp = http.post(f"{base_url}/{endpoint_path}:deployModel", json=deploy_payload)
    deploy_resp.raise_for_status()
    _poll_lro(
        session=http,
        base_url=base_url,
        initial_op=deploy_resp.json(),
        sleep_fn=sleep_fn,
        poll_interval_seconds=poll_interval_seconds,
        max_polls=max_polls,
    )
    return model_resource_name


def main() -> int:
    cfg = parse_deploy_config()
    resource_name = deploy_cpr_model(cfg)
    print(f"Deployed CPR model {resource_name} to endpoint {cfg['endpoint_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
