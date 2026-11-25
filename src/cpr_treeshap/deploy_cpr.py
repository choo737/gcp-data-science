"""Idempotent Agent Platform (formerly Vertex AI) REST v1 uploader and deployer for CPR TreeSHAP models."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from typing import Any, Callable, Mapping, Sequence

_AIP_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _normalize_artifact_uri_and_filename(artifact_uri: str, model_filename: str) -> tuple[str, str]:
    """Normalize a GCS directory or direct .ubj file URI into (directory_uri, filename)."""
    cleaned_uri = artifact_uri.strip().rstrip("/")
    cleaned_name = model_filename.strip() or "model.ubj"
    if not cleaned_uri.startswith("gs://"):
        raise ValueError(f"CPR_MODEL_ARTIFACT_URI must be a gs:// URI, got: {artifact_uri!r}")
    if cleaned_uri.endswith(".ubj"):
        rest = cleaned_uri[len("gs://"):]
        if "/" not in rest:
            raise ValueError(f"CPR_MODEL_ARTIFACT_URI must include bucket and object path: {artifact_uri!r}")
        parent_uri, file_part = cleaned_uri.rsplit("/", 1)
        return parent_uri, file_part
    return cleaned_uri, cleaned_name


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
    raw_model_filename = env.get("MODEL_FILENAME", "model.ubj").strip() or "model.ubj"
    reason_rule_version = env.get("REASON_RULE_VERSION", "bnm-fair-lending-v1.0").strip()
    additivity_tolerance = env.get("SHAP_ADDITIVITY_TOLERANCE", "0.00001").strip()
    machine_type = env.get("DEPLOY_MACHINE_TYPE", "n2-standard-4").strip()
    kms_key_name = (env.get("KMS_KEY_ID", "") or env.get("KMS_KEY_NAME", "")).strip()

    if not project_id or not endpoint_id or not cpr_image:
        raise ValueError("ML_OPS_PROJECT_ID, VERTEX_ENDPOINT_ID, and CPR_CONTAINER_IMAGE are required.")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("EXPECTED_MODEL_SHA256 must be a 64-character hex SHA-256 digest.")
    if not raw_features:
        raise ValueError("FEATURE_NAMES must be non-empty.")

    normalized_uri, model_filename = _normalize_artifact_uri_and_filename(artifact_uri, raw_model_filename)
    model_version_id = (
        env.get("MODEL_VERSION_ID", "").strip()
        or env.get("GITHUB_SHA", "").strip()
        or expected_sha256[:12]
    )

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
        "cpr_model_artifact_uri": normalized_uri,
        "expected_sha256": expected_sha256,
        "feature_names": features,
        "display_name": display_name,
        "model_filename": model_filename,
        "model_version_id": model_version_id,
        "kms_key_name": kms_key_name,
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
            raise RuntimeError(f"Agent Platform operation timed out after {max_polls} polls: {op.get('name', '')}")
        sleep_fn(poll_interval_seconds)
        polls += 1
        lro_name = str(op["name"]).lstrip("/")
        resp = session.get(f"{base_url}/{lro_name}")
        resp.raise_for_status()
        op = dict(resp.json())

    if "error" in op and op["error"]:
        raise RuntimeError(f"Agent Platform operation failed: {op['error']}")
    return op


def _feature_names_match(raw_val: str | None, expected: Sequence[str]) -> bool:
    if raw_val is None:
        return False
    stripped = raw_val.strip()
    if not stripped:
        return False
    if stripped.startswith("["):
        try:
            parsed_raw = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed_raw, list):
            return False
        parsed = [str(item).strip() for item in parsed_raw if str(item).strip()]
    else:
        parsed = [part.strip() for part in stripped.split(",") if part.strip()]
    return parsed == [str(item).strip() for item in expected]


def _tolerance_matches(raw_val: str | None, expected: str) -> bool:
    if raw_val is None or not raw_val.strip():
        return False
    try:
        return math.isclose(float(raw_val.strip()), float(expected.strip()), rel_tol=1e-12, abs_tol=0.0)
    except ValueError:
        return False


def _inspect_endpoint_deployments(
    session: Any,
    base_url: str,
    endpoint_path: str,
    display_name: str,
    expected_sha256: str,
    expected_image_uri: str,
    expected_feature_names: Sequence[str],
    expected_additivity_tolerance: str,
    expected_reason_rule_version: str,
) -> tuple[str | None, list[str]]:
    resp = session.get(f"{base_url}/{endpoint_path}")
    resp.raise_for_status()
    endpoint_doc = resp.json()
    deployed_models = endpoint_doc.get("deployedModels", [])
    all_deployed_ids = [
        str(dm["id"])
        for dm in deployed_models
        if isinstance(dm, Mapping) and dm.get("id") is not None and str(dm["id"]).strip()
    ]

    for dm in deployed_models:
        if not isinstance(dm, Mapping) or dm.get("displayName") != display_name:
            continue
        model_name = str(dm.get("model", "")).lstrip("/")
        if not model_name:
            continue
        model_resp = session.get(f"{base_url}/{model_name}")
        model_resp.raise_for_status()
        model_doc = model_resp.json()
        container_spec = model_doc.get("containerSpec", {})
        if not isinstance(container_spec, Mapping):
            continue
        image_uri = str(container_spec.get("imageUri", "")).strip()
        if image_uri != expected_image_uri.strip():
            continue
        env_list = container_spec.get("env", [])
        env_map = {str(item.get("name", "")): str(item.get("value", "")) for item in env_list if isinstance(item, Mapping)}
        sha_ok = env_map.get("EXPECTED_MODEL_SHA256", "").strip().lower() == expected_sha256.lower()
        features_ok = _feature_names_match(env_map.get("FEATURE_NAMES"), expected_feature_names)
        tol_ok = _tolerance_matches(env_map.get("SHAP_ADDITIVITY_TOLERANCE"), expected_additivity_tolerance)
        rule_ok = env_map.get("REASON_RULE_VERSION", "").strip() == expected_reason_rule_version.strip()
        if sha_ok and features_ok and tol_ok and rule_ok:
            matched_id = str(dm.get("id", "")).strip()
            stale_ids = [dm_id for dm_id in all_deployed_ids if dm_id != matched_id]
            return model_name, stale_ids
    return None, all_deployed_ids


def _undeploy_previous_models(
    session: Any,
    base_url: str,
    endpoint_path: str,
    deployed_model_ids: Sequence[str],
    sleep_fn: Callable[[float], None],
    poll_interval_seconds: float,
    max_polls: int,
) -> None:
    for old_id in deployed_model_ids:
        resp = session.post(
            f"{base_url}/{endpoint_path}:undeployModel",
            json={"deployedModelId": old_id},
        )
        resp.raise_for_status()
        _poll_lro(
            session=session,
            base_url=base_url,
            initial_op=resp.json(),
            sleep_fn=sleep_fn,
            poll_interval_seconds=poll_interval_seconds,
            max_polls=max_polls,
        )


def deploy_cpr_model(
    config: Mapping[str, Any],
    session: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = 10.0,
    max_polls: int = 120,
) -> str:
    """Idempotently upload and deploy the CPR model via Agent Platform v1 REST API."""
    http = session if session is not None else _default_authorized_session()
    region = str(config["region"])
    project_id = str(config["project_id"])
    endpoint_id = str(config["endpoint_id"])
    display_name = str(config["display_name"])
    expected_sha256 = str(config["expected_sha256"]).lower()
    model_version_id = str(config.get("model_version_id") or expected_sha256[:12]).strip()
    kms_key_name = str(config.get("kms_key_name", "")).strip()
    artifact_uri, model_filename = _normalize_artifact_uri_and_filename(
        str(config["cpr_model_artifact_uri"]),
        str(config["model_filename"]),
    )

    base_url = f"https://{region}-aiplatform.googleapis.com/v1"
    parent = f"projects/{project_id}/locations/{region}"
    endpoint_path = f"{parent}/endpoints/{endpoint_id}"

    existing_model, previous_deployed_ids = _inspect_endpoint_deployments(
        session=http,
        base_url=base_url,
        endpoint_path=endpoint_path,
        display_name=display_name,
        expected_sha256=expected_sha256,
        expected_image_uri=str(config["cpr_container_image"]),
        expected_feature_names=list(config["feature_names"]),
        expected_additivity_tolerance=str(config["additivity_tolerance"]),
        expected_reason_rule_version=str(config["reason_rule_version"]),
    )
    if existing_model is not None:
        if previous_deployed_ids:
            _undeploy_previous_models(
                session=http,
                base_url=base_url,
                endpoint_path=endpoint_path,
                deployed_model_ids=previous_deployed_ids,
                sleep_fn=sleep_fn,
                poll_interval_seconds=poll_interval_seconds,
                max_polls=max_polls,
            )
        print(f"Idempotent no-op: {display_name} ({existing_model}) with SHA-256 {expected_sha256} already deployed.")
        return existing_model

    env_vars = {
        "MODEL_FILENAME": model_filename,
        "EXPECTED_MODEL_SHA256": expected_sha256,
        "FEATURE_NAMES": json.dumps(list(config["feature_names"])),
        "MODEL_ID": display_name,
        "MODEL_VERSION_ID": model_version_id,
        "CONTAINER_IMAGE_DIGEST": str(config["cpr_container_image"]),
        "REASON_RULE_VERSION": str(config["reason_rule_version"]),
        "SHAP_ADDITIVITY_TOLERANCE": str(config["additivity_tolerance"]),
    }
    model_spec: dict[str, Any] = {
        "displayName": display_name,
        "artifactUri": artifact_uri,
        "containerSpec": {
            "imageUri": str(config["cpr_container_image"]),
            "predictRoute": "/predict",
            "healthRoute": "/health",
            "env": [{"name": k, "value": v} for k, v in env_vars.items()],
        },
    }
    if kms_key_name:
        model_spec["encryptionSpec"] = {"kmsKeyName": kms_key_name}

    upload_payload = {"model": model_spec}
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

    if previous_deployed_ids:
        _undeploy_previous_models(
            session=http,
            base_url=base_url,
            endpoint_path=endpoint_path,
            deployed_model_ids=previous_deployed_ids,
            sleep_fn=sleep_fn,
            poll_interval_seconds=poll_interval_seconds,
            max_polls=max_polls,
        )

    return model_resource_name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload and deploy CPR TreeSHAP model to Agent Platform Endpoint.")
    parser.add_argument("--kms-key-name", default=None, help="Optional KMS CryptoKey resource name for model CMEK.")
    parser.add_argument("--model-version-id", default=None, help="Explicit MODEL_VERSION_ID injected into container env.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = parse_deploy_config()
    if args.kms_key_name is not None and args.kms_key_name.strip():
        cfg["kms_key_name"] = args.kms_key_name.strip()
    if args.model_version_id is not None and args.model_version_id.strip():
        cfg["model_version_id"] = args.model_version_id.strip()

    resource_name = deploy_cpr_model(cfg)
    print(f"Deployed CPR model {resource_name} to endpoint {cfg['endpoint_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
