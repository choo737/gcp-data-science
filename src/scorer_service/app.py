"""Fail-closed Cloud Run v2 XAI scoring service and agent reason formatter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Any, Callable, Mapping

try:
    from .ledger_writer import (
        RESTRICTED_PII_KEYS,
        FailClosedLedgerWriter,
        LedgerCommitError,
    )
except ImportError:
    try:
        from scorer_service.ledger_writer import (
            RESTRICTED_PII_KEYS,
            FailClosedLedgerWriter,
            LedgerCommitError,
        )
    except ImportError:
        from ledger_writer import (
            RESTRICTED_PII_KEYS,
            FailClosedLedgerWriter,
            LedgerCommitError,
        )

_PSC_CACHE_TTL_SECONDS = 300.0
_PSC_DISCOVERY_LOCK = threading.Lock()
_PSC_CACHE: dict[tuple[str, str, str], tuple[str, float]] = {}


class PscEndpointRequiredError(RuntimeError):
    """Raised when private Vertex AI PSC endpoint routing is required but unconfigured."""


@dataclass(frozen=True)
class ScoringConfig:
    ml_ops_project_id: str
    region: str
    vertex_endpoint_id: str
    ml_audit_dataset_id: str
    ml_audit_table_id: str = "ml_xai_audit_ledger"
    ml_worm_bucket_name: str = ""
    shap_additivity_tolerance: float = 1e-5
    serving_track: str = "vertex_cpr_online"
    vertex_psc_endpoint_ip: str = ""
    environment: str = "dev"
    require_psc_endpoint: bool = False

    @classmethod
    def from_env(cls) -> "ScoringConfig":
        project_id = os.environ.get("ML_OPS_PROJECT_ID", "").strip()
        region = os.environ.get("REGION", "asia-southeast1").strip()
        endpoint_id = os.environ.get("VERTEX_ENDPOINT_ID", "").strip()
        audit_dataset = os.environ.get("ML_AUDIT_DATASET_ID", "").strip()
        audit_table = os.environ.get("ML_AUDIT_TABLE_ID", "ml_xai_audit_ledger").strip()
        worm_bucket = os.environ.get("ML_WORM_BUCKET_NAME", "").strip()
        tol = float(os.environ.get("SHAP_ADDITIVITY_TOLERANCE", "0.00001"))
        track = os.environ.get("SERVING_TRACK", "vertex_cpr_online").strip()
        psc_ip = os.environ.get("VERTEX_PSC_ENDPOINT_IP", "").strip()
        environment = os.environ.get("ENVIRONMENT", "dev").strip().lower()
        require_psc = os.environ.get("REQUIRE_PSC_ENDPOINT", "").strip().lower() in {"1", "true", "yes"}
        if not project_id or not endpoint_id or not audit_dataset:
            raise ValueError("ML_OPS_PROJECT_ID, VERTEX_ENDPOINT_ID, and ML_AUDIT_DATASET_ID are required.")
        return cls(
            ml_ops_project_id=project_id,
            region=region,
            vertex_endpoint_id=endpoint_id,
            ml_audit_dataset_id=audit_dataset,
            ml_audit_table_id=audit_table,
            ml_worm_bucket_name=worm_bucket,
            shap_additivity_tolerance=tol,
            serving_track=track,
            vertex_psc_endpoint_ip=psc_ip,
            environment=environment,
            require_psc_endpoint=require_psc,
        )


def _psc_cache_key(config: ScoringConfig) -> tuple[str, str, str]:
    return (config.ml_ops_project_id, config.region, config.vertex_endpoint_id)


def invalidate_psc_cache(config: ScoringConfig | None = None) -> None:
    """Clear cached PSC endpoint IP(s) after connection failures or during tests."""
    with _PSC_DISCOVERY_LOCK:
        if config is None:
            _PSC_CACHE.clear()
        else:
            _PSC_CACHE.pop(_psc_cache_key(config), None)


def extract_psc_ip_from_endpoint(endpoint_doc: Mapping[str, Any]) -> str:
    """Extract live PSC endpoint IP from Vertex AI Endpoint metadata."""
    psc_cfg = endpoint_doc.get("privateServiceConnectConfig", {})
    if not isinstance(psc_cfg, Mapping):
        return ""
    configs = psc_cfg.get("pscAutomationConfigs", [])
    if not isinstance(configs, list):
        return ""

    first_non_empty = ""
    for item in configs:
        if not isinstance(item, Mapping):
            continue
        ip = str(item.get("ipAddress", "")).strip()
        if not ip:
            continue
        if str(item.get("state", "")).strip() == "PSC_AUTOMATION_STATE_SUCCESSFUL":
            return ip
        if not first_non_empty:
            first_non_empty = ip
    return first_non_empty


def _default_endpoint_metadata_fetcher(config: ScoringConfig) -> Mapping[str, Any]:
    import google.auth  # type: ignore
    from google.auth.transport.requests import AuthorizedSession  # type: ignore

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    session = AuthorizedSession(credentials)
    url = (
        f"https://{config.region}-aiplatform.googleapis.com/v1/"
        f"projects/{config.ml_ops_project_id}/locations/{config.region}/"
        f"endpoints/{config.vertex_endpoint_id}"
    )
    resp = session.get(url, timeout=5.0)
    resp.raise_for_status()
    return dict(resp.json())


def resolve_psc_endpoint_ip(
    config: ScoringConfig,
    endpoint_metadata_fetcher: Callable[[ScoringConfig], Mapping[str, Any]] | None = None,
    force_refresh: bool = False,
    ttl_seconds: float = _PSC_CACHE_TTL_SECONDS,
) -> str:
    """Resolve PSC endpoint IP from env, TTL cache, or Vertex control-plane metadata."""
    env_ip = config.vertex_psc_endpoint_ip.strip()
    if not force_refresh and env_ip:
        return env_ip

    key = _psc_cache_key(config)
    with _PSC_DISCOVERY_LOCK:
        if not force_refresh:
            if env_ip:
                return env_ip
            cached = _PSC_CACHE.get(key)
            if cached is not None:
                cached_ip, cached_at = cached
                if cached_ip and (time.monotonic() - cached_at) < ttl_seconds:
                    return cached_ip

        fetcher = endpoint_metadata_fetcher or _default_endpoint_metadata_fetcher
        endpoint_doc = fetcher(config)
        discovered_ip = extract_psc_ip_from_endpoint(endpoint_doc)
        if discovered_ip:
            _PSC_CACHE[key] = (discovered_ip, time.monotonic())
            return discovered_ip
        _PSC_CACHE.pop(key, None)
        return ""


def build_vertex_predict_url(config: ScoringConfig, psc_ip: str | None = None) -> str:
    """Resolve the Vertex AI :predict URL over PSC or fail closed in prod."""
    host = (config.vertex_psc_endpoint_ip if psc_ip is None else psc_ip).strip()
    if host:
        scheme = "https" if (host.endswith(".prediction.p.vertexai.goog") or host.endswith(":443")) else "http"
        return (
            f"{scheme}://{host}/v1/"
            f"projects/{config.ml_ops_project_id}/locations/{config.region}/"
            f"endpoints/{config.vertex_endpoint_id}:predict"
        )
    if config.environment == "prod" or config.require_psc_endpoint:
        raise PscEndpointRequiredError(
            "VERTEX_PSC_ENDPOINT_IP is required in prod/PSC-enforced mode; refusing public endpoint call."
        )
    return (
        f"https://{config.region}-aiplatform.googleapis.com/v1/"
        f"projects/{config.ml_ops_project_id}/locations/{config.region}/"
        f"endpoints/{config.vertex_endpoint_id}:predict"
    )


def format_adverse_reasons_for_agent(ledger_row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract deterministic reason codes only so agents cannot invent credit explanations."""
    raw_reasons = ledger_row.get("top_adverse_reasons", [])
    if isinstance(raw_reasons, str):
        parsed_reasons = json.loads(raw_reasons)
    elif isinstance(raw_reasons, list):
        parsed_reasons = raw_reasons
    else:
        raise ValueError("top_adverse_reasons must be a JSON string or list.")

    deterministic_reasons: list[dict[str, str]] = []
    for item in parsed_reasons:
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("reason_code", "")).strip()
        text = str(item.get("reason_text", "")).strip()
        if code and text:
            deterministic_reasons.append({
                "reason_code": code,
                "reason_text": text,
            })

    return {
        "decision_id": str(ledger_row["decision_id"]),
        "entity_id": str(ledger_row["entity_id"]),
        "explained_class": str(ledger_row["explained_class"]),
        "reason_rule_version": str(ledger_row["reason_rule_version"]),
        "adverse_reasons": deterministic_reasons,
    }


def _default_http_post(url: str, instance: Mapping[str, Any]) -> dict[str, Any]:
    import google.auth  # type: ignore
    from google.auth.transport.requests import Request  # type: ignore
    import requests  # type: ignore

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json={"instances": [dict(instance)]},
        timeout=10,
    )
    resp.raise_for_status()
    return dict(resp.json())


def _is_connection_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    try:
        import requests  # type: ignore

        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
    except ImportError:
        pass
    return False


def _default_vertex_predict(
    config: ScoringConfig,
    instance: Mapping[str, Any],
    endpoint_metadata_fetcher: Callable[[ScoringConfig], Mapping[str, Any]] | None = None,
    http_post_fn: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    require_psc = config.environment == "prod" or config.require_psc_endpoint
    try:
        resolved_ip = resolve_psc_endpoint_ip(config, endpoint_metadata_fetcher=endpoint_metadata_fetcher)
    except Exception as exc:  # noqa: BLE001
        if require_psc:
            raise PscEndpointRequiredError(f"Failed to discover Vertex PSC endpoint IP: {exc}") from exc
        resolved_ip = ""

    url = build_vertex_predict_url(config, psc_ip=resolved_ip)
    post_fn = http_post_fn or _default_http_post

    try:
        body = post_fn(url, instance)
    except Exception as exc:
        if not resolved_ip or not _is_connection_error(exc):
            raise
        invalidate_psc_cache(config)
        try:
            refreshed_ip = resolve_psc_endpoint_ip(
                config,
                endpoint_metadata_fetcher=endpoint_metadata_fetcher,
                force_refresh=True,
            )
        except Exception as refresh_exc:  # noqa: BLE001
            if require_psc:
                raise PscEndpointRequiredError(
                    f"PSC endpoint connection failed and refresh errored: {refresh_exc}"
                ) from exc
            raise exc
        if not refreshed_ip:
            if require_psc:
                raise PscEndpointRequiredError(
                    "Cached PSC endpoint connection failed and control-plane returned no live IP."
                ) from exc
            raise exc
        refreshed_url = build_vertex_predict_url(config, psc_ip=refreshed_ip)
        body = post_fn(refreshed_url, instance)

    predictions = body.get("predictions", [])
    if not predictions:
        raise RuntimeError("Vertex AI :predict returned empty predictions list.")
    return dict(predictions[0])


class ScoringService:
    """Coordinates Vertex :predict inference, TreeSHAP tolerance, and fail-closed ledger commits."""

    def __init__(
        self,
        config: ScoringConfig,
        ledger_writer: FailClosedLedgerWriter,
        vertex_predict_fn: Callable[[ScoringConfig, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        endpoint_metadata_fetcher: Callable[[ScoringConfig], Mapping[str, Any]] | None = None,
        http_post_fn: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.ledger_writer = ledger_writer
        self._endpoint_metadata_fetcher = endpoint_metadata_fetcher
        self._http_post_fn = http_post_fn
        if vertex_predict_fn is not None:
            self._vertex_predict_fn = vertex_predict_fn
        else:
            self._vertex_predict_fn = lambda c, inst: _default_vertex_predict(
                c,
                inst,
                endpoint_metadata_fetcher=self._endpoint_metadata_fetcher,
                http_post_fn=self._http_post_fn,
            )

    def score_applicant(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """Score one applicant and release decision only after synchronous ledger commit."""
        entity_id = str(payload.get("entity_id", "")).strip()
        features = payload.get("features")
        if not entity_id or not isinstance(features, Mapping) or not features:
            return 400, {"error": "Payload must include non-empty 'entity_id' and 'features' object."}

        pii_hits = sorted(RESTRICTED_PII_KEYS.intersection(features.keys()))
        if pii_hits:
            return 400, {"error": f"Restricted PII keys rejected: {', '.join(pii_hits)}"}

        try:
            prediction = self._vertex_predict_fn(self.config, features)
        except PscEndpointRequiredError as exc:
            return 503, {"error": str(exc), "decision_released": False}
        except Exception as exc:  # noqa: BLE001
            return 502, {"error": f"Vertex AI :predict failure: {exc}", "decision_released": False}

        try:
            raw_margin = float(prediction["raw_margin"])
            base_logit = float(prediction["base_logit"])
            shap_sum_logit = float(prediction["shap_sum_logit"])
        except (KeyError, TypeError, ValueError) as exc:
            return 502, {"error": f"Malformed TreeSHAP prediction payload: {exc}", "decision_released": False}

        drift = abs((base_logit + shap_sum_logit) - raw_margin)
        allowed = self.config.shap_additivity_tolerance * max(1.0, abs(raw_margin))
        if drift > allowed:
            return 503, {
                "error": f"TreeSHAP additivity drift {drift} exceeded {allowed}",
                "decision_released": False,
            }

        try:
            row = self.ledger_writer.build_ledger_row(
                entity_id=entity_id,
                serving_track="vertex_cpr_online",
                prediction=prediction,
            )
            decision_id = self.ledger_writer.commit_row_sync(row)
        except (LedgerCommitError, ValueError) as exc:
            return 503, {
                "error": f"Audit ledger commit failed: {exc}",
                "decision_released": False,
            }

        agent_summary = format_adverse_reasons_for_agent(row)
        return 200, {
            "decision_id": decision_id,
            "entity_id": row["entity_id"],
            "approve_prob": row["approve_prob"],
            "explained_class": row["explained_class"],
            "top_adverse_reasons": agent_summary["adverse_reasons"],
            "reason_rule_version": row["reason_rule_version"],
            "ledger_committed": True,
        }


_HTTP_STATUS_TEXT = {
    200: "200 OK",
    400: "400 Bad Request",
    404: "404 Not Found",
    502: "502 Bad Gateway",
    503: "503 Service Unavailable",
}


def create_scorer_wsgi_app(service: ScoringService) -> Callable[..., list[bytes]]:
    """Create a PEP 3333 WSGI application for Cloud Run v2."""

    def application(environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()

        if path in ("/health", "/healthz") and method == "GET":
            body = json.dumps({"status": "ok"}).encode("utf-8")
            start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            return [body]

        if path == "/score" and method == "POST":
            try:
                length = int(environ.get("CONTENT_LENGTH") or 0)
                raw_body = environ["wsgi.input"].read(length) if length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
            except (ValueError, json.JSONDecodeError) as exc:
                body = json.dumps({"error": f"Invalid JSON request body: {exc}"}).encode("utf-8")
                start_response("400 Bad Request", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
                return [body]

            status_code, response_obj = service.score_applicant(payload)
            body = json.dumps(response_obj).encode("utf-8")
            start_response(
                _HTTP_STATUS_TEXT.get(status_code, f"{status_code} Error"),
                [("Content-Type", "application/json"), ("Content-Length", str(len(body)))],
            )
            return [body]

        body = json.dumps({"error": "Not found"}).encode("utf-8")
        start_response("404 Not Found", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        return [body]

    return application


_LAZY_APP: Callable[..., list[bytes]] | None = None
_LAZY_APP_LOCK = threading.Lock()


def application(environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
    """Thread-safe lazy Gunicorn WSGI entrypoint configured from Cloud Run env vars."""
    if environ.get("PATH_INFO", "/") in ("/health", "/healthz") and environ.get("REQUEST_METHOD", "GET").upper() == "GET":
        body = json.dumps({"status": "ok"}).encode("utf-8")
        start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        return [body]

    global _LAZY_APP
    if _LAZY_APP is None:
        with _LAZY_APP_LOCK:
            if _LAZY_APP is None:
                cfg = ScoringConfig.from_env()
                writer = FailClosedLedgerWriter(
                    project_id=cfg.ml_ops_project_id,
                    dataset_id=cfg.ml_audit_dataset_id,
                    table_id=cfg.ml_audit_table_id,
                    worm_bucket_name=cfg.ml_worm_bucket_name,
                )
                _LAZY_APP = create_scorer_wsgi_app(ScoringService(config=cfg, ledger_writer=writer))
    return _LAZY_APP(environ, start_response)
