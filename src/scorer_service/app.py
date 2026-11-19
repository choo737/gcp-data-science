"""Fail-closed Cloud Run v2 XAI scoring service and agent reason formatter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
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


def build_vertex_predict_url(config: ScoringConfig) -> str:
    """Resolve the Vertex AI :predict URL over PSC or fail closed in prod."""
    host = config.vertex_psc_endpoint_ip.strip()
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


def _default_vertex_predict(config: ScoringConfig, instance: Mapping[str, Any]) -> dict[str, Any]:
    url = build_vertex_predict_url(config)
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
    body = resp.json()
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
    ) -> None:
        self.config = config
        self.ledger_writer = ledger_writer
        self._vertex_predict_fn = vertex_predict_fn or _default_vertex_predict

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

        if path == "/health" and method == "GET":
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


def application(environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
    """Lazy Gunicorn WSGI entrypoint configured from Cloud Run environment variables."""
    if environ.get("PATH_INFO", "/") == "/health" and environ.get("REQUEST_METHOD", "GET").upper() == "GET":
        body = json.dumps({"status": "ok"}).encode("utf-8")
        start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        return [body]

    global _LAZY_APP
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
