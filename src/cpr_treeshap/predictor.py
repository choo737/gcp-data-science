"""Exact path-dependent C++ TreeSHAP predictor for Vertex AI CPR."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable, Mapping, Sequence

# Lock OpenMP thread count to 1 for deterministic TreeSHAP floating-point reduction.
os.environ["OMP_NUM_THREADS"] = "1"

RESTRICTED_PII_KEYS: frozenset[str] = frozenset({
    "customer_ic_number",
    "nric",
    "passport_number",
})

REASON_RULE_VERSION = "bnm-fair-lending-v1.0"
SCHEMA_VERSION = "v1"
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

DEFAULT_ADVERSE_REASON_CODES: dict[str, tuple[str, str]] = {
    "tenure_months": (
        "RC01",
        "Short customer relationship tenure relative to underwriting policy",
    ),
    "avg_monthly_spend": (
        "RC02",
        "Insufficient rolling 30-day net transaction volume",
    ),
    "rolling_order_count": (
        "RC03",
        "Low verified order frequency in trailing 30-day window",
    ),
    "days_since_last_order": (
        "RC04",
        "Extended account inactivity since last completed transaction",
    ),
    "cancellation_rate_30d": (
        "RC05",
        "Elevated 30-day order cancellation or reversal rate",
    ),
    "discount_utilization_ratio": (
        "RC06",
        "High promotional discount dependency relative to gross spend",
    ),
    "delinquency_count_12m": (
        "RC_DELINQ_12M",
        "Recent delinquency events in the past twelve months.",
    ),
    "credit_utilization_ratio": (
        "RC_HIGH_UTIL",
        "High revolving credit balance relative to total credit limit.",
    ),
    "debt_to_income_ratio": (
        "RC_HIGH_DTI",
        "Elevated debt-to-income obligation ratio.",
    ),
    "verified_income_band": (
        "RC_INCOME_BAND",
        "Verified income tier below threshold for requested credit.",
    ),
}


def _stable_sigmoid(margin: float) -> float:
    if margin >= 0.0:
        return 1.0 / (1.0 + math.exp(-margin))
    exp_x = math.exp(margin)
    return exp_x / (1.0 + exp_x)


def compute_sha256_file(path: Path) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_feature_names(raw_features: str) -> list[str]:
    """Parse FEATURE_NAMES from either a JSON list or comma-separated string."""
    stripped = raw_features.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("FEATURE_NAMES JSON must be a list of strings.")
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in stripped.split(",") if part.strip()]


def _parse_gcs_uri(gs_uri: str) -> tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    rest = gs_uri[len("gs://"):]
    if "/" not in rest:
        raise ValueError(f"GCS URI must include bucket and object path: {gs_uri}")
    bucket_name, blob_name = rest.split("/", 1)
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    return bucket_name, blob_name


def rank_adverse_features(
    shap_logit_map: Mapping[str, float],
    explained_class: str = "APPROVE",
    top_k: int = 4,
    reason_codes: Mapping[str, tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Rank top adverse features with deterministic alphabetical tie-breaking."""
    if top_k <= 0:
        return []
    codes = reason_codes if reason_codes is not None else DEFAULT_ADVERSE_REASON_CODES
    normalized_class = explained_class.strip().upper()

    if normalized_class == "APPROVE":
        candidates = [(feat, float(val)) for feat, val in shap_logit_map.items() if float(val) < 0.0]
        # Most negative log-odds reduces approval most; ties break alphabetically by feature name.
        candidates.sort(key=lambda kv: (kv[1], kv[0]))
    else:
        candidates = [(feat, float(val)) for feat, val in shap_logit_map.items() if float(val) > 0.0]
        # Most positive log-odds pushes toward rejection/default most; ties break alphabetically.
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))

    ranked: list[dict[str, Any]] = []
    for feat, val in candidates[:top_k]:
        code, description = codes.get(
            feat,
            (f"RC_{feat.upper()}", f"Adverse model factor: {feat}."),
        )
        ranked.append({
            "feature": feat,
            "reason_code": code,
            "reason_text": description,
            "shap_logit": float(val),
        })
    return ranked


class TreeShapCprPredictor:
    """Fail-closed XGBoost predictor enforcing exact TreeSHAP additivity."""

    def __init__(
        self,
        booster: Any,
        feature_names: Sequence[str],
        model_id: str,
        model_version_id: str,
        model_artifact_sha256: str,
        container_image_digest: str,
        additivity_tolerance: float = 1e-5,
        explained_class: str = "APPROVE",
        top_k_reasons: int = 4,
        reason_rule_version: str = REASON_RULE_VERSION,
        dmatrix_factory: Callable[[list[list[float]], Sequence[str]], Any] | None = None,
    ) -> None:
        os.environ["OMP_NUM_THREADS"] = "1"
        if not feature_names:
            raise ValueError("feature_names must contain at least one feature.")
        if additivity_tolerance <= 0.0:
            raise ValueError("additivity_tolerance must be positive.")
        if not model_artifact_sha256.strip():
            raise ValueError("model_artifact_sha256 must be non-empty.")

        self.booster = booster
        self.feature_names = list(feature_names)
        self.model_id = model_id
        self.model_version_id = model_version_id
        self.model_artifact_sha256 = model_artifact_sha256
        self.container_image_digest = container_image_digest
        self.additivity_tolerance = float(additivity_tolerance)
        self.explained_class = explained_class.strip().upper()
        self.top_k_reasons = int(top_k_reasons)
        self.reason_rule_version = reason_rule_version
        self._dmatrix_factory = dmatrix_factory

    @classmethod
    def load_from_file(
        cls,
        model_path: str | Path,
        expected_sha256: str,
        feature_names: Sequence[str],
        model_id: str,
        model_version_id: str,
        container_image_digest: str,
        additivity_tolerance: float = 1e-5,
        explained_class: str = "APPROVE",
        top_k_reasons: int = 4,
        reason_rule_version: str = REASON_RULE_VERSION,
        storage_client: Any | None = None,
    ) -> "TreeShapCprPredictor":
        """Verify SHA-256 manifest digest (for local or gs:// URIs) and load XGBoost Booster."""
        raw_path = str(model_path).strip()
        temp_path: Path | None = None
        try:
            if raw_path.startswith("gs://"):
                bucket_name, blob_name = _parse_gcs_uri(raw_path)
                if storage_client is None:
                    from google.cloud import storage  # type: ignore

                    storage_client = storage.Client()
                fd, tmp_name = tempfile.mkstemp(prefix="cpr-model-", suffix=".ubj")
                os.close(fd)
                temp_path = Path(tmp_name)
                os.chmod(temp_path, 0o600)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                blob.download_to_filename(str(temp_path))
                artifact = temp_path
            else:
                artifact = Path(model_path)
                if not artifact.is_file():
                    raise FileNotFoundError(f"Model artifact not found: {artifact}")

            actual_sha256 = compute_sha256_file(artifact)
            if not hmac.compare_digest(actual_sha256.lower(), expected_sha256.strip().lower()):
                raise ValueError(
                    f"Model artifact SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
                )
            import xgboost as xgb  # type: ignore

            booster = xgb.Booster()
            booster.load_model(str(artifact))
            return cls(
                booster=booster,
                feature_names=feature_names,
                model_id=model_id,
                model_version_id=model_version_id,
                model_artifact_sha256=actual_sha256,
                container_image_digest=container_image_digest,
                additivity_tolerance=additivity_tolerance,
                explained_class=explained_class,
                top_k_reasons=top_k_reasons,
                reason_rule_version=reason_rule_version,
            )
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def _build_dmatrix(self, row_values: list[float]) -> Any:
        if self._dmatrix_factory is not None:
            return self._dmatrix_factory([row_values], self.feature_names)
        import xgboost as xgb  # type: ignore

        return xgb.DMatrix([row_values], feature_names=self.feature_names)

    def _validate_and_extract_features(self, instance: Mapping[str, Any]) -> dict[str, float]:
        pii_hits = sorted(RESTRICTED_PII_KEYS.intersection(instance.keys()))
        if pii_hits:
            raise ValueError(f"Restricted PII keys rejected in feature payload: {', '.join(pii_hits)}")

        clean_features: dict[str, float] = {}
        for name in self.feature_names:
            if name not in instance:
                raise ValueError(f"Missing required feature: {name}")
            val = instance[name]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"Feature '{name}' must be numeric, got {type(val).__name__}")
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                raise ValueError(f"Feature '{name}' must be finite.")
            clean_features[name] = fval
        return clean_features

    def predict_instance(
        self,
        instance: Mapping[str, Any],
        entity_id: str | None = None,
    ) -> dict[str, Any]:
        """Score one instance and compute exact TreeSHAP log-odds with additivity enforcement."""
        clean_features = self._validate_and_extract_features(instance)
        row_values = [clean_features[name] for name in self.feature_names]
        dmatrix = self._build_dmatrix(row_values)

        raw_margin_arr = self.booster.predict(dmatrix, output_margin=True)
        contribs_arr = self.booster.predict(dmatrix, pred_contribs=True, approx_contribs=False)

        raw_margin = float(raw_margin_arr[0])
        contrib_row = [float(v) for v in contribs_arr[0]]
        if len(contrib_row) != len(self.feature_names) + 1:
            raise ValueError(
                f"Expected {len(self.feature_names) + 1} TreeSHAP contributions, got {len(contrib_row)}"
            )

        shap_values = contrib_row[:-1]
        base_logit = float(contrib_row[-1])
        shap_sum_logit = float(sum(shap_values))

        drift = abs((base_logit + shap_sum_logit) - raw_margin)
        allowed_drift = self.additivity_tolerance * max(1.0, abs(raw_margin))
        if drift > allowed_drift:
            raise ValueError(
                f"TreeSHAP additivity breach: |({base_logit} + {shap_sum_logit}) - {raw_margin}| = "
                f"{drift} > tolerance {allowed_drift}"
            )

        shap_logit_map = {
            name: float(val)
            for name, val in zip(self.feature_names, shap_values)
        }
        top_adverse = rank_adverse_features(
            shap_logit_map=shap_logit_map,
            explained_class=self.explained_class,
            top_k=self.top_k_reasons,
        )
        approve_prob = _stable_sigmoid(raw_margin)

        resolved_entity_id = entity_id if entity_id is not None else str(instance.get("entity_id", ""))
        return {
            "entity_id": resolved_entity_id,
            "model_id": self.model_id,
            "model_version_id": self.model_version_id,
            "model_artifact_sha256": self.model_artifact_sha256,
            "container_image_digest": self.container_image_digest,
            "schema_version": SCHEMA_VERSION,
            "explained_class": self.explained_class,
            "approve_prob": float(approve_prob),
            "raw_margin": float(raw_margin),
            "base_logit": float(base_logit),
            "shap_sum_logit": float(shap_sum_logit),
            "shap_logit_map": shap_logit_map,
            "top_adverse_reasons": top_adverse,
            "feature_payload": clean_features,
            "reason_rule_version": self.reason_rule_version,
        }

    def predict(self, request_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Handle Vertex AI :predict request envelope."""
        instances = request_payload.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError("Request body must contain a non-empty 'instances' list.")
        predictions = [self.predict_instance(inst) for inst in instances]
        return {"predictions": predictions}


def create_cpr_wsgi_app(predictor: TreeShapCprPredictor) -> Callable[..., list[bytes]]:
    """Create a PEP 3333 WSGI application for Vertex AI Custom Prediction Routine."""

    def application(environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET").upper()

        if path in ("/health", "/healthz") and method == "GET":
            body = json.dumps({"status": "ok", "model_loaded": True, "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}).encode("utf-8")
            start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            return [body]

        if path == "/predict" and method == "POST":
            try:
                length = int(environ.get("CONTENT_LENGTH") or 0)
                raw_body = environ["wsgi.input"].read(length) if length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
                result = predictor.predict(payload)
                body = json.dumps(result).encode("utf-8")
                start_response("200 OK", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
                return [body]
            except ValueError as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                start_response("400 Bad Request", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
                return [body]

        body = json.dumps({"error": "Not found"}).encode("utf-8")
        start_response("404 Not Found", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
        return [body]

    return application


def resolve_model_source_from_env() -> str:
    """Resolve model artifact URI from AIP_STORAGE_URI or MODEL_PATH."""
    aip_storage_uri = os.environ.get("AIP_STORAGE_URI", "").strip()
    model_filename = os.environ.get("MODEL_FILENAME", "model.ubj").strip() or "model.ubj"
    if aip_storage_uri:
        return f"{aip_storage_uri.rstrip('/')}/{model_filename}"
    return os.environ.get("MODEL_PATH", "").strip()


def _validate_cpr_env() -> tuple[str, str, list[str]]:
    model_source = resolve_model_source_from_env()
    expected_sha = os.environ.get("EXPECTED_MODEL_SHA256", "").strip()
    feature_names = parse_feature_names(os.environ.get("FEATURE_NAMES", ""))
    if not model_source or not expected_sha or not feature_names:
        raise RuntimeError("AIP_STORAGE_URI or MODEL_PATH, EXPECTED_MODEL_SHA256, and FEATURE_NAMES are required.")
    if not _SHA256_HEX_RE.fullmatch(expected_sha):
        raise ValueError("EXPECTED_MODEL_SHA256 must be a 64-character hex digest.")
    return model_source, expected_sha.lower(), feature_names


_LAZY_CPR_APP: Callable[..., list[bytes]] | None = None
_LAZY_CPR_LOCK = threading.Lock()
_BG_LOAD_THREAD: threading.Thread | None = None
_BG_LOAD_ERROR: Exception | None = None


def reset_lazy_state_for_tests() -> None:
    """Reset module-level lazy CPR WSGI and background loader state for unit tests."""
    global _LAZY_CPR_APP, _BG_LOAD_THREAD, _BG_LOAD_ERROR
    with _LAZY_CPR_LOCK:
        _LAZY_CPR_APP = None
        _BG_LOAD_THREAD = None
        _BG_LOAD_ERROR = None


def _load_cpr_app_locked() -> Callable[..., list[bytes]]:
    global _LAZY_CPR_APP
    if _LAZY_CPR_APP is not None:
        return _LAZY_CPR_APP
    if _BG_LOAD_ERROR is not None:
        raise _BG_LOAD_ERROR
    model_source, expected_sha, feature_names = _validate_cpr_env()
    predictor = TreeShapCprPredictor.load_from_file(
        model_path=model_source,
        expected_sha256=expected_sha,
        feature_names=feature_names,
        model_id=os.environ.get("MODEL_ID", "credit_xgb_cpr"),
        model_version_id=os.environ.get("MODEL_VERSION_ID", "1"),
        container_image_digest=os.environ.get("CONTAINER_IMAGE_DIGEST", "unpinned"),
        additivity_tolerance=float(os.environ.get("SHAP_ADDITIVITY_TOLERANCE", "0.00001")),
        reason_rule_version=os.environ.get("REASON_RULE_VERSION", REASON_RULE_VERSION),
    )
    _LAZY_CPR_APP = create_cpr_wsgi_app(predictor)
    return _LAZY_CPR_APP


def _ensure_cpr_app_loaded() -> Callable[..., list[bytes]]:
    if _LAZY_CPR_APP is None:
        with _LAZY_CPR_LOCK:
            return _load_cpr_app_locked()
    return _LAZY_CPR_APP


def _background_load_worker() -> None:
    global _BG_LOAD_ERROR
    try:
        with _LAZY_CPR_LOCK:
            _load_cpr_app_locked()
    except Exception as exc:  # noqa: BLE001
        with _LAZY_CPR_LOCK:
            _BG_LOAD_ERROR = exc


def _start_background_load_if_needed() -> None:
    global _BG_LOAD_THREAD
    with _LAZY_CPR_LOCK:
        if _LAZY_CPR_APP is not None or _BG_LOAD_ERROR is not None:
            return
        if _BG_LOAD_THREAD is None or not _BG_LOAD_THREAD.is_alive():
            _BG_LOAD_THREAD = threading.Thread(
                target=_background_load_worker,
                name="cpr-bg-model-loader",
                daemon=True,
            )
            _BG_LOAD_THREAD.start()


def application(environ: dict[str, Any], start_response: Callable[..., Any]) -> list[bytes]:
    """Thread-safe lazy Gunicorn WSGI entrypoint with readiness-gated GCS health checks."""
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()

    if path in ("/health", "/healthz") and method == "GET":
        if _LAZY_CPR_APP is not None:
            return _LAZY_CPR_APP(environ, start_response)
        if _BG_LOAD_ERROR is not None:
            body = json.dumps({"status": "error", "error": str(_BG_LOAD_ERROR)}).encode("utf-8")
            start_response("500 Internal Server Error", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            return [body]

        try:
            model_source, _, _ = _validate_cpr_env()
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"status": "error", "error": str(exc)}).encode("utf-8")
            start_response("500 Internal Server Error", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            return [body]

        if model_source.startswith("gs://"):
            _start_background_load_if_needed()
            if _LAZY_CPR_APP is not None:
                return _LAZY_CPR_APP(environ, start_response)
            if _BG_LOAD_ERROR is not None:
                body = json.dumps({"status": "error", "error": str(_BG_LOAD_ERROR)}).encode("utf-8")
                start_response("500 Internal Server Error", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
                return [body]
            body = json.dumps({
                "status": "loading",
                "model_loaded": False,
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            }).encode("utf-8")
            start_response("503 Service Unavailable", [("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
            return [body]

    app_fn = _ensure_cpr_app_loaded()
    return app_fn(environ, start_response)
