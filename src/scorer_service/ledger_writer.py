"""Synchronous BigQuery Storage Write API COMMITTED-stream ledger writer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Mapping
import uuid

LOCKED_LEDGER_COLUMNS: tuple[str, ...] = (
    "decision_id",
    "entity_id",
    "decision_timestamp",
    "serving_track",
    "model_id",
    "model_version_id",
    "model_artifact_sha256",
    "container_image_digest",
    "schema_version",
    "explained_class",
    "approve_prob",
    "raw_margin",
    "base_logit",
    "shap_sum_logit",
    "shap_logit_map",
    "top_adverse_reasons",
    "feature_payload",
    "reason_rule_version",
)

RESTRICTED_PII_KEYS: frozenset[str] = frozenset({
    "customer_ic_number",
    "nric",
    "passport_number",
})


class LedgerCommitError(RuntimeError):
    """Raised when synchronous XAI audit ledger persistence fails."""


def _serialize_json_field(value: Any) -> str:
    if isinstance(value, str):
        # Validate JSON string is well-formed before writing to a BigQuery JSON column.
        json.loads(value)
        return value
    return json.dumps(value, sort_keys=True)


class FailClosedLedgerWriter:
    """Commits 18-column XAI ledger rows synchronously before decision release."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str = "ml_xai_audit_ledger",
        write_client: Any | None = None,
        storage_client: Any | None = None,
        worm_bucket_name: str = "",
    ) -> None:
        if not project_id.strip() or not dataset_id.strip() or not table_id.strip():
            raise ValueError("project_id, dataset_id, and table_id must be non-empty.")
        self.project_id = project_id.strip()
        self.dataset_id = dataset_id.strip()
        self.table_id = table_id.strip()
        self.worm_bucket_name = worm_bucket_name.strip()
        self._write_client = write_client
        self._storage_client = storage_client

    @property
    def table_parent(self) -> str:
        return f"projects/{self.project_id}/datasets/{self.dataset_id}/tables/{self.table_id}"

    def _get_write_client(self) -> Any:
        if self._write_client is None:
            from google.cloud.bigquery_storage_v1 import BigQueryWriteClient  # type: ignore

            self._write_client = BigQueryWriteClient()
        return self._write_client

    def _get_storage_client(self) -> Any:
        if self._storage_client is None:
            from google.cloud import storage  # type: ignore

            self._storage_client = storage.Client(project=self.project_id)
        return self._storage_client

    def build_ledger_row(
        self,
        entity_id: str,
        serving_track: str,
        prediction: Mapping[str, Any],
        decision_id: str | None = None,
        decision_timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Construct and validate the locked 18-column ml_xai_audit_ledger row."""
        feature_payload = prediction.get("feature_payload", {})
        if isinstance(feature_payload, Mapping):
            pii_hits = sorted(RESTRICTED_PII_KEYS.intersection(feature_payload.keys()))
            if pii_hits:
                raise ValueError(f"Restricted PII keys forbidden in ledger payload: {', '.join(pii_hits)}")

        ts = decision_timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_iso = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        resolved_decision_id = decision_id or str(uuid.uuid4())
        resolved_entity_id = entity_id.strip() or str(prediction.get("entity_id", "")).strip()
        if not resolved_entity_id:
            raise ValueError("entity_id must be non-empty for audit ledger logging.")

        row: dict[str, Any] = {
            "decision_id": resolved_decision_id,
            "entity_id": resolved_entity_id,
            "decision_timestamp": ts_iso,
            "serving_track": serving_track,
            "model_id": str(prediction["model_id"]),
            "model_version_id": str(prediction["model_version_id"]),
            "model_artifact_sha256": str(prediction["model_artifact_sha256"]),
            "container_image_digest": str(prediction["container_image_digest"]),
            "schema_version": str(prediction.get("schema_version", "v1")),
            "explained_class": str(prediction.get("explained_class", "APPROVE")),
            "approve_prob": float(prediction["approve_prob"]),
            "raw_margin": float(prediction["raw_margin"]),
            "base_logit": float(prediction["base_logit"]),
            "shap_sum_logit": float(prediction["shap_sum_logit"]),
            "shap_logit_map": _serialize_json_field(prediction["shap_logit_map"]),
            "top_adverse_reasons": _serialize_json_field(prediction["top_adverse_reasons"]),
            "feature_payload": _serialize_json_field(feature_payload),
            "reason_rule_version": str(prediction["reason_rule_version"]),
        }
        missing = [col for col in LOCKED_LEDGER_COLUMNS if col not in row]
        if missing:
            raise ValueError(f"Missing locked ledger columns: {', '.join(missing)}")
        return row

    def _build_append_request(self, stream_name: str, row: Mapping[str, Any]) -> Any:
        try:
            from google.cloud.bigquery_storage_v1 import types  # type: ignore

            proto_rows = types.AppendRowsRequest.ProtoData()
            proto_rows.rows.serialized_rows.append(json.dumps(row).encode("utf-8"))
            req = types.AppendRowsRequest()
            req.write_stream = stream_name
            req.proto_rows = proto_rows
            return req
        except ImportError:
            return {"write_stream": stream_name, "rows": [dict(row)]}

    def commit_row_sync(self, row: Mapping[str, Any]) -> str:
        """Persist one audit row synchronously on a COMMITTED BigQuery write stream."""
        missing = [col for col in LOCKED_LEDGER_COLUMNS if col not in row]
        if missing:
            raise LedgerCommitError(f"Cannot commit incomplete ledger row: {missing}")

        client = self._get_write_client()
        try:
            try:
                from google.cloud.bigquery_storage_v1 import types  # type: ignore

                write_stream = types.WriteStream(type_=types.WriteStream.Type.COMMITTED)
            except ImportError:
                write_stream = {"type_": "COMMITTED"}

            created = client.create_write_stream(parent=self.table_parent, write_stream=write_stream)
            stream_name = getattr(created, "name", None) or created["name"]
            request = self._build_append_request(stream_name, row)
            responses = list(client.append_rows(iter([request])))
            if not responses:
                raise LedgerCommitError("BigQuery Storage Write API returned zero append responses.")
            for resp in responses:
                row_errors = getattr(resp, "row_errors", None)
                if row_errors is None and isinstance(resp, Mapping):
                    row_errors = resp.get("row_errors")
                if row_errors:
                    raise LedgerCommitError(f"BigQuery Storage Write API rejected row: {row_errors}")
                err = getattr(resp, "error", None)
                if err is None and isinstance(resp, Mapping):
                    err = resp.get("error")
                if err and getattr(err, "code", 0) != 0:
                    raise LedgerCommitError(f"BigQuery Storage Write API stream error: {err}")
        except LedgerCommitError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LedgerCommitError(f"BigQuery Storage Write API commit failed: {exc}") from exc

        if self.worm_bucket_name:
            try:
                ts_date = str(row["decision_timestamp"])[:10].replace("-", "/")
                blob_name = f"decisions/{ts_date}/{row['decision_id']}.json"
                storage_client = self._get_storage_client()
                bucket = storage_client.bucket(self.worm_bucket_name)
                blob = bucket.blob(blob_name)
                blob.upload_from_string(
                    json.dumps(dict(row), sort_keys=True) + "\n",
                    content_type="application/json",
                )
            except Exception as exc:  # noqa: BLE001
                raise LedgerCommitError(f"WORM audit receipt write failed: {exc}") from exc

        return str(row["decision_id"])
