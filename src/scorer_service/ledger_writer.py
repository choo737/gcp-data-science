"""Synchronous BigQuery Storage Write API _default-stream Protobuf ledger writer."""

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

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_STRING_FIELDS: frozenset[str] = frozenset({
    "decision_id",
    "entity_id",
    "serving_track",
    "model_id",
    "model_version_id",
    "model_artifact_sha256",
    "container_image_digest",
    "schema_version",
    "explained_class",
    "reason_rule_version",
})
_DOUBLE_FIELDS: frozenset[str] = frozenset({
    "approve_prob",
    "raw_margin",
    "base_logit",
    "shap_sum_logit",
})
_JSON_FIELDS: frozenset[str] = frozenset({
    "shap_logit_map",
    "top_adverse_reasons",
    "feature_payload",
})

_CACHED_DESCRIPTOR_PROTO: Any = None
_CACHED_ROW_CLASS: Any = None


class LedgerCommitError(RuntimeError):
    """Raised when synchronous XAI audit ledger persistence fails."""


def _serialize_json_field(value: Any) -> str:
    if isinstance(value, str):
        parsed = json.loads(value)
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _to_epoch_micros(value: Any) -> int:
    """Convert ISO-8601 string, datetime, or integer microseconds to UTC epoch microseconds."""
    if isinstance(value, bool):
        raise ValueError("decision_timestamp cannot be a bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError("decision_timestamp cannot be empty.")
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        raise ValueError(f"Unsupported decision_timestamp type: {type(value).__name__}")
    delta = dt.astimezone(timezone.utc) - _EPOCH_UTC
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _build_ledger_proto_schema() -> tuple[Any, Any]:
    """Build and cache a self-contained proto2 DescriptorProto and message class."""
    global _CACHED_DESCRIPTOR_PROTO, _CACHED_ROW_CLASS
    if _CACHED_DESCRIPTOR_PROTO is not None and _CACHED_ROW_CLASS is not None:
        return _CACHED_DESCRIPTOR_PROTO, _CACHED_ROW_CLASS

    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "ml_xai_audit_ledger_row.proto"
    file_proto.syntax = "proto2"

    msg_proto = file_proto.message_type.add()
    msg_proto.name = "MlXaiAuditLedgerRow"

    for number, col_name in enumerate(LOCKED_LEDGER_COLUMNS, start=1):
        field = msg_proto.field.add()
        field.name = col_name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REQUIRED
        if col_name == "decision_timestamp":
            field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
        elif col_name in _DOUBLE_FIELDS:
            field.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
        else:
            field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(file_proto.SerializeToString())
    msg_descriptor = pool.FindMessageTypeByName("MlXaiAuditLedgerRow")
    row_class = message_factory.GetMessageClass(msg_descriptor)

    descriptor_copy = descriptor_pb2.DescriptorProto()
    descriptor_copy.CopyFrom(msg_proto)
    _CACHED_DESCRIPTOR_PROTO = descriptor_copy
    _CACHED_ROW_CLASS = row_class
    return _CACHED_DESCRIPTOR_PROTO, _CACHED_ROW_CLASS


def get_ledger_row_descriptor_proto() -> Any:
    """Return a copy of the self-contained proto2 DescriptorProto for MlXaiAuditLedgerRow."""
    descriptor_proto, _ = _build_ledger_proto_schema()
    return descriptor_proto


def serialize_ledger_row_proto(row: Mapping[str, Any]) -> bytes:
    """Serialize one 18-column audit row into Protobuf wire bytes."""
    _, row_class = _build_ledger_proto_schema()
    kwargs: dict[str, Any] = {}
    for col_name in LOCKED_LEDGER_COLUMNS:
        if col_name not in row:
            raise LedgerCommitError(f"Cannot serialize incomplete ledger row: missing {col_name}")
        val = row[col_name]
        if col_name == "decision_timestamp":
            kwargs[col_name] = _to_epoch_micros(val)
        elif col_name in _DOUBLE_FIELDS:
            kwargs[col_name] = float(val)
        elif col_name in _JSON_FIELDS:
            kwargs[col_name] = _serialize_json_field(val)
        elif col_name in _STRING_FIELDS:
            kwargs[col_name] = str(val)
        else:
            kwargs[col_name] = str(val)
    msg = row_class(**kwargs)
    return msg.SerializeToString()


class FailClosedLedgerWriter:
    """Commits 18-column XAI ledger rows synchronously to BigQuery Storage Write _default stream."""

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

    @property
    def default_write_stream(self) -> str:
        return f"{self.table_parent}/streams/_default"

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

            descriptor_proto = get_ledger_row_descriptor_proto()
            serialized_row = serialize_ledger_row_proto(row)

            req = types.AppendRowsRequest()
            req.write_stream = stream_name
            req.proto_rows.writer_schema.proto_descriptor.CopyFrom(descriptor_proto)
            req.proto_rows.rows.serialized_rows.append(serialized_row)
            return req
        except ImportError:
            return {
                "write_stream": stream_name,
                "rows": [dict(row)],
            }

    def commit_row_sync(self, row: Mapping[str, Any]) -> str:
        """Persist one audit row synchronously on the BigQuery _default write stream."""
        missing = [col for col in LOCKED_LEDGER_COLUMNS if col not in row]
        if missing:
            raise LedgerCommitError(f"Cannot commit incomplete ledger row: {missing}")

        client = self._get_write_client()
        try:
            request = self._build_append_request(self.default_write_stream, row)
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
                if err is not None and getattr(err, "code", 0) != 0:
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
