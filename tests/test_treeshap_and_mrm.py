"""Hermetic unit tests for CPR TreeSHAP, fail-closed ledger writer, and MRM gate."""

from __future__ import annotations

import array
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.deploy_cpr_endpoint import parse_deploy_config
from src.cpr_treeshap.deploy_cpr import deploy_cpr_model
import src.cpr_treeshap.predictor as cpr_predictor
from src.cpr_treeshap.predictor import (
    REASON_RULE_VERSION,
    TreeShapCprPredictor,
    TreeShapIntegrityError,
    create_cpr_wsgi_app,
    parse_feature_names,
    rank_adverse_features,
    resolve_model_source_from_env,
)
from src.mrm_validation.evaluate_gate import (
    compute_ks_statistic,
    compute_roc_auc,
    compute_subgroup_fairness,
    evaluate_challenger_gate,
    main as mrm_main,
)
from src.scorer_service.app import (
    PscEndpointRequiredError,
    ScoringConfig,
    ScoringService,
    build_vertex_predict_url,
    create_scorer_wsgi_app,
    format_adverse_reasons_for_agent,
    invalidate_psc_cache,
)
from src.scorer_service.ledger_writer import (
    LOCKED_LEDGER_COLUMNS,
    FailClosedLedgerWriter,
    LedgerCommitError,
)


def _float32(val: float) -> float:
    """Simulate IEEE-754 float32 rounding returned by XGBoost C++ inference."""
    return struct.unpack("f", struct.pack("f", val))[0]


class _Float32BoosterAdapter:
    """Test booster simulating float32 margin and pred_contribs arrays."""

    def __init__(self, margin_f32: float, contribs_f32: list[float]) -> None:
        self._margin = array.array("f", [margin_f32])
        self._contribs = array.array("f", contribs_f32)

    def predict(
        self,
        dmatrix: object,
        output_margin: bool = False,
        pred_contribs: bool = False,
        approx_contribs: bool = False,
    ) -> object:
        if output_margin:
            return [self._margin[0]]
        if pred_contribs:
            assert approx_contribs is False
            return [list(self._contribs)]
        raise AssertionError("Unexpected predict mode")


class TreeShapAndMrmTest(unittest.TestCase):
    """Verify exact TreeSHAP math, fail-closed Storage Write commits, and MRM gates."""

    def test_omp_num_threads_locked_and_float32_additivity_passes(self) -> None:
        features = ["tenure_months", "delinquency_count_12m", "credit_utilization_ratio"]
        contribs = [_float32(0.45), _float32(-0.85), _float32(-0.20), _float32(0.10)]
        raw_margin = _float32(sum(contribs))
        booster = _Float32BoosterAdapter(margin_f32=raw_margin, contribs_f32=contribs)

        predictor = TreeShapCprPredictor(
            booster=booster,
            feature_names=features,
            model_id="credit_xgb_cpr",
            model_version_id="v3",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            additivity_tolerance=1e-5,
            dmatrix_factory=lambda rows, feature_names: {"rows": rows, "features": feature_names},
        )
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")

        out = predictor.predict_instance(
            {
                "tenure_months": 36,
                "delinquency_count_12m": 2,
                "credit_utilization_ratio": 0.78,
            },
            entity_id="cust-001",
        )
        self.assertEqual(out["entity_id"], "cust-001")
        self.assertIsInstance(out["raw_margin"], float)
        self.assertIsInstance(out["base_logit"], float)
        self.assertIsInstance(out["shap_sum_logit"], float)
        self.assertAlmostEqual(
            out["base_logit"] + out["shap_sum_logit"],
            out["raw_margin"],
            places=5,
        )
        self.assertEqual(out["top_adverse_reasons"][0]["feature"], "delinquency_count_12m")
        self.assertEqual(out["top_adverse_reasons"][0]["reason_code"], "RC_DELINQ_12M")

    def test_treeshap_additivity_breach_fails_closed(self) -> None:
        features = ["tenure_months", "delinquency_count_12m"]
        booster = _Float32BoosterAdapter(
            margin_f32=_float32(1.25),
            contribs_f32=[_float32(0.10), _float32(-0.20), _float32(0.05)],
        )
        predictor = TreeShapCprPredictor(
            booster=booster,
            feature_names=features,
            model_id="credit_xgb_cpr",
            model_version_id="v3",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            additivity_tolerance=1e-5,
            dmatrix_factory=lambda rows, feature_names: rows,
        )
        with self.assertRaisesRegex(TreeShapIntegrityError, "additivity breach"):
            predictor.predict_instance({"tenure_months": 12, "delinquency_count_12m": 1})

        cpr_app = create_cpr_wsgi_app(predictor)
        statuses: list[str] = []
        req = json.dumps({"instances": [{"tenure_months": 12, "delinquency_count_12m": 1}]}).encode("utf-8")
        chunks = cpr_app(
            {
                "PATH_INFO": "/predict",
                "REQUEST_METHOD": "POST",
                "CONTENT_LENGTH": str(len(req)),
                "wsgi.input": io.BytesIO(req),
            },
            lambda s, _h: statuses.append(s),
        )
        self.assertEqual(statuses[0], "500 Internal Server Error")
        self.assertIn("additivity breach", json.loads(b"".join(chunks).decode("utf-8"))["error"])

    def test_alphabetical_tie_breaking_and_pii_rejection(self) -> None:
        tied = rank_adverse_features(
            {
                "zeta_feature": -0.35,
                "alpha_feature": -0.35,
                "beta_feature": -0.35,
                "positive_helper": 0.60,
            },
            explained_class="APPROVE",
            top_k=3,
        )
        self.assertEqual([item["feature"] for item in tied], ["alpha_feature", "beta_feature", "zeta_feature"])

        default_tied = rank_adverse_features(
            {
                "zeta_feature": 0.40,
                "alpha_feature": 0.40,
                "negative_helper": -0.20,
            },
            explained_class="DEFAULT",
            top_k=2,
        )
        self.assertEqual([item["feature"] for item in default_tied], ["alpha_feature", "zeta_feature"])

        booster = _Float32BoosterAdapter(margin_f32=0.0, contribs_f32=[0.0, 0.0])
        predictor = TreeShapCprPredictor(
            booster=booster,
            feature_names=["tenure_months"],
            model_id="credit_xgb_cpr",
            model_version_id="v1",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            dmatrix_factory=lambda rows, feature_names: rows,
        )
        for pii_key in ("customer_ic_number", "nric", "passport_number"):
            with self.subTest(pii_key=pii_key):
                with self.assertRaisesRegex(ValueError, "Restricted PII"):
                    predictor.predict_instance({"tenure_months": 24, pii_key: "900101145566"})

    def test_load_from_file_sha256_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_file = Path(td) / "model.ubj"
            model_file.write_bytes(b"xgboost-binary-placeholder")
            bad_sha = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                TreeShapCprPredictor.load_from_file(
                    model_path=model_file,
                    expected_sha256=bad_sha,
                    feature_names=["tenure_months"],
                    model_id="credit_xgb_cpr",
                    model_version_id="v1",
                    container_image_digest="sha256:" + "b" * 64,
                )
            real_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
            self.assertEqual(len(real_sha), 64)

    def test_cpr_wsgi_app_health_and_predict(self) -> None:
        booster = _Float32BoosterAdapter(margin_f32=_float32(0.30), contribs_f32=[_float32(0.20), _float32(0.10)])
        predictor = TreeShapCprPredictor(
            booster=booster,
            feature_names=["tenure_months"],
            model_id="credit_xgb_cpr",
            model_version_id="v1",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            dmatrix_factory=lambda rows, feature_names: rows,
        )
        app = create_cpr_wsgi_app(predictor)
        statuses: list[str] = []
        payload = json.dumps({"instances": [{"tenure_months": 18}]}).encode("utf-8")
        body_chunks = app(
            {
                "PATH_INFO": "/predict",
                "REQUEST_METHOD": "POST",
                "CONTENT_LENGTH": str(len(payload)),
                "wsgi.input": io.BytesIO(payload),
            },
            lambda status, _headers: statuses.append(status),
        )
        self.assertEqual(statuses[0], "200 OK")
        parsed = json.loads(b"".join(body_chunks).decode("utf-8"))
        self.assertEqual(len(parsed["predictions"]), 1)

    def test_fail_closed_ledger_writer_and_scorer_503_on_storage_write_failure(self) -> None:
        self.assertEqual(len(LOCKED_LEDGER_COLUMNS), 18)
        cfg = ScoringConfig(
            ml_ops_project_id="ml-ops-proj",
            region="asia-southeast1",
            vertex_endpoint_id="1004200001",
            ml_audit_dataset_id="ml_xai_audit",
            ml_worm_bucket_name="worm-audit-bucket",
        )

        prediction_payload = {
            "entity_id": "cust-777",
            "model_id": "credit_xgb_cpr",
            "model_version_id": "v3",
            "model_artifact_sha256": "a" * 64,
            "container_image_digest": "sha256:" + "b" * 64,
            "schema_version": "v1",
            "explained_class": "APPROVE",
            "approve_prob": 0.45,
            "raw_margin": -0.20,
            "base_logit": 0.10,
            "shap_sum_logit": -0.30,
            "shap_logit_map": {"delinquency_count_12m": -0.30},
            "top_adverse_reasons": [
                {
                    "feature": "delinquency_count_12m",
                    "reason_code": "RC_DELINQ_12M",
                    "reason_text": "Recent delinquency events in the past twelve months.",
                    "shap_logit": -0.30,
                }
            ],
            "feature_payload": {"delinquency_count_12m": 2.0},
            "reason_rule_version": REASON_RULE_VERSION,
        }

        # Case 1: Storage Write returns row_errors -> must fail closed with HTTP 503.
        bad_write_client = MagicMock()
        bad_write_client.append_rows.return_value = iter([MagicMock(row_errors=["schema error"], error=None)])
        bad_writer = FailClosedLedgerWriter(
            project_id=cfg.ml_ops_project_id,
            dataset_id=cfg.ml_audit_dataset_id,
            write_client=bad_write_client,
        )
        failing_service = ScoringService(
            config=cfg,
            ledger_writer=bad_writer,
            vertex_predict_fn=lambda _c, _inst: prediction_payload,
        )
        status, body = failing_service.score_applicant({"entity_id": "cust-777", "features": {"delinquency_count_12m": 2}})
        self.assertEqual(status, 503)
        self.assertFalse(body["decision_released"])
        self.assertNotIn("approve_prob", body)
        bad_write_client.create_write_stream.assert_not_called()

        # Case 2: Storage Write raises exception -> WSGI app returns 503 Service Unavailable.
        exc_write_client = MagicMock()
        exc_write_client.append_rows.side_effect = RuntimeError("bq storage unavailable")
        exc_writer = FailClosedLedgerWriter(
            project_id=cfg.ml_ops_project_id,
            dataset_id=cfg.ml_audit_dataset_id,
            write_client=exc_write_client,
        )
        wsgi_app = create_scorer_wsgi_app(
            ScoringService(config=cfg, ledger_writer=exc_writer, vertex_predict_fn=lambda _c, _inst: prediction_payload)
        )
        statuses: list[str] = []
        req_bytes = json.dumps({"entity_id": "cust-777", "features": {"delinquency_count_12m": 2}}).encode("utf-8")
        resp_bytes = wsgi_app(
            {
                "PATH_INFO": "/score",
                "REQUEST_METHOD": "POST",
                "CONTENT_LENGTH": str(len(req_bytes)),
                "wsgi.input": io.BytesIO(req_bytes),
            },
            lambda s, _h: statuses.append(s),
        )
        self.assertEqual(statuses[0], "503 Service Unavailable")
        self.assertFalse(json.loads(b"".join(resp_bytes).decode("utf-8"))["decision_released"])

        # Case 3: Storage Write + WORM succeed -> HTTP 200 and agent helper omits raw logits.
        ok_write_client = MagicMock()
        captured_requests: list[object] = []

        def _capture_append(req_iter: object) -> object:
            captured_requests.extend(list(req_iter))  # type: ignore[arg-type]
            return iter([MagicMock(row_errors=[], error=None)])

        ok_write_client.append_rows.side_effect = _capture_append

        ok_blob = MagicMock()
        ok_bucket = MagicMock()
        ok_bucket.blob.return_value = ok_blob
        ok_storage = MagicMock()
        ok_storage.bucket.return_value = ok_bucket

        ok_writer = FailClosedLedgerWriter(
            project_id=cfg.ml_ops_project_id,
            dataset_id=cfg.ml_audit_dataset_id,
            write_client=ok_write_client,
            storage_client=ok_storage,
            worm_bucket_name=cfg.ml_worm_bucket_name,
        )
        ok_service = ScoringService(
            config=cfg,
            ledger_writer=ok_writer,
            vertex_predict_fn=lambda _c, _inst: prediction_payload,
        )
        ok_status, ok_body = ok_service.score_applicant({"entity_id": "cust-777", "features": {"delinquency_count_12m": 2}})
        self.assertEqual(ok_status, 200)
        self.assertTrue(ok_body["ledger_committed"])
        ok_write_client.create_write_stream.assert_not_called()
        self.assertEqual(len(captured_requests), 1)
        captured_req = captured_requests[0]
        stream_val = captured_req["write_stream"] if isinstance(captured_req, dict) else captured_req.write_stream  # type: ignore[attr-defined]
        self.assertTrue(stream_val.endswith("/streams/_default"))
        ok_blob.upload_from_string.assert_called_once()

        row = ok_writer.build_ledger_row("cust-777", "vertex_cpr_online", prediction_payload, decision_id="dec-1")
        self.assertEqual(set(row.keys()), set(LOCKED_LEDGER_COLUMNS))
        agent_view = format_adverse_reasons_for_agent(row)
        self.assertEqual(agent_view["decision_id"], "dec-1")
        self.assertNotIn("feature_payload", agent_view)
        self.assertNotIn("shap_logit_map", agent_view)
        self.assertNotIn("raw_margin", agent_view)
        self.assertEqual(agent_view["adverse_reasons"][0]["reason_code"], "RC_DELINQ_12M")

    def test_mrm_gate_metrics_fairness_parity_and_cli(self) -> None:
        y_true = [1, 1, 0, 0, 1, 1, 0, 0]
        good_scores = [0.95, 0.85, 0.25, 0.15, 0.90, 0.80, 0.20, 0.10]
        auc = compute_roc_auc(y_true, good_scores)
        ks = compute_ks_statistic(y_true, good_scores)
        self.assertAlmostEqual(auc, 1.0)
        self.assertAlmostEqual(ks, 1.0)

        # Disparate impact >= 0.80 passes; < 0.80 rejects.
        fair_groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
        fair_card = evaluate_challenger_gate(
            y_true=y_true,
            challenger_scores=good_scores,
            incumbent_scores=[0.90, 0.80, 0.30, 0.20, 0.88, 0.78, 0.22, 0.12],
            groups=fair_groups,
            treeshap_logits=[0.10, 0.20, -0.30],
            bqml_logits=[0.100001, 0.200002, -0.299999],
            min_disparate_impact=0.80,
        )
        self.assertTrue(fair_card["approved"])
        self.assertGreaterEqual(fair_card["fairness"]["disparate_impact_ratio"], 0.80)

        # Group B has 1/4 positive predictions (0.25), Group A has 4/4 (1.0) -> DI = 0.25 < 0.80.
        biased_y = [1, 1, 1, 0, 0, 0, 0, 1]
        biased_scores = [0.95, 0.90, 0.85, 0.80, 0.40, 0.30, 0.20, 0.70]
        biased_card = evaluate_challenger_gate(
            y_true=biased_y,
            challenger_scores=biased_scores,
            incumbent_scores=biased_scores,
            groups=["A", "A", "A", "A", "B", "B", "B", "B"],
            treeshap_logits=[0.10, 0.20],
            bqml_logits=[0.10, 0.20],
            min_disparate_impact=0.80,
        )
        self.assertFalse(biased_card["approved"])
        self.assertTrue(any("disparate_impact_below_min" in c for c in biased_card["failed_checks"]))

        # TreeSHAP vs BQML parity breach > 1e-5 rejects.
        parity_fail = evaluate_challenger_gate(
            y_true=y_true,
            challenger_scores=good_scores,
            incumbent_scores=good_scores,
            groups=fair_groups,
            treeshap_logits=[0.10],
            bqml_logits=[0.12],
            shap_parity_tolerance=1e-5,
        )
        self.assertFalse(parity_fail["approved"])
        self.assertTrue(any("shap_parity_breach" in c for c in parity_fail["failed_checks"]))

        # CLI test generating model_card.json.
        with tempfile.TemporaryDirectory() as td:
            in_path = Path(td) / "eval.json"
            out_path = Path(td) / "model_card.json"
            in_path.write_text(
                json.dumps({
                    "y_true": y_true,
                    "challenger_scores": good_scores,
                    "incumbent_scores": good_scores,
                    "groups": fair_groups,
                    "treeshap_logits": [0.1, -0.2],
                    "bqml_logits": [0.1, -0.2],
                }),
                encoding="utf-8",
            )
            rc = mrm_main(["--input-json", str(in_path), "--output-card", str(out_path)])
            self.assertEqual(rc, 0)
            card_json = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(card_json["approved"])

    def test_ledger_writer_protobuf_wire_deserialization_on_default_stream(self) -> None:
        try:
            from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
            from google.cloud.bigquery_storage_v1 import types as bq_types
        except ImportError:
            self.skipTest("protobuf and google-cloud-bigquery-storage not installed in bare Python")

        captured_requests: list[bq_types.AppendRowsRequest] = []
        write_client = MagicMock()

        def _capture_append(req_iter: object) -> object:
            captured_requests.extend(list(req_iter))  # type: ignore[arg-type]
            return iter([bq_types.AppendRowsResponse()])

        write_client.append_rows.side_effect = _capture_append

        writer = FailClosedLedgerWriter(
            project_id="ml-ops-proj",
            dataset_id="ml_xai_audit",
            write_client=write_client,
        )
        fixed_dt = datetime(2026, 11, 19, 3, 30, 15, 123456, tzinfo=timezone.utc)
        prediction_payload = {
            "entity_id": "ent-42",
            "model_id": "credit_xgb_cpr",
            "model_version_id": "v9",
            "model_artifact_sha256": "a" * 64,
            "container_image_digest": "sha256:" + "b" * 64,
            "schema_version": "v1",
            "explained_class": "APPROVE",
            "approve_prob": 0.731058,
            "raw_margin": 1.0,
            "base_logit": 0.25,
            "shap_sum_logit": 0.75,
            "shap_logit_map": {"tenure_months": 0.75},
            "top_adverse_reasons": [{"reason_code": "RC01", "reason_text": "Short tenure"}],
            "feature_payload": {"tenure_months": 12},
            "reason_rule_version": REASON_RULE_VERSION,
        }
        row = writer.build_ledger_row(
            entity_id="ent-42",
            serving_track="vertex_cpr_online",
            prediction=prediction_payload,
            decision_id="dec-proto-1",
            decision_timestamp=fixed_dt,
        )
        committed_id = writer.commit_row_sync(row)
        self.assertEqual(committed_id, "dec-proto-1")
        write_client.create_write_stream.assert_not_called()
        self.assertEqual(len(captured_requests), 1)

        req = captured_requests[0]
        self.assertTrue(req.write_stream.endswith("/streams/_default"))
        set_field_names = {f.name for f, _ in req._pb.ListFields()}
        self.assertNotIn("offset", set_field_names)

        proto_desc = req.proto_rows.writer_schema.proto_descriptor
        self.assertEqual(proto_desc.name, "MlXaiAuditLedgerRow")
        self.assertEqual(len(proto_desc.field), 18)
        self.assertTrue(all(f.label == descriptor_pb2.FieldDescriptorProto.LABEL_REQUIRED for f in proto_desc.field))

        fresh_pool = descriptor_pool.DescriptorPool()
        fdp = descriptor_pb2.FileDescriptorProto(name="verify_ledger_row.proto", syntax="proto2")
        fdp.message_type.add().CopyFrom(proto_desc)
        fresh_pool.AddSerializedFile(fdp.SerializeToString())
        row_cls = message_factory.GetMessageClass(fresh_pool.FindMessageTypeByName("MlXaiAuditLedgerRow"))
        decoded = row_cls()
        decoded.ParseFromString(req.proto_rows.rows.serialized_rows[0])

        self.assertEqual(decoded.decision_id, "dec-proto-1")
        self.assertEqual(decoded.entity_id, "ent-42")
        expected_micros = int(fixed_dt.timestamp() * 1_000_000)
        self.assertEqual(decoded.decision_timestamp, expected_micros)
        self.assertEqual(decoded.serving_track, "vertex_cpr_online")
        self.assertEqual(decoded.model_id, "credit_xgb_cpr")
        self.assertEqual(decoded.model_version_id, "v9")
        self.assertEqual(decoded.model_artifact_sha256, "a" * 64)
        self.assertEqual(decoded.container_image_digest, "sha256:" + "b" * 64)
        self.assertEqual(decoded.schema_version, "v1")
        self.assertEqual(decoded.explained_class, "APPROVE")
        self.assertAlmostEqual(decoded.approve_prob, 0.731058)
        self.assertAlmostEqual(decoded.raw_margin, 1.0)
        self.assertAlmostEqual(decoded.base_logit, 0.25)
        self.assertAlmostEqual(decoded.shap_sum_logit, 0.75)
        self.assertEqual(json.loads(decoded.shap_logit_map), {"tenure_months": 0.75})
        self.assertEqual(json.loads(decoded.top_adverse_reasons), [{"reason_code": "RC01", "reason_text": "Short tenure"}])
        self.assertEqual(json.loads(decoded.feature_payload), {"tenure_months": 12})
        self.assertEqual(decoded.reason_rule_version, REASON_RULE_VERSION)

        # Verify response.error.code != 0 raises LedgerCommitError.
        err_resp = bq_types.AppendRowsResponse()
        err_resp.error.code = 3
        err_resp.error.message = "invalid argument"
        write_client.append_rows.side_effect = None
        write_client.append_rows.return_value = iter([err_resp])
        with self.assertRaisesRegex(LedgerCommitError, "stream error"):
            writer.commit_row_sync(row)

    def test_container_entrypoint_imports_and_scorer_wsgi_health(self) -> None:
        src_dir = REPO_ROOT / "src"
        checks = [
            (str(src_dir), "import scorer_service.app as m; assert callable(m.application)"),
            (str(src_dir / "scorer_service"), "import app as m; assert callable(m.application)"),
            (str(src_dir / "cpr_treeshap"), "import predictor as m; assert callable(m.application)"),
        ]
        for py_path, code in checks:
            env = os.environ.copy()
            env["PYTHONPATH"] = py_path
            res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
            self.assertEqual(res.returncode, 0, f"Import failed for PYTHONPATH={py_path}: {res.stderr}")

        from src.scorer_service.app import application as scorer_wsgi_application

        statuses: list[str] = []
        chunks = scorer_wsgi_application(
            {"PATH_INFO": "/health", "REQUEST_METHOD": "GET"},
            lambda s, _h: statuses.append(s),
        )
        self.assertEqual(statuses[0], "200 OK")
        self.assertEqual(json.loads(b"".join(chunks).decode("utf-8")), {"status": "ok"})

    def test_psc_routing_fail_closed_and_cpr_storage_uri(self) -> None:
        psc_cfg = ScoringConfig(
            ml_ops_project_id="ml-ops-proj",
            region="asia-southeast1",
            vertex_endpoint_id="1004200001",
            ml_audit_dataset_id="ml_xai_audit",
            vertex_psc_endpoint_ip="10.20.0.15",
            environment="prod",
        )
        self.assertEqual(
            build_vertex_predict_url(psc_cfg),
            "http://10.20.0.15/v1/projects/ml-ops-proj/locations/asia-southeast1/endpoints/1004200001:predict",
        )

        prod_no_psc = ScoringConfig(
            ml_ops_project_id="ml-ops-proj",
            region="asia-southeast1",
            vertex_endpoint_id="1004200001",
            ml_audit_dataset_id="ml_xai_audit",
            vertex_psc_endpoint_ip="",
            environment="prod",
        )
        with self.assertRaises(PscEndpointRequiredError):
            build_vertex_predict_url(prod_no_psc)

        writer = FailClosedLedgerWriter(project_id="ml-ops-proj", dataset_id="ml_xai_audit", write_client=MagicMock())
        svc = ScoringService(
            config=prod_no_psc,
            ledger_writer=writer,
            endpoint_metadata_fetcher=lambda _c: {"privateServiceConnectConfig": {"pscAutomationConfigs": []}},
        )
        status, body = svc.score_applicant({"entity_id": "ent-1", "features": {"tenure_months": 12}})
        self.assertEqual(status, 503)
        self.assertFalse(body["decision_released"])

        old_aip = os.environ.get("AIP_STORAGE_URI")
        old_fn = os.environ.get("MODEL_FILENAME")
        try:
            os.environ["AIP_STORAGE_URI"] = "gs://ml-models/run-1/"
            os.environ["MODEL_FILENAME"] = "model.ubj"
            self.assertEqual(resolve_model_source_from_env(), "gs://ml-models/run-1/model.ubj")
        finally:
            if old_aip is None:
                os.environ.pop("AIP_STORAGE_URI", None)
            else:
                os.environ["AIP_STORAGE_URI"] = old_aip
            if old_fn is None:
                os.environ.pop("MODEL_FILENAME", None)
            else:
                os.environ["MODEL_FILENAME"] = old_fn

        self.assertEqual(parse_feature_names('["tenure_months", "avg_monthly_spend"]'), ["tenure_months", "avg_monthly_spend"])
        self.assertEqual(parse_feature_names("tenure_months, avg_monthly_spend"), ["tenure_months", "avg_monthly_spend"])

        fake_blob = MagicMock()
        fake_blob.download_to_filename.side_effect = lambda p: Path(p).write_bytes(b"untrusted-model")
        fake_bucket = MagicMock()
        fake_bucket.blob.return_value = fake_blob
        fake_storage = MagicMock()
        fake_storage.bucket.return_value = fake_bucket
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            TreeShapCprPredictor.load_from_file(
                model_path="gs://ml-models/run-1/model.ubj",
                expected_sha256="0" * 64,
                feature_names=["tenure_months"],
                model_id="credit_xgb_cpr",
                model_version_id="v1",
                container_image_digest="sha256:" + "b" * 64,
                storage_client=fake_storage,
            )

        parsed = parse_deploy_config({
            "ML_OPS_PROJECT_ID": "ml-ops-proj",
            "REGION": "asia-southeast1",
            "VERTEX_ENDPOINT_ID": "1004200001",
            "CPR_CONTAINER_IMAGE": "asia-southeast1-docker.pkg.dev/ml-ops-proj/repo/cpr@sha256:" + "a" * 64,
            "CPR_MODEL_ARTIFACT_URI": "gs://ml-models/run-1/",
            "EXPECTED_MODEL_SHA256": "c" * 64,
            "FEATURE_NAMES": '["tenure_months", "avg_monthly_spend"]',
        })
        self.assertEqual(parsed["cpr_model_artifact_uri"], "gs://ml-models/run-1")
        self.assertEqual(parsed["feature_names"], ["tenure_months", "avg_monthly_spend"])

    def test_deploy_cpr_rest_upload_lro_deploy_and_idempotency(self) -> None:
        cfg = parse_deploy_config({
            "ML_OPS_PROJECT_ID": "ml-ops-proj",
            "REGION": "asia-southeast1",
            "VERTEX_ENDPOINT_ID": "1004200001",
            "CPR_CONTAINER_IMAGE": "asia-southeast1-docker.pkg.dev/ml-ops-proj/repo/cpr@sha256:" + "a" * 64,
            "CPR_MODEL_ARTIFACT_URI": "gs://ml-models/run-1/",
            "EXPECTED_MODEL_SHA256": "c" * 64,
            "FEATURE_NAMES": '["tenure_months", "avg_monthly_spend"]',
            "MODEL_DISPLAY_NAME": "ent-credit-cpr-treeshap",
        })

        def _make_resp(payload: dict[str, object]) -> MagicMock:
            resp = MagicMock()
            resp.json.return_value = payload
            resp.raise_for_status.return_value = None
            return resp

        # Case 1: Fresh upload + LRO poll + deployModel + LRO poll.
        session = MagicMock()
        session.get.side_effect = [
            _make_resp({"deployedModels": []}),
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload", "done": False}),
            _make_resp({
                "name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload",
                "done": True,
                "response": {"model": "projects/ml-ops-proj/locations/asia-southeast1/models/9001"},
            }),
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-deploy", "done": True}),
        ]
        session.post.side_effect = [
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload", "done": False}),
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-deploy", "done": False}),
        ]
        deployed = deploy_cpr_model(cfg, session=session, sleep_fn=lambda _s: None, poll_interval_seconds=0.0)
        self.assertEqual(deployed, "projects/ml-ops-proj/locations/asia-southeast1/models/9001")
        self.assertEqual(session.post.call_count, 2)
        upload_url, upload_kwargs = session.post.call_args_list[0][0][0], session.post.call_args_list[0][1]
        self.assertTrue(upload_url.endswith("/projects/ml-ops-proj/locations/asia-southeast1/models:upload"))
        env_items = {e["name"]: e["value"] for e in upload_kwargs["json"]["model"]["containerSpec"]["env"]}
        self.assertEqual(env_items["EXPECTED_MODEL_SHA256"], "c" * 64)
        self.assertEqual(env_items["MODEL_VERSION_ID"], ("c" * 64)[:12])
        self.assertEqual(
            env_items["CONTAINER_IMAGE_DIGEST"],
            "asia-southeast1-docker.pkg.dev/ml-ops-proj/repo/cpr@sha256:" + "a" * 64,
        )
        deploy_url, deploy_kwargs = session.post.call_args_list[1][0][0], session.post.call_args_list[1][1]
        self.assertTrue(deploy_url.endswith("/endpoints/1004200001:deployModel"))
        self.assertEqual(deploy_kwargs["json"]["trafficSplit"], {"0": 100})

        # Case 2: Idempotent skip when matching displayName, imageUri, SHA-256, and runtime env vars already deployed.
        idem_session = MagicMock()
        idem_session.get.side_effect = [
            _make_resp({
                "deployedModels": [
                    {
                        "id": "1",
                        "displayName": "ent-credit-cpr-treeshap",
                        "model": "projects/ml-ops-proj/locations/asia-southeast1/models/9001",
                    }
                ]
            }),
            _make_resp({
                "name": "projects/ml-ops-proj/locations/asia-southeast1/models/9001",
                "containerSpec": {
                    "imageUri": cfg["cpr_container_image"],
                    "env": [
                        {"name": "EXPECTED_MODEL_SHA256", "value": "c" * 64},
                        {"name": "FEATURE_NAMES", "value": '["tenure_months", "avg_monthly_spend"]'},
                        {"name": "SHAP_ADDITIVITY_TOLERANCE", "value": "0.00001"},
                        {"name": "REASON_RULE_VERSION", "value": "bnm-fair-lending-v1.0"},
                    ],
                },
            }),
        ]
        idem_result = deploy_cpr_model(cfg, session=idem_session, sleep_fn=lambda _s: None)
        self.assertEqual(idem_result, "projects/ml-ops-proj/locations/asia-southeast1/models/9001")
        idem_session.post.assert_not_called()

        # Case 2b: Container image or env drift triggers re-upload, deploy, and undeploy of stale model.
        for drift_label, drifted_spec in (
            (
                "image_drift",
                {
                    "imageUri": "asia-southeast1-docker.pkg.dev/ml-ops-proj/repo/cpr@sha256:" + "f" * 64,
                    "env": [
                        {"name": "EXPECTED_MODEL_SHA256", "value": "c" * 64},
                        {"name": "FEATURE_NAMES", "value": '["tenure_months", "avg_monthly_spend"]'},
                        {"name": "SHAP_ADDITIVITY_TOLERANCE", "value": "0.00001"},
                        {"name": "REASON_RULE_VERSION", "value": "bnm-fair-lending-v1.0"},
                    ],
                },
            ),
            (
                "env_drift",
                {
                    "imageUri": cfg["cpr_container_image"],
                    "env": [
                        {"name": "EXPECTED_MODEL_SHA256", "value": "c" * 64},
                        {"name": "FEATURE_NAMES", "value": '["tenure_months"]'},
                        {"name": "SHAP_ADDITIVITY_TOLERANCE", "value": "0.00001"},
                        {"name": "REASON_RULE_VERSION", "value": "bnm-fair-lending-v1.0"},
                    ],
                },
            ),
        ):
            with self.subTest(drift=drift_label):
                drift_session = MagicMock()
                drift_session.get.side_effect = [
                    _make_resp({
                        "deployedModels": [
                            {
                                "id": "old-dm-1",
                                "displayName": "ent-credit-cpr-treeshap",
                                "model": "projects/ml-ops-proj/locations/asia-southeast1/models/9001",
                            }
                        ]
                    }),
                    _make_resp({
                        "name": "projects/ml-ops-proj/locations/asia-southeast1/models/9001",
                        "containerSpec": drifted_spec,
                    }),
                    _make_resp({
                        "name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload",
                        "done": True,
                        "response": {"model": "projects/ml-ops-proj/locations/asia-southeast1/models/9002"},
                    }),
                    _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-deploy", "done": True}),
                    _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-undeploy", "done": True}),
                ]
                drift_session.post.side_effect = [
                    _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload", "done": False}),
                    _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-deploy", "done": False}),
                    _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-undeploy", "done": False}),
                ]
                redeployed = deploy_cpr_model(cfg, session=drift_session, sleep_fn=lambda _s: None, poll_interval_seconds=0.0)
                self.assertEqual(redeployed, "projects/ml-ops-proj/locations/asia-southeast1/models/9002")
                self.assertEqual(drift_session.post.call_count, 3)

        # Case 3: LRO error raises RuntimeError.
        err_session = MagicMock()
        err_session.get.side_effect = [
            _make_resp({"deployedModels": []}),
            _make_resp({
                "name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload",
                "done": True,
                "error": {"code": 13, "message": "internal upload failure"},
            }),
        ]
        err_session.post.return_value = _make_resp({
            "name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload",
            "done": False,
        })
        with self.assertRaisesRegex(RuntimeError, "Agent Platform operation failed"):
            deploy_cpr_model(cfg, session=err_session, sleep_fn=lambda _s: None, poll_interval_seconds=0.0)

    def test_psc_lazy_discovery_cache_fail_closed_and_stale_refresh(self) -> None:
        invalidate_psc_cache()
        try:
            prod_cfg = ScoringConfig(
                ml_ops_project_id="ml-ops-proj",
                region="asia-southeast1",
                vertex_endpoint_id="1004200001",
                ml_audit_dataset_id="ml_xai_audit",
                vertex_psc_endpoint_ip="",
                environment="prod",
            )
            ok_prediction = {
                "predictions": [
                    {
                        "entity_id": "ent-100",
                        "model_id": "credit_xgb_cpr",
                        "model_version_id": "v1",
                        "model_artifact_sha256": "a" * 64,
                        "container_image_digest": "sha256:" + "b" * 64,
                        "schema_version": "v1",
                        "explained_class": "APPROVE",
                        "approve_prob": 0.80,
                        "raw_margin": 1.386294,
                        "base_logit": 0.50,
                        "shap_sum_logit": 0.886294,
                        "shap_logit_map": {"tenure_months": 0.886294},
                        "top_adverse_reasons": [],
                        "feature_payload": {"tenure_months": 24},
                        "reason_rule_version": REASON_RULE_VERSION,
                    }
                ]
            }

            # (a) Empty VERTEX_PSC_ENDPOINT_IP in prod succeeds via control-plane discovery and caches.
            write_client = MagicMock()
            write_client.append_rows.side_effect = lambda reqs: iter([MagicMock(row_errors=[], error=None)])
            writer = FailClosedLedgerWriter(project_id="ml-ops-proj", dataset_id="ml_xai_audit", write_client=write_client)

            fetcher = MagicMock(return_value={
                "privateServiceConnectConfig": {
                    "pscAutomationConfigs": [
                        {"state": "PSC_AUTOMATION_STATE_PENDING", "ipAddress": ""},
                        {"state": "PSC_AUTOMATION_STATE_SUCCESSFUL", "ipAddress": "10.10.0.25"},
                    ]
                }
            })
            seen_urls: list[str] = []

            def _post_ok(url: str, _inst: object) -> dict[str, object]:
                seen_urls.append(url)
                return ok_prediction

            svc = ScoringService(
                config=prod_cfg,
                ledger_writer=writer,
                endpoint_metadata_fetcher=fetcher,
                http_post_fn=_post_ok,
            )
            s1, b1 = svc.score_applicant({"entity_id": "ent-100", "features": {"tenure_months": 24}})
            s2, b2 = svc.score_applicant({"entity_id": "ent-100", "features": {"tenure_months": 24}})
            self.assertEqual(s1, 200)
            self.assertEqual(s2, 200)
            self.assertTrue(b1["ledger_committed"])
            self.assertTrue(b2["ledger_committed"])
            self.assertEqual(fetcher.call_count, 1)
            self.assertEqual(
                seen_urls,
                [
                    "http://10.10.0.25/v1/projects/ml-ops-proj/locations/asia-southeast1/endpoints/1004200001:predict",
                    "http://10.10.0.25/v1/projects/ml-ops-proj/locations/asia-southeast1/endpoints/1004200001:predict",
                ],
            )

            # (b) Empty ipAddress in prod fails closed (503) and does NOT permanently cache "".
            invalidate_psc_cache(prod_cfg)
            dynamic_fetcher = MagicMock(side_effect=[
                {"privateServiceConnectConfig": {"pscAutomationConfigs": [{"ipAddress": ""}]}},
                {"privateServiceConnectConfig": {"pscAutomationConfigs": [{"ipAddress": "10.10.0.26"}]}},
            ])
            seen_urls.clear()
            svc_dyn = ScoringService(
                config=prod_cfg,
                ledger_writer=writer,
                endpoint_metadata_fetcher=dynamic_fetcher,
                http_post_fn=_post_ok,
            )
            s_fail, b_fail = svc_dyn.score_applicant({"entity_id": "ent-100", "features": {"tenure_months": 24}})
            self.assertEqual(s_fail, 503)
            self.assertFalse(b_fail["decision_released"])
            s_recover, _ = svc_dyn.score_applicant({"entity_id": "ent-100", "features": {"tenure_months": 24}})
            self.assertEqual(s_recover, 200)
            self.assertEqual(dynamic_fetcher.call_count, 2)
            self.assertTrue(seen_urls[0].startswith("http://10.10.0.26/"))

            # (c) ConnectionError on stale cached IP invalidates cache and refreshes once.
            stale_attempts: list[str] = []

            def _post_stale_then_ok(url: str, _inst: object) -> dict[str, object]:
                stale_attempts.append(url)
                if "10.10.0.26" in url:
                    raise ConnectionError("stale PSC endpoint IP")
                return ok_prediction

            refresh_fetcher = MagicMock(return_value={
                "privateServiceConnectConfig": {
                    "pscAutomationConfigs": [{"state": "PSC_AUTOMATION_STATE_SUCCESSFUL", "ipAddress": "10.10.0.30"}]
                }
            })
            svc_refresh = ScoringService(
                config=prod_cfg,
                ledger_writer=writer,
                endpoint_metadata_fetcher=refresh_fetcher,
                http_post_fn=_post_stale_then_ok,
            )
            s_ref, _ = svc_refresh.score_applicant({"entity_id": "ent-100", "features": {"tenure_months": 24}})
            self.assertEqual(s_ref, 200)
            self.assertEqual(refresh_fetcher.call_count, 1)
            self.assertEqual(len(stale_attempts), 2)
            self.assertTrue(stale_attempts[0].startswith("http://10.10.0.26/"))
            self.assertTrue(stale_attempts[1].startswith("http://10.10.0.30/"))

            # (d) Stale VERTEX_PSC_ENDPOINT_IP env var refreshes once to 10.0.0.2, then second /score hits cache directly.
            invalidate_psc_cache()
            stale_env_cfg = ScoringConfig(
                ml_ops_project_id="ml-ops-proj",
                region="asia-southeast1",
                vertex_endpoint_id="1004200001",
                ml_audit_dataset_id="ml_xai_audit",
                vertex_psc_endpoint_ip="10.0.0.1",
                environment="prod",
            )
            env_attempts: list[str] = []

            def _post_env_failover(url: str, _inst: object) -> dict[str, object]:
                env_attempts.append(url)
                if "10.0.0.1" in url:
                    raise ConnectionError("old env IP unreachable after model swap")
                return ok_prediction

            env_fetcher = MagicMock(return_value={
                "privateServiceConnectConfig": {
                    "pscAutomationConfigs": [{"state": "PSC_AUTOMATION_STATE_SUCCESSFUL", "ipAddress": "10.0.0.2"}]
                }
            })
            svc_env = ScoringService(
                config=stale_env_cfg,
                ledger_writer=writer,
                endpoint_metadata_fetcher=env_fetcher,
                http_post_fn=_post_env_failover,
            )
            r1_status, _ = svc_env.score_applicant({"entity_id": "ent-100", "features": {"tenure_months": 24}})
            r2_status, _ = svc_env.score_applicant({"entity_id": "ent-100", "features": {"tenure_months": 24}})
            self.assertEqual(r1_status, 200)
            self.assertEqual(r2_status, 200)
            self.assertEqual(env_fetcher.call_count, 1)
            self.assertEqual(
                env_attempts,
                [
                    "http://10.0.0.1/v1/projects/ml-ops-proj/locations/asia-southeast1/endpoints/1004200001:predict",
                    "http://10.0.0.2/v1/projects/ml-ops-proj/locations/asia-southeast1/endpoints/1004200001:predict",
                    "http://10.0.0.2/v1/projects/ml-ops-proj/locations/asia-southeast1/endpoints/1004200001:predict",
                ],
            )
        finally:
            invalidate_psc_cache()

    def test_cpr_thread_safe_lazy_init_and_non_blocking_health(self) -> None:
        env_keys = (
            "AIP_STORAGE_URI",
            "MODEL_PATH",
            "EXPECTED_MODEL_SHA256",
            "FEATURE_NAMES",
            "MODEL_FILENAME",
        )
        saved_env = {k: os.environ.get(k) for k in env_keys}
        try:
            cpr_predictor.reset_lazy_state_for_tests()

            # Missing env on /health returns 500 Internal Server Error.
            for k in env_keys:
                os.environ.pop(k, None)
            err_statuses: list[str] = []
            cpr_predictor.application(
                {"PATH_INFO": "/health", "REQUEST_METHOD": "GET"},
                lambda s, _h: err_statuses.append(s),
            )
            self.assertEqual(err_statuses[0], "500 Internal Server Error")

            # Cold gs:// AIP_STORAGE_URI on /health starts background load, returns 503 while in-flight, then 200.
            os.environ["AIP_STORAGE_URI"] = "gs://ml-models/run-1"
            os.environ["EXPECTED_MODEL_SHA256"] = "a" * 64
            os.environ["FEATURE_NAMES"] = '["tenure_months"]'

            booster = _Float32BoosterAdapter(margin_f32=_float32(0.30), contribs_f32=[_float32(0.20), _float32(0.10)])
            fake_pred = TreeShapCprPredictor(
                booster=booster,
                feature_names=["tenure_months"],
                model_id="credit_xgb_cpr",
                model_version_id="v1",
                model_artifact_sha256="a" * 64,
                container_image_digest="sha256:" + "b" * 64,
                dmatrix_factory=lambda rows, feature_names: rows,
            )

            load_started = threading.Event()
            allow_finish = threading.Event()

            def _slow_gcs_load(**_kwargs: object) -> TreeShapCprPredictor:
                load_started.set()
                allow_finish.wait(timeout=5.0)
                return fake_pred

            with patch.object(cpr_predictor.TreeShapCprPredictor, "load_from_file", side_effect=_slow_gcs_load):
                health_statuses: list[str] = []
                chunks = cpr_predictor.application(
                    {"PATH_INFO": "/health", "REQUEST_METHOD": "GET"},
                    lambda s, _h: health_statuses.append(s),
                )
                self.assertEqual(health_statuses[0], "503 Service Unavailable")
                payload = json.loads(b"".join(chunks).decode("utf-8"))
                self.assertEqual(payload["status"], "loading")
                self.assertFalse(payload["model_loaded"])
                self.assertTrue(load_started.wait(timeout=2.0))

                allow_finish.set()
                if cpr_predictor._BG_LOAD_THREAD is not None:
                    cpr_predictor._BG_LOAD_THREAD.join(timeout=2.0)

                ready_statuses: list[str] = []
                ready_chunks = cpr_predictor.application(
                    {"PATH_INFO": "/health", "REQUEST_METHOD": "GET"},
                    lambda s, _h: ready_statuses.append(s),
                )
                self.assertEqual(ready_statuses[0], "200 OK")
                ready_payload = json.loads(b"".join(ready_chunks).decode("utf-8"))
                self.assertTrue(ready_payload["model_loaded"])

            # Failed background gs:// SHA-256 verification causes /health to return 500.
            cpr_predictor.reset_lazy_state_for_tests()
            with patch.object(
                cpr_predictor.TreeShapCprPredictor,
                "load_from_file",
                side_effect=ValueError("Model artifact SHA-256 mismatch"),
            ):
                cpr_predictor.application(
                    {"PATH_INFO": "/health", "REQUEST_METHOD": "GET"},
                    lambda _s, _h: None,
                )
                if cpr_predictor._BG_LOAD_THREAD is not None:
                    cpr_predictor._BG_LOAD_THREAD.join(timeout=2.0)
                fail_statuses: list[str] = []
                fail_chunks = cpr_predictor.application(
                    {"PATH_INFO": "/health", "REQUEST_METHOD": "GET"},
                    lambda s, _h: fail_statuses.append(s),
                )
                self.assertEqual(fail_statuses[0], "500 Internal Server Error")
                self.assertIn("SHA-256 mismatch", json.loads(b"".join(fail_chunks).decode("utf-8"))["error"])

            # Concurrent /predict requests trigger load_from_file exactly once under _LAZY_CPR_LOCK.
            cpr_predictor.reset_lazy_state_for_tests()
            os.environ.pop("AIP_STORAGE_URI", None)
            with tempfile.TemporaryDirectory() as td:
                model_path = Path(td) / "model.ubj"
                model_path.write_bytes(b"stub-bytes")
                os.environ["MODEL_PATH"] = str(model_path)
                os.environ["EXPECTED_MODEL_SHA256"] = hashlib.sha256(b"stub-bytes").hexdigest()

                barrier = threading.Barrier(4)
                results: list[str] = []
                lock = threading.Lock()

                with patch.object(cpr_predictor.TreeShapCprPredictor, "load_from_file", return_value=fake_pred) as mock_load:
                    def _worker() -> None:
                        barrier.wait()
                        statuses: list[str] = []
                        req = json.dumps({"instances": [{"tenure_months": 18}]}).encode("utf-8")
                        cpr_predictor.application(
                            {
                                "PATH_INFO": "/predict",
                                "REQUEST_METHOD": "POST",
                                "CONTENT_LENGTH": str(len(req)),
                                "wsgi.input": io.BytesIO(req),
                            },
                            lambda s, _h: statuses.append(s),
                        )
                        with lock:
                            results.append(statuses[0])

                    threads = [threading.Thread(target=_worker) for _ in range(4)]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()

                    self.assertEqual(mock_load.call_count, 1)
                    self.assertEqual(results, ["200 OK"] * 4)
        finally:
            cpr_predictor.reset_lazy_state_for_tests()
            for k, val in saved_env.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val

    def test_evaluate_gate_fails_closed_when_subgroup_lacks_negatives_or_positives(self) -> None:
        with self.assertRaisesRegex(ValueError, "neg_total=0"):
            compute_subgroup_fairness(
                y_true=[1, 1, 1, 0],
                y_pred=[1, 1, 0, 0],
                groups=["A", "A", "B", "B"],
            )
        with self.assertRaisesRegex(ValueError, "pos_total=0"):
            compute_subgroup_fairness(
                y_true=[0, 0, 1, 0],
                y_pred=[0, 0, 1, 0],
                groups=["A", "A", "B", "B"],
            )

    def test_ledger_writer_writes_worm_before_bigquery_and_enforces_require_worm(self) -> None:
        call_order: list[str] = []

        def _record_append(req_iter: object) -> object:
            list(req_iter)  # type: ignore[arg-type]
            call_order.append("bigquery_append")
            return iter([MagicMock(row_errors=[], error=None)])

        write_client = MagicMock()
        write_client.append_rows.side_effect = _record_append

        blob = MagicMock()
        blob.upload_from_string.side_effect = lambda *_args, **_kwargs: call_order.append("gcs_worm_upload")
        bucket = MagicMock()
        bucket.blob.return_value = blob
        storage_client = MagicMock()
        storage_client.bucket.return_value = bucket

        writer = FailClosedLedgerWriter(
            project_id="ml-ops-proj",
            dataset_id="ml_xai_audit",
            write_client=write_client,
            storage_client=storage_client,
            worm_bucket_name="worm-audit-bucket",
            require_worm=True,
        )
        prediction_payload = {
            "entity_id": "ent-worm-1",
            "model_id": "credit_xgb_cpr",
            "model_version_id": "v1",
            "model_artifact_sha256": "a" * 64,
            "container_image_digest": "sha256:" + "b" * 64,
            "schema_version": "v1",
            "explained_class": "APPROVE",
            "approve_prob": 0.65,
            "raw_margin": 0.619,
            "base_logit": 0.119,
            "shap_sum_logit": 0.50,
            "shap_logit_map": {"tenure_months": 0.50},
            "top_adverse_reasons": [],
            "feature_payload": {"tenure_months": 24},
            "reason_rule_version": REASON_RULE_VERSION,
        }
        row = writer.build_ledger_row("ent-worm-1", "vertex_cpr_online", prediction_payload, decision_id="dec-worm-1")
        committed_id = writer.commit_row_sync(row)
        self.assertEqual(committed_id, "dec-worm-1")
        self.assertEqual(call_order, ["gcs_worm_upload", "bigquery_append"])

        # GCS WORM write failure must abort before BigQuery Storage Write is called.
        call_order.clear()
        write_client.append_rows.reset_mock()
        blob.upload_from_string.side_effect = RuntimeError("gcs bucket lock error")
        with self.assertRaisesRegex(LedgerCommitError, "WORM audit receipt write failed"):
            writer.commit_row_sync(row)
        write_client.append_rows.assert_not_called()

        # require_worm=True with empty worm_bucket_name must fail closed before BigQuery write.
        no_worm_writer = FailClosedLedgerWriter(
            project_id="ml-ops-proj",
            dataset_id="ml_xai_audit",
            write_client=write_client,
            worm_bucket_name="",
            require_worm=True,
        )
        with self.assertRaisesRegex(LedgerCommitError, "require_worm"):
            no_worm_writer.commit_row_sync(row)
        write_client.append_rows.assert_not_called()

    def test_chunked_http_bodies_and_ubj_uri_normalization(self) -> None:
        # 1. CPR predictor URI normalization without producing model.ubj/model.ubj
        old_aip = os.environ.get("AIP_STORAGE_URI")
        old_fn = os.environ.get("MODEL_FILENAME")
        try:
            os.environ["MODEL_FILENAME"] = "model.ubj"
            os.environ["AIP_STORAGE_URI"] = "gs://ml-models/run-1/model.ubj"
            self.assertEqual(resolve_model_source_from_env(), "gs://ml-models/run-1/model.ubj")
            os.environ["AIP_STORAGE_URI"] = "gs://ml-models/run-1"
            self.assertEqual(resolve_model_source_from_env(), "gs://ml-models/run-1/model.ubj")
        finally:
            if old_aip is None:
                os.environ.pop("AIP_STORAGE_URI", None)
            else:
                os.environ["AIP_STORAGE_URI"] = old_aip
            if old_fn is None:
                os.environ.pop("MODEL_FILENAME", None)
            else:
                os.environ["MODEL_FILENAME"] = old_fn

        parsed_file_uri = parse_deploy_config({
            "ML_OPS_PROJECT_ID": "ml-ops-proj",
            "REGION": "asia-southeast1",
            "VERTEX_ENDPOINT_ID": "1004200001",
            "CPR_CONTAINER_IMAGE": "asia-southeast1-docker.pkg.dev/ml-ops-proj/repo/cpr@sha256:" + "a" * 64,
            "CPR_MODEL_ARTIFACT_URI": "gs://ml-models/run-1/custom.ubj",
            "EXPECTED_MODEL_SHA256": "c" * 64,
            "FEATURE_NAMES": '["tenure_months"]',
        })
        self.assertEqual(parsed_file_uri["cpr_model_artifact_uri"], "gs://ml-models/run-1")
        self.assertEqual(parsed_file_uri["model_filename"], "custom.ubj")

        # 2. CPR /predict handles chunked HTTP request (no CONTENT_LENGTH) and rejects oversized payload
        booster = _Float32BoosterAdapter(margin_f32=_float32(0.30), contribs_f32=[_float32(0.20), _float32(0.10)])
        predictor = TreeShapCprPredictor(
            booster=booster,
            feature_names=["tenure_months"],
            model_id="credit_xgb_cpr",
            model_version_id="v1",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            dmatrix_factory=lambda rows, feature_names: rows,
        )
        cpr_app = create_cpr_wsgi_app(predictor)
        cpr_statuses: list[str] = []
        chunked_cpr_body = json.dumps({"instances": [{"tenure_months": 18}]}).encode("utf-8")
        cpr_resp = cpr_app(
            {
                "PATH_INFO": "/predict",
                "REQUEST_METHOD": "POST",
                "HTTP_TRANSFER_ENCODING": "chunked",
                "wsgi.input": io.BytesIO(chunked_cpr_body),
            },
            lambda s, _h: cpr_statuses.append(s),
        )
        self.assertEqual(cpr_statuses[0], "200 OK")
        self.assertEqual(len(json.loads(b"".join(cpr_resp).decode("utf-8"))["predictions"]), 1)

        over_statuses: list[str] = []
        cpr_app(
            {
                "PATH_INFO": "/predict",
                "REQUEST_METHOD": "POST",
                "CONTENT_LENGTH": str(cpr_predictor.MAX_REQUEST_BYTES + 1),
                "wsgi.input": io.BytesIO(b"{}"),
            },
            lambda s, _h: over_statuses.append(s),
        )
        self.assertEqual(over_statuses[0], "400 Bad Request")

        # 3. Scorer /score handles chunked HTTP request (no CONTENT_LENGTH)
        cfg = ScoringConfig(
            ml_ops_project_id="ml-ops-proj",
            region="asia-southeast1",
            vertex_endpoint_id="1004200001",
            ml_audit_dataset_id="ml_xai_audit",
        )
        write_client = MagicMock()
        write_client.append_rows.side_effect = lambda reqs: iter([MagicMock(row_errors=[], error=None)])
        writer = FailClosedLedgerWriter(project_id="ml-ops-proj", dataset_id="ml_xai_audit", write_client=write_client)
        pred_payload = {
            "entity_id": "ent-chunk",
            "model_id": "credit_xgb_cpr",
            "model_version_id": "v1",
            "model_artifact_sha256": "a" * 64,
            "container_image_digest": "sha256:" + "b" * 64,
            "schema_version": "v1",
            "explained_class": "APPROVE",
            "approve_prob": 0.65,
            "raw_margin": 0.619,
            "base_logit": 0.119,
            "shap_sum_logit": 0.50,
            "shap_logit_map": {"tenure_months": 0.50},
            "top_adverse_reasons": [],
            "feature_payload": {"tenure_months": 24},
            "reason_rule_version": REASON_RULE_VERSION,
        }
        scorer_app = create_scorer_wsgi_app(
            ScoringService(config=cfg, ledger_writer=writer, vertex_predict_fn=lambda _c, _inst: pred_payload)
        )
        sc_statuses: list[str] = []
        chunked_sc_body = json.dumps({"entity_id": "ent-chunk", "features": {"tenure_months": 24}}).encode("utf-8")
        sc_resp = scorer_app(
            {
                "PATH_INFO": "/score",
                "REQUEST_METHOD": "POST",
                "HTTP_TRANSFER_ENCODING": "chunked",
                "wsgi.input": io.BytesIO(chunked_sc_body),
            },
            lambda s, _h: sc_statuses.append(s),
        )
        self.assertEqual(sc_statuses[0], "200 OK")
        self.assertTrue(json.loads(b"".join(sc_resp).decode("utf-8"))["ledger_committed"])

    def test_deploy_cpr_cmek_version_id_and_undeploy_previous_models(self) -> None:
        kms_key = "projects/ml-ops-proj/locations/asia-southeast1/keyRings/ml/cryptoKeys/cpr"
        cfg = parse_deploy_config({
            "ML_OPS_PROJECT_ID": "ml-ops-proj",
            "REGION": "asia-southeast1",
            "VERTEX_ENDPOINT_ID": "1004200001",
            "CPR_CONTAINER_IMAGE": "asia-southeast1-docker.pkg.dev/ml-ops-proj/repo/cpr@sha256:" + "a" * 64,
            "CPR_MODEL_ARTIFACT_URI": "gs://ml-models/run-2/model.ubj",
            "EXPECTED_MODEL_SHA256": "d" * 64,
            "FEATURE_NAMES": '["tenure_months"]',
            "MODEL_DISPLAY_NAME": "ent-credit-cpr-treeshap",
            "MODEL_VERSION_ID": "git-abc1234",
            "KMS_KEY_ID": kms_key,
        })

        def _make_resp(payload: dict[str, object]) -> MagicMock:
            resp = MagicMock()
            resp.json.return_value = payload
            resp.raise_for_status.return_value = None
            return resp

        session = MagicMock()
        session.get.side_effect = [
            _make_resp({
                "deployedModels": [
                    {
                        "id": "old-dm-101",
                        "displayName": "ent-credit-cpr-treeshap",
                        "model": "projects/ml-ops-proj/locations/asia-southeast1/models/8001",
                    }
                ]
            }),
            _make_resp({
                "name": "projects/ml-ops-proj/locations/asia-southeast1/models/8001",
                "containerSpec": {"env": [{"name": "EXPECTED_MODEL_SHA256", "value": "e" * 64}]},
            }),
            _make_resp({
                "name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload",
                "done": True,
                "response": {"model": "projects/ml-ops-proj/locations/asia-southeast1/models/9002"},
            }),
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-deploy", "done": True}),
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-undeploy", "done": True}),
        ]
        session.post.side_effect = [
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-upload", "done": False}),
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-deploy", "done": False}),
            _make_resp({"name": "projects/ml-ops-proj/locations/asia-southeast1/operations/op-undeploy", "done": False}),
        ]

        deployed = deploy_cpr_model(cfg, session=session, sleep_fn=lambda _s: None, poll_interval_seconds=0.0)
        self.assertEqual(deployed, "projects/ml-ops-proj/locations/asia-southeast1/models/9002")
        self.assertEqual(session.post.call_count, 3)

        upload_payload = session.post.call_args_list[0][1]["json"]["model"]
        self.assertEqual(upload_payload["artifactUri"], "gs://ml-models/run-2")
        self.assertEqual(upload_payload["encryptionSpec"], {"kmsKeyName": kms_key})
        env_items = {e["name"]: e["value"] for e in upload_payload["containerSpec"]["env"]}
        self.assertEqual(env_items["MODEL_VERSION_ID"], "git-abc1234")

        undeploy_url = session.post.call_args_list[2][0][0]
        undeploy_payload = session.post.call_args_list[2][1]["json"]
        self.assertTrue(undeploy_url.endswith("/endpoints/1004200001:undeployModel"))
        self.assertEqual(undeploy_payload, {"deployedModelId": "old-dm-101"})

    def test_scaled_shap_tolerance_and_kfp_chronological_split_contract(self) -> None:
        # Large raw_margin (100.0) scales allowed drift to 1e-5 * 100.0 = 1e-3.
        # Drift of 5e-4 passes, while drift of 2e-3 fails closed.
        pass_booster = _Float32BoosterAdapter(margin_f32=100.0005, contribs_f32=[60.0, 40.0])
        pass_pred = TreeShapCprPredictor(
            booster=pass_booster,
            feature_names=["tenure_months"],
            model_id="credit_xgb_cpr",
            model_version_id="v1",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            additivity_tolerance=1e-5,
            dmatrix_factory=lambda rows, feature_names: rows,
        )
        res = pass_pred.predict_instance({"tenure_months": 12})
        self.assertAlmostEqual(res["raw_margin"], 100.0005, places=3)

        fail_booster = _Float32BoosterAdapter(margin_f32=100.002, contribs_f32=[60.0, 40.0])
        fail_pred = TreeShapCprPredictor(
            booster=fail_booster,
            feature_names=["tenure_months"],
            model_id="credit_xgb_cpr",
            model_version_id="v1",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            additivity_tolerance=1e-5,
            dmatrix_factory=lambda rows, feature_names: rows,
        )
        with self.assertRaisesRegex(TreeShapIntegrityError, "additivity breach"):
            fail_pred.predict_instance({"tenure_months": 12})

        # Verify KFP chronological out-of-time split & scaled SHAP contract in credit_scoring_kfp.py
        kfp_src = (REPO_ROOT / "src" / "pipelines" / "credit_scoring_kfp.py").read_text(encoding="utf-8")
        self.assertIn("ordered_df = df.assign(_split_ts=parsed_ts).sort_values", kfp_src)
        self.assertIn("allowed = shap_tolerance * max(1.0, abs(raw_margin))", kfp_src)

    def test_case_insensitive_nested_pii_and_non_mapping_json_rejected_400(self) -> None:
        cfg = ScoringConfig(
            ml_ops_project_id="ml-ops-proj",
            region="asia-southeast1",
            vertex_endpoint_id="1004200001",
            ml_audit_dataset_id="ml_xai_audit",
        )
        write_client = MagicMock()
        writer = FailClosedLedgerWriter(project_id="ml-ops-proj", dataset_id="ml_xai_audit", write_client=write_client)
        svc = ScoringService(config=cfg, ledger_writer=writer, vertex_predict_fn=lambda _c, _inst: {})
        scorer_app = create_scorer_wsgi_app(svc)

        for bad_key in ("NRIC", "Customer_IC_Number", " Passport_Number "):
            with self.subTest(scorer_pii=bad_key):
                status, body = svc.score_applicant({"entity_id": "ent-1", "features": {"tenure_months": 12, bad_key: "123"}})
                self.assertEqual(status, 400)
                self.assertIn("Restricted PII", body["error"])

                nested_status, _ = svc.score_applicant(
                    {"entity_id": "ent-1", "features": {"nested": {bad_key: "123"}}}
                )
                self.assertEqual(nested_status, 400)

                with self.assertRaisesRegex(ValueError, "Restricted PII"):
                    writer.build_ledger_row(
                        "ent-1",
                        "vertex_cpr_online",
                        {
                            "model_id": "m",
                            "model_version_id": "v",
                            "model_artifact_sha256": "a" * 64,
                            "container_image_digest": "d",
                            "approve_prob": 0.5,
                            "raw_margin": 0.0,
                            "base_logit": 0.0,
                            "shap_sum_logit": 0.0,
                            "shap_logit_map": {},
                            "top_adverse_reasons": [],
                            "feature_payload": {"nested": {bad_key: "123"}},
                            "reason_rule_version": REASON_RULE_VERSION,
                        },
                    )
        write_client.append_rows.assert_not_called()

        for non_obj in (b"[]", b"null", b'"hello"'):
            with self.subTest(scorer_body=non_obj):
                statuses: list[str] = []
                scorer_app(
                    {
                        "PATH_INFO": "/score",
                        "REQUEST_METHOD": "POST",
                        "CONTENT_LENGTH": str(len(non_obj)),
                        "wsgi.input": io.BytesIO(non_obj),
                    },
                    lambda s, _h: statuses.append(s),
                )
                self.assertEqual(statuses[0], "400 Bad Request")

        booster = _Float32BoosterAdapter(margin_f32=_float32(0.30), contribs_f32=[_float32(0.20), _float32(0.10)])
        predictor = TreeShapCprPredictor(
            booster=booster,
            feature_names=["tenure_months"],
            model_id="credit_xgb_cpr",
            model_version_id="v1",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            dmatrix_factory=lambda rows, feature_names: rows,
        )
        cpr_app = create_cpr_wsgi_app(predictor)
        for bad_cpr_body in (
            b"[]",
            b"null",
            b'{"instances": [null]}',
            b'{"instances": [[1]]}',
            json.dumps({"instances": [{"tenure_months": 12, "NRIC": "900101145566"}]}).encode("utf-8"),
            json.dumps({"instances": [{"tenure_months": 12, "nested": {"Customer_IC_Number": "x"}}]}).encode("utf-8"),
        ):
            with self.subTest(cpr_body=bad_cpr_body):
                cpr_statuses: list[str] = []
                cpr_app(
                    {
                        "PATH_INFO": "/predict",
                        "REQUEST_METHOD": "POST",
                        "CONTENT_LENGTH": str(len(bad_cpr_body)),
                        "wsgi.input": io.BytesIO(bad_cpr_body),
                    },
                    lambda s, _h: cpr_statuses.append(s),
                )
                self.assertEqual(cpr_statuses[0], "400 Bad Request")

    def test_cpr_non_approve_prob_inversion_transient_retry_and_lock_free_psc_fetch(self) -> None:
        # 1. approve_prob inverts when explained_class != APPROVE.
        booster = _Float32BoosterAdapter(margin_f32=_float32(1.0), contribs_f32=[_float32(0.6), _float32(0.4)])
        default_pred = TreeShapCprPredictor(
            booster=booster,
            feature_names=["tenure_months"],
            model_id="credit_xgb_cpr",
            model_version_id="v1",
            model_artifact_sha256="a" * 64,
            container_image_digest="sha256:" + "b" * 64,
            explained_class="DEFAULT",
            dmatrix_factory=lambda rows, feature_names: rows,
        )
        out = default_pred.predict_instance({"tenure_months": 12})
        self.assertLess(out["approve_prob"], 0.30)
        self.assertAlmostEqual(out["approve_prob"], 1.0 / (1.0 + pow(2.718281828459045, 1.0)), places=5)

        # 2. Transient GCS load failure allows subsequent retry on /health to succeed.
        env_keys = ("AIP_STORAGE_URI", "MODEL_PATH", "EXPECTED_MODEL_SHA256", "FEATURE_NAMES")
        saved = {k: os.environ.get(k) for k in env_keys}
        try:
            cpr_predictor.reset_lazy_state_for_tests()
            os.environ["AIP_STORAGE_URI"] = "gs://ml-models/run-1"
            os.environ["EXPECTED_MODEL_SHA256"] = "a" * 64
            os.environ["FEATURE_NAMES"] = '["tenure_months"]'
            ok_pred = TreeShapCprPredictor(
                booster=booster,
                feature_names=["tenure_months"],
                model_id="credit_xgb_cpr",
                model_version_id="v1",
                model_artifact_sha256="a" * 64,
                container_image_digest="sha256:" + "b" * 64,
                dmatrix_factory=lambda rows, feature_names: rows,
            )
            with patch.object(
                cpr_predictor.TreeShapCprPredictor,
                "load_from_file",
                side_effect=[ConnectionError("transient gcs 503"), ok_pred],
            ):
                s1: list[str] = []
                cpr_predictor.application({"PATH_INFO": "/health", "REQUEST_METHOD": "GET"}, lambda s, _h: s1.append(s))
                self.assertEqual(s1[0], "503 Service Unavailable")
                if cpr_predictor._BG_LOAD_THREAD is not None:
                    cpr_predictor._BG_LOAD_THREAD.join(timeout=2.0)

                s2: list[str] = []
                cpr_predictor.application({"PATH_INFO": "/health", "REQUEST_METHOD": "GET"}, lambda s, _h: s2.append(s))
                self.assertEqual(s2[0], "503 Service Unavailable")
                if cpr_predictor._BG_LOAD_THREAD is not None:
                    cpr_predictor._BG_LOAD_THREAD.join(timeout=2.0)

                s3: list[str] = []
                c3 = cpr_predictor.application({"PATH_INFO": "/health", "REQUEST_METHOD": "GET"}, lambda s, _h: s3.append(s))
                self.assertEqual(s3[0], "200 OK")
                self.assertTrue(json.loads(b"".join(c3).decode("utf-8"))["model_loaded"])
        finally:
            cpr_predictor.reset_lazy_state_for_tests()
            for k, val in saved.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val

        # 3. resolve_psc_endpoint_ip must not hold _PSC_DISCOVERY_LOCK while performing network calls.
        from src.scorer_service import app as scorer_app_mod

        invalidate_psc_cache()
        try:
            lock_acquired_in_fetcher: list[bool] = []

            def _fetcher(_cfg: ScoringConfig) -> dict[str, object]:
                acquired = scorer_app_mod._PSC_DISCOVERY_LOCK.acquire(blocking=False)
                lock_acquired_in_fetcher.append(acquired)
                if acquired:
                    scorer_app_mod._PSC_DISCOVERY_LOCK.release()
                return {
                    "privateServiceConnectConfig": {
                        "pscAutomationConfigs": [
                            {"state": "PSC_AUTOMATION_STATE_SUCCESSFUL", "ipAddress": "10.30.0.10"}
                        ]
                    }
                }

            cfg = ScoringConfig(
                ml_ops_project_id="ml-ops-proj",
                region="asia-southeast1",
                vertex_endpoint_id="1004200001",
                ml_audit_dataset_id="ml_xai_audit",
            )
            ip = scorer_app_mod.resolve_psc_endpoint_ip(cfg, endpoint_metadata_fetcher=_fetcher)
            self.assertEqual(ip, "10.30.0.10")
            self.assertEqual(lock_acquired_in_fetcher, [True])
        finally:
            invalidate_psc_cache()

        # 4. cpr_treeshap pyproject.toml declares direct runtime dependencies google-auth and requests.
        pyproject = (REPO_ROOT / "src" / "cpr_treeshap" / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"google-auth==2.62.2"', pyproject)
        self.assertIn('"requests==2.34.2"', pyproject)


if __name__ == "__main__":
    unittest.main()
