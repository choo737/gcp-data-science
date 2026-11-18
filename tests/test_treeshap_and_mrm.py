"""Hermetic unit tests for CPR TreeSHAP, fail-closed ledger writer, and MRM gate."""

from __future__ import annotations

import array
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cpr_treeshap.predictor import (
    REASON_RULE_VERSION,
    TreeShapCprPredictor,
    create_cpr_wsgi_app,
    rank_adverse_features,
)
from src.mrm_validation.evaluate_gate import (
    compute_ks_statistic,
    compute_roc_auc,
    compute_subgroup_fairness,
    evaluate_challenger_gate,
    main as mrm_main,
)
from src.scorer_service.app import (
    ScoringConfig,
    ScoringService,
    create_scorer_wsgi_app,
    format_adverse_reasons_for_agent,
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
        with self.assertRaisesRegex(ValueError, "additivity breach"):
            predictor.predict_instance({"tenure_months": 12, "delinquency_count_12m": 1})

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
        bad_write_client.create_write_stream.return_value = MagicMock(name="stream-1")
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

        # Case 2: Storage Write raises exception -> WSGI app returns 503 Service Unavailable.
        exc_write_client = MagicMock()
        exc_write_client.create_write_stream.side_effect = RuntimeError("bq storage unavailable")
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
        stream_obj = MagicMock()
        stream_obj.name = "projects/ml-ops-proj/datasets/ml_xai_audit/tables/ml_xai_audit_ledger/streams/s1"
        ok_write_client.create_write_stream.return_value = stream_obj
        ok_write_client.append_rows.return_value = iter([MagicMock(row_errors=[], error=None)])

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
        biased_y = [1, 1, 1, 1, 0, 0, 0, 1]
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


if __name__ == "__main__":
    unittest.main()
