"""Vertex AI KFP v2 training, TreeSHAP parity, MRM gate, and Model Registry pipeline."""

import hashlib
from pathlib import Path
from typing import Any, List, NamedTuple, Sequence

from kfp import dsl  # type: ignore

DEFAULT_FEATURE_COLUMNS: List[str] = [
    "tenure_months",
    "avg_monthly_spend",
    "rolling_order_count",
    "days_since_last_order",
    "cancellation_rate_30d",
    "discount_utilization_ratio",
]


class MrmGateOutputs(NamedTuple):
    approved: bool
    summary_json: str


def compute_artifact_sha256(artifact_path: str | Path) -> str:
    """Return SHA-256 hex digest for a trained model file."""
    path = Path(artifact_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=[
        "google-cloud-bigquery>=3.25.0",
        "db-dtypes>=1.3.0",
        "pyarrow>=17.0.0",
        "pandas>=2.2.0",
    ],
)
def extract_pit_features(
    core_project_id: str,
    silver_dataset_id: str,
    feature_table: str,
    ml_ops_project_id: str,
    ml_audit_dataset_id: str,
    labels_table: str,
    feature_columns: List[str],
    output_dataset: dsl.Output[dsl.Dataset],
) -> str:
    """Materialize point-in-time training features and labels via ML.ENTITY_FEATURES_AT_TIME."""
    import re
    from google.cloud import bigquery  # type: ignore

    project_re = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
    ident_re = re.compile(r"^[A-Za-z0-9_]{1,1024}$")
    col_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

    if not project_re.fullmatch(core_project_id) or not project_re.fullmatch(ml_ops_project_id):
        raise ValueError("Invalid GCP project ID passed to extract_pit_features.")
    for name in (silver_dataset_id, feature_table, ml_audit_dataset_id, labels_table):
        if not ident_re.fullmatch(name):
            raise ValueError(f"Invalid BigQuery identifier: {name}")
    if not feature_columns or any(not col_re.fullmatch(c) for c in feature_columns):
        raise ValueError("Invalid feature_columns specification.")

    feature_select_cols = ", ".join(feature_columns)
    output_feature_cols = ", ".join(f"f.{c}" for c in feature_columns)
    feature_fqn = f"{core_project_id}.{silver_dataset_id}.{feature_table}"
    labels_fqn = f"{ml_ops_project_id}.{ml_audit_dataset_id}.{labels_table}"

    sql = f"""
    SELECT
      f.entity_id,
      {output_feature_cols},
      f.approved_flag,
      f.demographic_group
    FROM ML.ENTITY_FEATURES_AT_TIME(
      (
        SELECT
          entity_id,
          feature_timestamp,
          {feature_select_cols}
        FROM `{feature_fqn}`
        WHERE feature_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY)
      ),
      (
        SELECT
          entity_id,
          label_timestamp AS time,
          approved_flag,
          demographic_group
        FROM `{labels_fqn}`
        WHERE DATE(label_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY)
      ),
      num_rows => 1,
      ignore_feature_nulls => FALSE
    ) AS f
    """
    client = bigquery.Client(project=ml_ops_project_id)
    df = client.query(sql).to_dataframe()
    df.to_parquet(output_dataset.path, index=False)
    return str(len(df))


@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=[
        "pyarrow>=17.0.0",
        "pandas>=2.2.0",
        "xgboost>=2.1.0",
        "scikit-learn>=1.5.0",
    ],
)
def train_xgboost_challenger(
    dataset: dsl.Input[dsl.Dataset],
    feature_columns: List[str],
    model_artifact: dsl.Output[dsl.Model],
    eval_dataset: dsl.Output[dsl.Dataset],
) -> str:
    """Train deterministic single-thread XGBoost challenger and persist model.ubj + holdout split."""
    import hashlib
    import os
    import pandas as pd  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    import xgboost as xgb  # type: ignore

    os.environ["OMP_NUM_THREADS"] = "1"
    df = pd.read_parquet(dataset.path)
    cols = list(feature_columns)
    train_df, holdout_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["approved_flag"],
    )

    dtrain = xgb.DMatrix(train_df[cols], label=train_df["approved_flag"], feature_names=cols)
    booster = xgb.train(
        params={
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 4,
            "eta": 0.1,
            "nthread": 1,
            "seed": 42,
        },
        dtrain=dtrain,
        num_boost_round=50,
    )

    os.makedirs(model_artifact.path, exist_ok=True)
    model_file = os.path.join(model_artifact.path, "model.ubj")
    booster.save_model(model_file)
    with open(model_file, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()

    model_artifact.metadata["artifact_sha256"] = digest
    model_artifact.metadata["model_filename"] = "model.ubj"
    holdout_df.to_parquet(eval_dataset.path, index=False)
    return digest


@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=[
        "pyarrow>=17.0.0",
        "pandas>=2.2.0",
        "xgboost>=2.1.0",
        "scikit-learn>=1.5.0",
    ],
)
def evaluate_mrm_gate(
    eval_dataset: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    feature_columns: List[str],
    expected_sha256: str,
    min_auc: float,
    min_disparate_impact: float,
    max_brier: float,
    max_ece: float,
    shap_tolerance: float,
    model_card: dsl.Output[dsl.Artifact],
) -> MrmGateOutputs:
    """Evaluate holdout AUC, Brier, ECE, 80% disparate impact, and exact TreeSHAP additivity."""
    from collections import namedtuple
    import hashlib
    import hmac
    import json
    import os
    import pandas as pd  # type: ignore
    from sklearn.metrics import brier_score_loss, roc_auc_score  # type: ignore
    import xgboost as xgb  # type: ignore

    os.environ["OMP_NUM_THREADS"] = "1"
    model_file = (
        os.path.join(model_artifact.path, "model.ubj")
        if os.path.isdir(model_artifact.path)
        else model_artifact.path
    )
    with open(model_file, "rb") as handle:
        actual_sha = hashlib.sha256(handle.read()).hexdigest()
    if not hmac.compare_digest(actual_sha.lower(), expected_sha256.strip().lower()):
        raise RuntimeError(f"Model artifact SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}")

    booster = xgb.Booster()
    booster.load_model(model_file)

    df = pd.read_parquet(eval_dataset.path)
    cols = list(feature_columns)
    y_true = [int(v) for v in df["approved_flag"].tolist()]
    dtest = xgb.DMatrix(df[cols], feature_names=cols)

    probs = [float(v) for v in booster.predict(dtest)]
    raw_margins = [float(v) for v in booster.predict(dtest, output_margin=True)]
    contribs = booster.predict(dtest, pred_contribs=True, approx_contribs=False)

    auc_roc = float(roc_auc_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))

    # 10-bin Expected Calibration Error (ECE)
    ece = 0.0
    total_n = len(y_true)
    for b_idx in range(10):
        lo = b_idx / 10.0
        hi = (b_idx + 1) / 10.0
        if b_idx == 9:
            members = [i for i, p in enumerate(probs) if lo <= p <= hi]
        else:
            members = [i for i, p in enumerate(probs) if lo <= p < hi]
        if members:
            avg_conf = sum(probs[i] for i in members) / len(members)
            avg_acc = sum(y_true[i] for i in members) / len(members)
            ece += (len(members) / total_n) * abs(avg_acc - avg_conf)

    # 80% disparate impact across demographic_group
    preds = [1 if p >= 0.5 else 0 for p in probs]
    groups = [str(g) for g in df["demographic_group"].fillna("UNKNOWN").tolist()]
    grp_totals: dict[str, int] = {}
    grp_pos: dict[str, int] = {}
    for pred, grp in zip(preds, groups):
        grp_totals[grp] = grp_totals.get(grp, 0) + 1
        grp_pos[grp] = grp_pos.get(grp, 0) + pred
    pos_rates = {g: grp_pos[g] / grp_totals[g] for g in grp_totals if grp_totals[g] > 0}
    max_rate = max(pos_rates.values()) if pos_rates else 0.0
    min_rate = min(pos_rates.values()) if pos_rates else 0.0
    disparate_impact = (min_rate / max_rate) if max_rate > 0.0 else 0.0

    # Exact C++ TreeSHAP additivity check
    max_abs_diff = 0.0
    for row_idx, raw_margin in enumerate(raw_margins):
        contrib_row = [float(v) for v in contribs[row_idx]]
        shap_sum = sum(contrib_row[:-1])
        base_logit = contrib_row[-1]
        diff = abs((base_logit + shap_sum) - raw_margin)
        if diff > max_abs_diff:
            max_abs_diff = diff

    failed_checks: list[str] = []
    if auc_roc < min_auc:
        failed_checks.append(f"auc_roc:{auc_roc:.6f}<{min_auc:.6f}")
    if brier > max_brier:
        failed_checks.append(f"brier:{brier:.6f}>{max_brier:.6f}")
    if ece > max_ece:
        failed_checks.append(f"ece:{ece:.6f}>{max_ece:.6f}")
    if disparate_impact < min_disparate_impact:
        failed_checks.append(f"disparate_impact:{disparate_impact:.6f}<{min_disparate_impact:.6f}")
    if max_abs_diff > shap_tolerance:
        failed_checks.append(f"treeshap_additivity:{max_abs_diff:.8f}>{shap_tolerance:.8f}")

    approved = len(failed_checks) == 0
    card_payload = {
        "approved": approved,
        "artifact_sha256": actual_sha,
        "metrics": {
            "auc_roc": auc_roc,
            "brier_score": brier,
            "ece": ece,
            "disparate_impact_ratio": disparate_impact,
            "treeshap_max_abs_diff": max_abs_diff,
        },
        "thresholds": {
            "min_auc": min_auc,
            "max_brier": max_brier,
            "max_ece": max_ece,
            "min_disparate_impact": min_disparate_impact,
            "shap_tolerance": shap_tolerance,
        },
        "subgroup_positive_rates": pos_rates,
        "failed_checks": failed_checks,
    }
    summary_json = json.dumps(card_payload, indent=2, sort_keys=True)
    with open(model_card.path, "w", encoding="utf-8") as handle:
        handle.write(summary_json + "\n")

    if not approved:
        raise RuntimeError(f"MRM gate rejected challenger model: {'; '.join(failed_checks)}")

    output_cls = namedtuple("MrmGateOutputs", ["approved", "summary_json"])
    return output_cls(approved=approved, summary_json=summary_json)


@dsl.component(
    base_image="python:3.13-slim",
    packages_to_install=[
        "google-cloud-aiplatform>=1.70.0",
    ],
)
def register_cpr_model(
    ml_ops_project_id: str,
    region: str,
    model_display_name: str,
    cpr_serving_image: str,
    model_artifact: dsl.Input[dsl.Model],
    feature_columns: List[str],
    artifact_sha256: str,
    mrm_approved: bool,
    reason_rule_version: str = "bnm-fair-lending-v1.0",
    vertex_endpoint_id: str = "",
    deploy_to_endpoint: bool = False,
) -> str:
    """Upload SHA-256 verified CPR model to Vertex AI Model Registry and optionally deploy."""
    import json
    from google.cloud import aiplatform  # type: ignore

    if not mrm_approved:
        raise RuntimeError("MRM gate rejected challenger model; Vertex AI registration blocked.")

    aiplatform.init(project=ml_ops_project_id, location=region)
    uploaded = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact.uri,
        serving_container_image_uri=cpr_serving_image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_environment_variables={
            "MODEL_FILENAME": "model.ubj",
            "EXPECTED_MODEL_SHA256": artifact_sha256,
            "FEATURE_NAMES": json.dumps(list(feature_columns)),
            "REASON_RULE_VERSION": reason_rule_version,
        },
    )
    if deploy_to_endpoint and vertex_endpoint_id:
        endpoint_name = (
            vertex_endpoint_id
            if vertex_endpoint_id.startswith("projects/")
            else f"projects/{ml_ops_project_id}/locations/{region}/endpoints/{vertex_endpoint_id}"
        )
        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        uploaded.deploy(
            endpoint=endpoint,
            deployed_model_display_name=model_display_name,
            machine_type="n2-standard-4",
            min_replica_count=1,
            max_replica_count=3,
            traffic_percentage=100,
        )
    return uploaded.resource_name


@dsl.pipeline(name="credit-scoring-xai-pipeline")
def credit_scoring_xai_pipeline(
    core_project_id: str,
    silver_dataset_id: str,
    feature_table: str,
    ml_ops_project_id: str,
    ml_audit_dataset_id: str,
    cpr_serving_image: str,
    region: str = "asia-southeast1",
    labels_table: str = "ml_training_labels",
    feature_columns: List[str] = DEFAULT_FEATURE_COLUMNS,
    min_auc: float = 0.75,
    min_disparate_impact: float = 0.80,
    max_brier: float = 0.25,
    max_ece: float = 0.10,
    shap_tolerance: float = 1e-5,
    reason_rule_version: str = "bnm-fair-lending-v1.0",
    vertex_endpoint_id: str = "",
    deploy_to_endpoint: bool = False,
) -> None:
    """Kubeflow Pipelines v2 DAG for Track B CPR training, MRM gating, and registration."""
    feat_task = extract_pit_features(
        core_project_id=core_project_id,
        silver_dataset_id=silver_dataset_id,
        feature_table=feature_table,
        ml_ops_project_id=ml_ops_project_id,
        ml_audit_dataset_id=ml_audit_dataset_id,
        labels_table=labels_table,
        feature_columns=feature_columns,
    )
    train_task = train_xgboost_challenger(
        dataset=feat_task.outputs["output_dataset"],
        feature_columns=feature_columns,
    )
    gate_task = evaluate_mrm_gate(
        eval_dataset=train_task.outputs["eval_dataset"],
        model_artifact=train_task.outputs["model_artifact"],
        feature_columns=feature_columns,
        expected_sha256=train_task.outputs["Output"],
        min_auc=min_auc,
        min_disparate_impact=min_disparate_impact,
        max_brier=max_brier,
        max_ece=max_ece,
        shap_tolerance=shap_tolerance,
    )
    with dsl.If(gate_task.outputs["approved"] == True):  # noqa: E712
        register_cpr_model(
            ml_ops_project_id=ml_ops_project_id,
            region=region,
            model_display_name="credit-scoring-cpr-treeshap",
            cpr_serving_image=cpr_serving_image,
            model_artifact=train_task.outputs["model_artifact"],
            feature_columns=feature_columns,
            artifact_sha256=train_task.outputs["Output"],
            mrm_approved=gate_task.outputs["approved"],
            reason_rule_version=reason_rule_version,
            vertex_endpoint_id=vertex_endpoint_id,
            deploy_to_endpoint=deploy_to_endpoint,
        )


def build_credit_scoring_pipeline(
    pipeline_name: str = "credit-scoring-xai-pipeline",
    feature_columns: Sequence[str] = tuple(DEFAULT_FEATURE_COLUMNS),
) -> Any:
    """Return the compiled KFP v2 pipeline definition."""
    _ = pipeline_name
    _ = feature_columns
    return credit_scoring_xai_pipeline
