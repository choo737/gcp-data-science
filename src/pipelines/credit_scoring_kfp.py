"""Agent Platform (formerly Vertex AI) KFP v2 training, TreeSHAP parity, MRM gate, and Model Registry pipeline."""

import hashlib
from pathlib import Path
from typing import Any, List, NamedTuple, Sequence

from kfp import dsl  # type: ignore

PINNED_PYTHON_BASE_IMAGE = (
    "python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90"
)

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
    base_image=PINNED_PYTHON_BASE_IMAGE,
    packages_to_install=[
        "google-cloud-bigquery==3.25.0",
        "db-dtypes==1.3.0",
        "pyarrow==26.0.1",
        "pandas==2.2.3",
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
      f.time AS decision_timestamp,
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
        WHERE label_timestamp >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY))
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
    base_image=PINNED_PYTHON_BASE_IMAGE,
    packages_to_install=[
        "pyarrow==26.0.1",
        "pandas==2.2.3",
        "xgboost==2.1.4",
        "scikit-learn==1.5.2",
    ],
)
def train_xgboost_challenger(
    dataset: dsl.Input[dsl.Dataset],
    feature_columns: List[str],
    model_artifact: dsl.Output[dsl.Model],
    eval_dataset: dsl.Output[dsl.Dataset],
) -> str:
    """Train deterministic single-thread XGBoost challenger with chronological out-of-time split."""
    import hashlib
    import os
    import pandas as pd  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    import xgboost as xgb  # type: ignore

    os.environ["OMP_NUM_THREADS"] = "1"
    df = pd.read_parquet(dataset.path)
    cols = list(feature_columns)

    # Chronological out-of-time split (80% earliest train, 20% latest holdout) prevents rolling-window leakage.
    ts_col = next(
        (c for c in ("decision_timestamp", "feature_timestamp", "label_timestamp", "time") if c in df.columns),
        None,
    )
    use_oot_split = False
    if ts_col is not None:
        parsed_ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if parsed_ts.notna().all() and parsed_ts.nunique() > 1:
            sort_cols = ["_split_ts", "entity_id"] if "entity_id" in df.columns else ["_split_ts"]
            ordered_df = df.assign(_split_ts=parsed_ts).sort_values(by=sort_cols, kind="mergesort").drop(
                columns=["_split_ts"]
            )
            split_idx = max(1, min(len(ordered_df) - 1, int(len(ordered_df) * 0.8)))
            cand_train = ordered_df.iloc[:split_idx].reset_index(drop=True)
            cand_holdout = ordered_df.iloc[split_idx:].reset_index(drop=True)
            if cand_train["approved_flag"].nunique() >= 2 and cand_holdout["approved_flag"].nunique() >= 2:
                train_df, holdout_df = cand_train, cand_holdout
                use_oot_split = True

    if not use_oot_split:
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
    base_image=PINNED_PYTHON_BASE_IMAGE,
    packages_to_install=[
        "pyarrow==26.0.1",
        "pandas==2.2.3",
        "xgboost==2.1.4",
        "scikit-learn==1.5.2",
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
    min_ks: float = 0.30,
    max_fpr_gap: float = 0.05,
) -> MrmGateOutputs:
    """Evaluate holdout AUC, KS, Brier, ECE, subgroup fairness, and scaled TreeSHAP additivity."""
    from collections import namedtuple
    import hashlib
    import hmac
    import json
    from operator import itemgetter
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

    # Exact two-sample Kolmogorov-Smirnov statistic (overall holdout)
    n_pos_all = sum(y_true)
    n_neg_all = len(y_true) - n_pos_all
    if n_pos_all == 0 or n_neg_all == 0:
        raise RuntimeError("Holdout set must contain both positive and negative examples for KS calculation.")
    paired_all = sorted(zip(probs, y_true), key=itemgetter(0))
    cum_pos = 0
    cum_neg = 0
    ks_stat = 0.0
    p_idx = 0
    while p_idx < len(paired_all):
        thresh = paired_all[p_idx][0]
        while p_idx < len(paired_all) and paired_all[p_idx][0] == thresh:
            if paired_all[p_idx][1] == 1:
                cum_pos += 1
            else:
                cum_neg += 1
            p_idx += 1
        ks_diff = abs((cum_pos / n_pos_all) - (cum_neg / n_neg_all))
        if ks_diff > ks_stat:
            ks_stat = ks_diff

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

    # Subgroup disparate impact, false-positive-rate gap, and subgroup KS across demographic_group
    preds = [1 if p >= 0.5 else 0 for p in probs]
    raw_groups = df["demographic_group"].tolist()
    groups = [str(g).strip() if g is not None and pd.notna(g) else "" for g in raw_groups]
    grp_totals: dict[str, int] = {}
    grp_pred_pos: dict[str, int] = {}
    grp_pos_total: dict[str, int] = {}
    grp_neg_total: dict[str, int] = {}
    grp_false_pos: dict[str, int] = {}
    grp_pairs: dict[str, list[tuple[float, int]]] = {}
    for prob, pred, yt, grp in zip(probs, preds, y_true, groups):
        if not grp:
            continue
        grp_totals[grp] = grp_totals.get(grp, 0) + 1
        grp_pred_pos[grp] = grp_pred_pos.get(grp, 0) + pred
        if yt == 1:
            grp_pos_total[grp] = grp_pos_total.get(grp, 0) + 1
        else:
            grp_neg_total[grp] = grp_neg_total.get(grp, 0) + 1
            if pred == 1:
                grp_false_pos[grp] = grp_false_pos.get(grp, 0) + 1
        grp_pairs.setdefault(grp, []).append((prob, yt))
    if len(grp_totals) < 2:
        raise RuntimeError("MRM fairness gate requires at least 2 demographic subgroups in holdout data.")

    pos_rates: dict[str, float] = {}
    fpr_rates: dict[str, float] = {}
    subgroup_ks: dict[str, float] = {}
    for g in sorted(grp_totals):
        n_g_pos = grp_pos_total.get(g, 0)
        n_g_neg = grp_neg_total.get(g, 0)
        if n_g_pos == 0 or n_g_neg == 0:
            raise RuntimeError(
                f"Subgroup '{g}' must contain both positive and negative examples (pos={n_g_pos}, neg={n_g_neg})."
            )
        pos_rates[g] = grp_pred_pos[g] / grp_totals[g]
        fpr_rates[g] = grp_false_pos.get(g, 0) / n_g_neg
        g_sorted = sorted(grp_pairs[g], key=itemgetter(0))
        g_cpos = 0
        g_cneg = 0
        g_ks = 0.0
        g_idx = 0
        while g_idx < len(g_sorted):
            g_thresh = g_sorted[g_idx][0]
            while g_idx < len(g_sorted) and g_sorted[g_idx][0] == g_thresh:
                if g_sorted[g_idx][1] == 1:
                    g_cpos += 1
                else:
                    g_cneg += 1
                g_idx += 1
            g_diff = abs((g_cpos / n_g_pos) - (g_cneg / n_g_neg))
            if g_diff > g_ks:
                g_ks = g_diff
        subgroup_ks[g] = float(g_ks)

    max_rate = max(pos_rates.values())
    min_rate = min(pos_rates.values())
    disparate_impact = (min_rate / max_rate) if max_rate > 0.0 else 0.0
    observed_fpr_gap = max(fpr_rates.values()) - min(fpr_rates.values())

    # Exact C++ TreeSHAP additivity check scaled by max(1.0, abs(raw_margin))
    max_abs_diff = 0.0
    max_allowed_tol = shap_tolerance
    shap_breach = False
    for row_idx, raw_margin in enumerate(raw_margins):
        contrib_row = [float(v) for v in contribs[row_idx]]
        shap_sum = sum(contrib_row[:-1])
        base_logit = contrib_row[-1]
        diff = abs((base_logit + shap_sum) - raw_margin)
        allowed = shap_tolerance * max(1.0, abs(raw_margin))
        if diff > max_abs_diff:
            max_abs_diff = diff
        if diff > allowed:
            shap_breach = True
            if allowed > max_allowed_tol:
                max_allowed_tol = allowed

    failed_checks: list[str] = []
    if auc_roc < min_auc:
        failed_checks.append(f"auc_roc:{auc_roc:.6f}<{min_auc:.6f}")
    if ks_stat < min_ks:
        failed_checks.append(f"ks_stat:{ks_stat:.6f}<{min_ks:.6f}")
    if brier > max_brier:
        failed_checks.append(f"brier:{brier:.6f}>{max_brier:.6f}")
    if ece > max_ece:
        failed_checks.append(f"ece:{ece:.6f}>{max_ece:.6f}")
    if disparate_impact < min_disparate_impact:
        failed_checks.append(f"disparate_impact:{disparate_impact:.6f}<{min_disparate_impact:.6f}")
    if observed_fpr_gap > max_fpr_gap:
        failed_checks.append(f"max_fpr_gap:{observed_fpr_gap:.6f}>{max_fpr_gap:.6f}")
    if shap_breach:
        failed_checks.append(f"treeshap_additivity:{max_abs_diff:.8f}>{max_allowed_tol:.8f}")

    approved = len(failed_checks) == 0
    card_payload = {
        "approved": approved,
        "artifact_sha256": actual_sha,
        "metrics": {
            "auc_roc": auc_roc,
            "ks_statistic": ks_stat,
            "brier_score": brier,
            "ece": ece,
            "disparate_impact_ratio": disparate_impact,
            "max_fpr_gap": observed_fpr_gap,
            "treeshap_max_abs_diff": max_abs_diff,
        },
        "thresholds": {
            "min_auc": min_auc,
            "min_ks": min_ks,
            "max_brier": max_brier,
            "max_ece": max_ece,
            "min_disparate_impact": min_disparate_impact,
            "max_fpr_gap": max_fpr_gap,
            "shap_tolerance": shap_tolerance,
        },
        "subgroup_positive_rates": pos_rates,
        "subgroup_false_positive_rates": fpr_rates,
        "subgroup_ks": subgroup_ks,
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
    base_image=PINNED_PYTHON_BASE_IMAGE,
    packages_to_install=[
        "google-cloud-aiplatform==1.70.0",
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
    shap_tolerance: float = 1e-5,
    vertex_endpoint_id: str = "",
    deploy_to_endpoint: bool = False,
    kms_key_name: str = "",
) -> str:
    """Upload SHA-256 verified CPR model to Agent Platform Model Registry and optionally deploy."""
    import json
    from google.cloud import aiplatform  # type: ignore

    if not mrm_approved:
        raise RuntimeError("MRM gate rejected challenger model; Agent Platform registration blocked.")

    aiplatform.init(project=ml_ops_project_id, location=region)
    cleaned_kms_key = kms_key_name.strip() or None
    uploaded = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact.uri,
        serving_container_image_uri=cpr_serving_image,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        encryption_spec_key_name=cleaned_kms_key,
        serving_container_environment_variables={
            "MODEL_FILENAME": "model.ubj",
            "EXPECTED_MODEL_SHA256": artifact_sha256,
            "FEATURE_NAMES": json.dumps(list(feature_columns)),
            "REASON_RULE_VERSION": reason_rule_version,
            "CONTAINER_IMAGE_DIGEST": cpr_serving_image,
            "MODEL_ID": model_display_name,
            "MODEL_VERSION_ID": artifact_sha256[:12],
            "SHAP_ADDITIVITY_TOLERANCE": str(shap_tolerance),
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
    ml_ops_project_id: str,
    ml_audit_dataset_id: str,
    cpr_serving_image: str,
    feature_table: str = "vw_ml_customer_features_online",
    region: str = "asia-southeast1",
    labels_table: str = "ml_training_labels",
    feature_columns: List[str] = DEFAULT_FEATURE_COLUMNS,
    min_auc: float = 0.75,
    min_ks: float = 0.30,
    min_disparate_impact: float = 0.80,
    max_fpr_gap: float = 0.05,
    max_brier: float = 0.25,
    max_ece: float = 0.10,
    shap_tolerance: float = 1e-5,
    reason_rule_version: str = "bnm-fair-lending-v1.0",
    vertex_endpoint_id: str = "",
    deploy_to_endpoint: bool = False,
    kms_key_name: str = "",
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
        min_ks=min_ks,
        min_disparate_impact=min_disparate_impact,
        max_fpr_gap=max_fpr_gap,
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
            shap_tolerance=shap_tolerance,
            vertex_endpoint_id=vertex_endpoint_id,
            deploy_to_endpoint=deploy_to_endpoint,
            kms_key_name=kms_key_name,
        )


def build_credit_scoring_pipeline(
    pipeline_name: str = "credit-scoring-xai-pipeline",
    feature_columns: Sequence[str] = tuple(DEFAULT_FEATURE_COLUMNS),
) -> Any:
    """Return a KFP v2 pipeline definition configured with the requested name and feature list."""
    default_features = list(feature_columns)
    if pipeline_name == "credit-scoring-xai-pipeline" and default_features == DEFAULT_FEATURE_COLUMNS:
        return credit_scoring_xai_pipeline

    @dsl.pipeline(name=pipeline_name)
    def custom_credit_scoring_pipeline(
        core_project_id: str,
        silver_dataset_id: str,
        ml_ops_project_id: str,
        ml_audit_dataset_id: str,
        cpr_serving_image: str,
        feature_table: str = "vw_ml_customer_features_online",
        region: str = "asia-southeast1",
        labels_table: str = "ml_training_labels",
        feature_columns: List[str] = default_features,
        min_auc: float = 0.75,
        min_ks: float = 0.30,
        min_disparate_impact: float = 0.80,
        max_fpr_gap: float = 0.05,
        max_brier: float = 0.25,
        max_ece: float = 0.10,
        shap_tolerance: float = 1e-5,
        reason_rule_version: str = "bnm-fair-lending-v1.0",
        vertex_endpoint_id: str = "",
        deploy_to_endpoint: bool = False,
        kms_key_name: str = "",
    ) -> None:
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
            min_ks=min_ks,
            min_disparate_impact=min_disparate_impact,
            max_fpr_gap=max_fpr_gap,
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
                shap_tolerance=shap_tolerance,
                vertex_endpoint_id=vertex_endpoint_id,
                deploy_to_endpoint=deploy_to_endpoint,
                kms_key_name=kms_key_name,
            )

    return custom_credit_scoring_pipeline
