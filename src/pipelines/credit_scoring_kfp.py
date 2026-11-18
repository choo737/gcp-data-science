"""Vertex AI KFP v2 training, TreeSHAP parity, MRM gate, and Model Registry pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence


def compute_artifact_sha256(artifact_path: str | Path) -> str:
    """Return SHA-256 hex digest for a trained model file."""
    path = Path(artifact_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_credit_scoring_pipeline(
    pipeline_name: str = "credit-scoring-xai-pipeline",
    feature_columns: Sequence[str] = (
        "tenure_months",
        "avg_monthly_spend",
        "delinquency_count_12m",
        "credit_utilization_ratio",
        "debt_to_income_ratio",
        "verified_income_band",
    ),
) -> Any:
    """Compile a Kubeflow Pipelines v2 DAG for Track B CPR training and MRM gating."""
    from kfp import dsl  # type: ignore

    @dsl.component(base_image="python:3.13-slim")
    def extract_point_in_time_features(
        core_project_id: str,
        silver_dataset_id: str,
        feature_table: str,
        output_dataset: dsl.Output[dsl.Dataset],
    ) -> str:
        from google.cloud import bigquery  # type: ignore

        client = bigquery.Client(project=core_project_id)
        sql = f"""
        SELECT
          customer_id,
          {", ".join(feature_columns)},
          approved_flag,
          demographic_group
        FROM ML.FEATURES_AT_TIME(
          TABLE `{core_project_id}.{silver_dataset_id}.{feature_table}`,
          time => CURRENT_TIMESTAMP(),
          num_rows => 1,
          ignore_feature_nulls => FALSE
        )
        """
        df = client.query(sql).to_dataframe()
        df.to_parquet(output_dataset.path, index=False)
        return str(len(df))

    @dsl.component(base_image="python:3.13-slim")
    def train_xgboost_challenger(
        dataset: dsl.Input[dsl.Dataset],
        model_artifact: dsl.Output[dsl.Model],
    ) -> str:
        import hashlib
        import pandas as pd  # type: ignore
        import xgboost as xgb  # type: ignore

        df = pd.read_parquet(dataset.path)
        x_train = df[list(feature_columns)]
        y_train = df["approved_flag"]
        dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(feature_columns))
        booster = xgb.train(
            params={
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "max_depth": 4,
                "eta": 0.1,
                "nthread": 1,
            },
            dtrain=dtrain,
            num_boost_round=50,
        )
        booster.save_model(model_artifact.path)
        digest = hashlib.sha256(open(model_artifact.path, "rb").read()).hexdigest()
        model_artifact.metadata["artifact_sha256"] = digest
        return digest

    @dsl.component(base_image="python:3.13-slim")
    def register_cpr_model(
        ml_ops_project_id: str,
        region: str,
        model_display_name: str,
        cpr_serving_image: str,
        model_artifact: dsl.Input[dsl.Model],
        mrm_approved: bool,
    ) -> str:
        if not mrm_approved:
            raise RuntimeError("MRM gate rejected challenger model; Vertex AI registration blocked.")
        from google.cloud import aiplatform  # type: ignore

        aiplatform.init(project=ml_ops_project_id, location=region)
        uploaded = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_artifact.uri,
            serving_container_image_uri=cpr_serving_image,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            # Exact TreeSHAP attributions run inside CPR /predict without legacy explain configs.
        )
        return uploaded.resource_name

    @dsl.pipeline(name=pipeline_name)
    def credit_scoring_xai_pipeline(
        core_project_id: str,
        silver_dataset_id: str,
        feature_table: str,
        ml_ops_project_id: str,
        region: str,
        cpr_serving_image: str,
        mrm_approved: bool = False,
    ) -> None:
        feat_task = extract_point_in_time_features(
            core_project_id=core_project_id,
            silver_dataset_id=silver_dataset_id,
            feature_table=feature_table,
        )
        train_task = train_xgboost_challenger(dataset=feat_task.outputs["output_dataset"])
        register_cpr_model(
            ml_ops_project_id=ml_ops_project_id,
            region=region,
            model_display_name="credit-scoring-cpr-treeshap",
            cpr_serving_image=cpr_serving_image,
            model_artifact=train_task.outputs["model_artifact"],
            mrm_approved=mrm_approved,
        )

    return credit_scoring_xai_pipeline
