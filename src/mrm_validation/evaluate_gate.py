"""Mathematical MRM evaluation gate: ROC-AUC, KS, fairness, and TreeSHAP parity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _validate_binary_inputs(y_true: Sequence[int], scores: Sequence[float]) -> tuple[list[int], list[float]]:
    if len(y_true) != len(scores) or not y_true:
        raise ValueError("y_true and scores must be non-empty sequences of equal length.")
    labels = [int(y) for y in y_true]
    if any(y not in (0, 1) for y in labels):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive (1) and negative (0) examples.")
    return labels, [float(s) for s in scores]


def compute_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute exact tie-aware ROC-AUC using average-rank Mann-Whitney U."""
    labels, scores = _validate_binary_inputs(y_true, y_score)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for pos in range(idx, end):
            ranks[indexed[pos][0]] = avg_rank
        idx = end

    pos_rank_sum = sum(rank for label, rank in zip(labels, ranks) if label == 1)
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


def compute_ks_statistic(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute exact two-sample Kolmogorov-Smirnov separation between positives and negatives."""
    labels, scores = _validate_binary_inputs(y_true, y_score)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    paired = sorted(zip(scores, labels), key=lambda pair: pair[0])
    cum_pos = 0
    cum_neg = 0
    max_ks = 0.0
    idx = 0
    total = len(paired)

    while idx < total:
        threshold = paired[idx][0]
        while idx < total and paired[idx][0] == threshold:
            if paired[idx][1] == 1:
                cum_pos += 1
            else:
                cum_neg += 1
            idx += 1
        cdf_pos = cum_pos / n_pos
        cdf_neg = cum_neg / n_neg
        diff = abs(cdf_pos - cdf_neg)
        if diff > max_ks:
            max_ks = diff

    return float(max_ks)


def compute_subgroup_fairness(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: Sequence[str],
) -> dict[str, Any]:
    """Compute subgroup approval rates, false-positive rates, disparate impact, and FPR gap."""
    if not (len(y_true) == len(y_pred) == len(groups)) or not y_true:
        raise ValueError("y_true, y_pred, and groups must be non-empty sequences of equal length.")

    group_stats: dict[str, dict[str, int]] = {}
    for yt_raw, yp_raw, grp_raw in zip(y_true, y_pred, groups):
        yt = int(yt_raw)
        yp = int(yp_raw)
        grp = str(grp_raw)
        if yt not in (0, 1) or yp not in (0, 1):
            raise ValueError("y_true and y_pred must contain only 0 and 1.")
        bucket = group_stats.setdefault(
            grp,
            {"total": 0, "pred_pos": 0, "pos_total": 0, "neg_total": 0, "false_pos": 0},
        )
        bucket["total"] += 1
        if yp == 1:
            bucket["pred_pos"] += 1
        if yt == 1:
            bucket["pos_total"] += 1
        else:
            bucket["neg_total"] += 1
            if yp == 1:
                bucket["false_pos"] += 1

    if len(group_stats) < 2:
        raise ValueError("At least two subgroups are required for fairness evaluation.")

    subgroup_metrics: dict[str, dict[str, float]] = {}
    positive_rates: list[float] = []
    fpr_values: list[float] = []

    for grp in sorted(group_stats):
        stats = group_stats[grp]
        if stats["neg_total"] == 0 or stats["pos_total"] == 0:
            raise ValueError(
                f"Subgroup '{grp}' must contain both positive (1) and negative (0) examples "
                f"(pos_total={stats['pos_total']}, neg_total={stats['neg_total']})."
            )
        pos_rate = stats["pred_pos"] / stats["total"]
        fpr = stats["false_pos"] / stats["neg_total"]
        subgroup_metrics[grp] = {
            "count": float(stats["total"]),
            "positive_rate": float(pos_rate),
            "false_positive_rate": float(fpr),
        }
        positive_rates.append(pos_rate)
        fpr_values.append(fpr)

    max_pos_rate = max(positive_rates)
    min_pos_rate = min(positive_rates)
    disparate_impact = (min_pos_rate / max_pos_rate) if max_pos_rate > 0.0 else 0.0
    max_fpr_gap = max(fpr_values) - min(fpr_values)

    return {
        "subgroups": subgroup_metrics,
        "disparate_impact_ratio": float(disparate_impact),
        "max_fpr_gap": float(max_fpr_gap),
    }


def compute_shap_parity_max_diff(
    treeshap_logits: Sequence[float],
    bqml_logits: Sequence[float],
) -> float:
    """Compute maximum absolute log-odds difference between TreeSHAP and BQML attributions."""
    if len(treeshap_logits) != len(bqml_logits) or not treeshap_logits:
        raise ValueError("treeshap_logits and bqml_logits must be non-empty sequences of equal length.")
    return float(max(abs(float(a) - float(b)) for a, b in zip(treeshap_logits, bqml_logits)))


def evaluate_challenger_gate(
    y_true: Sequence[int],
    challenger_scores: Sequence[float],
    incumbent_scores: Sequence[float],
    groups: Sequence[str],
    treeshap_logits: Sequence[float],
    bqml_logits: Sequence[float],
    model_id: str = "credit_xgb_cpr",
    model_version_id: str = "v1",
    model_artifact_sha256: str = "",
    decision_threshold: float = 0.5,
    min_auc: float = 0.75,
    min_ks: float = 0.30,
    min_disparate_impact: float = 0.80,
    max_fpr_gap: float = 0.05,
    shap_parity_tolerance: float = 1e-5,
    max_auc_regression: float = 0.005,
) -> dict[str, Any]:
    """Run all MRM challenger vs incumbent gates and return a structured Model Card."""
    challenger_auc = compute_roc_auc(y_true, challenger_scores)
    incumbent_auc = compute_roc_auc(y_true, incumbent_scores)
    challenger_ks = compute_ks_statistic(y_true, challenger_scores)
    incumbent_ks = compute_ks_statistic(y_true, incumbent_scores)

    challenger_preds = [1 if float(s) >= decision_threshold else 0 for s in challenger_scores]
    fairness = compute_subgroup_fairness(y_true, challenger_preds, groups)
    shap_max_diff = compute_shap_parity_max_diff(treeshap_logits, bqml_logits)

    failed_checks: list[str] = []
    if challenger_auc < min_auc:
        failed_checks.append(f"roc_auc_below_floor:{challenger_auc:.6f}<{min_auc:.6f}")
    if challenger_auc < (incumbent_auc - max_auc_regression):
        failed_checks.append(
            f"roc_auc_regression:{challenger_auc:.6f}<{incumbent_auc - max_auc_regression:.6f}"
        )
    if challenger_ks < min_ks:
        failed_checks.append(f"ks_below_floor:{challenger_ks:.6f}<{min_ks:.6f}")
    if fairness["disparate_impact_ratio"] < min_disparate_impact:
        failed_checks.append(
            f"disparate_impact_below_min:{fairness['disparate_impact_ratio']:.6f}<{min_disparate_impact:.6f}"
        )
    if fairness["max_fpr_gap"] > max_fpr_gap:
        failed_checks.append(f"fpr_gap_above_max:{fairness['max_fpr_gap']:.6f}>{max_fpr_gap:.6f}")
    if shap_max_diff > shap_parity_tolerance:
        failed_checks.append(f"shap_parity_breach:{shap_max_diff:.8f}>{shap_parity_tolerance:.8f}")

    return {
        "model_id": model_id,
        "model_version_id": model_version_id,
        "model_artifact_sha256": model_artifact_sha256,
        "approved": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "thresholds": {
            "decision_threshold": float(decision_threshold),
            "min_auc": float(min_auc),
            "min_ks": float(min_ks),
            "min_disparate_impact": float(min_disparate_impact),
            "max_fpr_gap": float(max_fpr_gap),
            "shap_parity_tolerance": float(shap_parity_tolerance),
            "max_auc_regression": float(max_auc_regression),
        },
        "metrics": {
            "challenger_roc_auc": challenger_auc,
            "incumbent_roc_auc": incumbent_auc,
            "roc_auc_delta": float(challenger_auc - incumbent_auc),
            "challenger_ks": challenger_ks,
            "incumbent_ks": incumbent_ks,
        },
        "fairness": fairness,
        "shap_parity": {
            "max_abs_diff_logit": shap_max_diff,
            "tolerance": float(shap_parity_tolerance),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def evaluate_from_payload(payload: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """Evaluate a serialized MRM benchmark payload into a Model Card dict."""
    kwargs: dict[str, Any] = {
        "y_true": payload["y_true"],
        "challenger_scores": payload["challenger_scores"],
        "incumbent_scores": payload["incumbent_scores"],
        "groups": payload["groups"],
        "treeshap_logits": payload["treeshap_logits"],
        "bqml_logits": payload["bqml_logits"],
        "model_id": payload.get("model_id", "credit_xgb_cpr"),
        "model_version_id": payload.get("model_version_id", "v1"),
        "model_artifact_sha256": payload.get("model_artifact_sha256", ""),
    }
    for key, val in overrides.items():
        if val is not None:
            kwargs[key] = val
    return evaluate_challenger_gate(**kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MRM challenger vs incumbent gate.")
    parser.add_argument("--input-json", required=True, help="Path to evaluation dataset JSON.")
    parser.add_argument("--output-card", default="model_card.json", help="Path to output model_card.json.")
    parser.add_argument("--min-disparate-impact", type=float, default=None)
    parser.add_argument("--max-fpr-gap", type=float, default=None)
    parser.add_argument("--shap-tolerance", type=float, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    card = evaluate_from_payload(
        payload,
        min_disparate_impact=args.min_disparate_impact,
        max_fpr_gap=args.max_fpr_gap,
        shap_parity_tolerance=args.shap_tolerance,
    )
    out_path = Path(args.output_card)
    out_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if card["approved"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
