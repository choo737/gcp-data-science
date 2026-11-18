"""Model Risk Management (MRM) validation gate and Model Card generator."""

from .evaluate_gate import (
    compute_ks_statistic,
    compute_roc_auc,
    compute_shap_parity_max_diff,
    compute_subgroup_fairness,
    evaluate_challenger_gate,
)

__all__ = [
    "compute_ks_statistic",
    "compute_roc_auc",
    "compute_shap_parity_max_diff",
    "compute_subgroup_fairness",
    "evaluate_challenger_gate",
]
