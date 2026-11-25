"""CLI wrapper delegating CPR model upload and deployment to src/cpr_treeshap/deploy_cpr.py."""

from __future__ import annotations

from pathlib import Path
import sys

_CPR_DIR = str(Path(__file__).resolve().parent.parent / "src" / "cpr_treeshap")
if _CPR_DIR not in sys.path:
    sys.path.insert(0, _CPR_DIR)

from deploy_cpr import deploy_cpr_model, main, parse_deploy_config  # noqa: E402

__all__ = ["deploy_cpr_model", "main", "parse_deploy_config"]


if __name__ == "__main__":
    raise SystemExit(main())
