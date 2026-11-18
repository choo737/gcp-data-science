"""Verify Terraform module, stage, and locked 32-variable symmetry contracts."""

from pathlib import Path
import re
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent

MODULE_FILES = frozenset({
    "README.md",
    "main.tf",
    "variables.tf",
    "outputs.tf",
    "versions.tf",
})

STAGE_BASE_FILES = frozenset({
    "README.md",
    "main.tf",
    "variables.tf",
    "outputs.tf",
    "versions.tf",
    "terraform.tfvars.example",
    ".terraform.lock.hcl",
})

ENTERPRISE_EXTRA_ENTRIES = frozenset({"envs"})
EXPECTED_ENV_EXAMPLES = frozenset({
    "dev.tfvars.example",
    "staging.tfvars.example",
    "prod.tfvars.example",
})

EXPECTED_MODULES = frozenset({
    "modules/ds/feature-store",
    "modules/ds/xai-serving-endpoint",
    "modules/ds/bqml-xai-pipeline",
})

EXPECTED_STAGES = frozenset({
    "stages/sandbox",
    "stages/enterprise",
})

LOCKED_ENTERPRISE_VARIABLES = frozenset({
    "alert_notification_channel_ids",
    "bqml_schedule_cron",
    "core_project_id",
    "cpr_container_image",
    "deploy_cpr_model",
    "domain_name",
    "ds_exp_project_id",
    "ds_serving_track",
    "egress_connector_subnet_id",
    "enable_bqml_schedule",
    "enable_online_feature_store",
    "enable_scorer_service",
    "environment",
    "fairness_min_disparate_impact",
    "feature_group_source_table",
    "feature_online_store_max_nodes",
    "feature_online_store_min_nodes",
    "feature_sync_cron",
    "kms_key_id",
    "labels",
    "manage_kms_iam",
    "ml_artifact_repo_id",
    "ml_audit_dataset_id",
    "ml_audit_table_id",
    "ml_ops_project_id",
    "ml_worm_bucket_name",
    "prefix",
    "region",
    "scorer_container_image",
    "shap_additivity_tolerance",
    "silver_dataset_id",
    "vpc_id",
})

IGNORED_NAMES = frozenset({
    ".terraform",
    "__pycache__",
    ".DS_Store",
    "terraform.tfstate",
    "terraform.tfstate.backup",
    "terraform.tfvars",
})


def _is_ignored(name: str) -> bool:
    if name in IGNORED_NAMES:
        return True
    if name.endswith((".auto.tfvars", ".auto.tfvars.json", ".tfplan")):
        return True
    if name.startswith("terraform.tfstate."):
        return True
    return False


def _visible_entries(directory: Path) -> set[str]:
    return {
        entry.name
        for entry in directory.iterdir()
        if not _is_ignored(entry.name)
    }


def _discover_modules() -> list[Path]:
    ds_dir = REPO_ROOT / "modules" / "ds"
    if not ds_dir.is_dir():
        return []
    return [
        mod_dir
        for mod_dir in sorted(ds_dir.iterdir())
        if mod_dir.is_dir() and not mod_dir.name.startswith(".")
    ]


def _discover_stages() -> list[Path]:
    stages_dir = REPO_ROOT / "stages"
    if not stages_dir.is_dir():
        return []
    return [
        stage_dir
        for stage_dir in sorted(stages_dir.iterdir())
        if stage_dir.is_dir() and not stage_dir.name.startswith(".")
    ]


class RepoSymmetryTest(unittest.TestCase):
    """Validate 3 five-file modules, 2 stages, lockfiles, and exact 32 enterprise variables."""

    def test_modules_obey_five_file_contract(self) -> None:
        modules = _discover_modules()
        discovered_rel = {str(m.relative_to(REPO_ROOT)) for m in modules}
        self.assertEqual(discovered_rel, EXPECTED_MODULES)

        for mod_dir in modules:
            with self.subTest(module=str(mod_dir.relative_to(REPO_ROOT))):
                actual = _visible_entries(mod_dir)
                self.assertEqual(
                    actual,
                    MODULE_FILES,
                    f"{mod_dir} must contain exact 5-file contract: {sorted(MODULE_FILES)}",
                )

    def test_stages_obey_contract_lockfiles_and_envs(self) -> None:
        stages = _discover_stages()
        discovered_rel = {str(s.relative_to(REPO_ROOT)) for s in stages}
        self.assertEqual(discovered_rel, EXPECTED_STAGES)

        for stage_dir in stages:
            rel = str(stage_dir.relative_to(REPO_ROOT))
            with self.subTest(stage=rel):
                actual = _visible_entries(stage_dir)
                expected = set(STAGE_BASE_FILES)
                if rel == "stages/enterprise":
                    expected |= ENTERPRISE_EXTRA_ENTRIES
                self.assertEqual(actual, expected)
                for name in STAGE_BASE_FILES:
                    self.assertTrue((stage_dir / name).is_file(), f"Missing {name} in {rel}")

                lock_text = (stage_dir / ".terraform.lock.hcl").read_text(encoding="utf-8")
                self.assertIn("registry.terraform.io/hashicorp/google", lock_text)
                self.assertIn("registry.terraform.io/hashicorp/google-beta", lock_text)
                self.assertIn("registry.terraform.io/hashicorp/time", lock_text)
                self.assertIn("h1:", lock_text)
                self.assertIn("zh:", lock_text)

                if rel == "stages/enterprise":
                    envs_dir = stage_dir / "envs"
                    self.assertTrue(envs_dir.is_dir())
                    self.assertEqual(_visible_entries(envs_dir), EXPECTED_ENV_EXAMPLES)
                    prod_cfg = (envs_dir / "prod.tfvars.example").read_text(encoding="utf-8")
                    self.assertIn("@sha256:", prod_cfg)

    def test_enterprise_variables_match_locked_32_contract(self) -> None:
        vars_path = REPO_ROOT / "stages" / "enterprise" / "variables.tf"
        var_names = re.findall(r'^\s*variable\s+"([A-Za-z0-9_]+)"\s*\{', vars_path.read_text(encoding="utf-8"), re.MULTILINE)
        self.assertEqual(len(var_names), 32)
        self.assertEqual(set(var_names), LOCKED_ENTERPRISE_VARIABLES)

    def test_versions_tf_pins_providers_and_stage_gcs_backend(self) -> None:
        targets = _discover_modules() + _discover_stages()
        self.assertEqual(len(targets), len(EXPECTED_MODULES) + len(EXPECTED_STAGES))
        tf_pattern = re.compile(r'required_version\s*=\s*">=\s*1\.8\.0"')
        google_pattern = re.compile(
            r'source\s*=\s*"hashicorp/google"[\s\S]*?version\s*=\s*">=\s*8\.0\.0,\s*<\s*9\.0\.0"'
        )
        time_pattern = re.compile(
            r'source\s*=\s*"hashicorp/time"[\s\S]*?version\s*=\s*">=\s*0\.11\.0'
        )
        for target in targets:
            versions_path = target / "versions.tf"
            with self.subTest(versions=str(versions_path.relative_to(REPO_ROOT))):
                text = versions_path.read_text(encoding="utf-8")
                self.assertRegex(text, tf_pattern)
                self.assertRegex(text, google_pattern)
                if target in _discover_stages():
                    self.assertRegex(text, time_pattern)
                    self.assertRegex(text, r'backend\s+"gcs"\s*\{\s*\}')

    def test_no_unsafe_coalesce_or_and_trimspace_in_terraform(self) -> None:
        bad_coalesce = re.compile(r'coalesce\([^)]*""')
        bad_trim_and = re.compile(r'var\.[A-Za-z0-9_]+\s*!=\s*null\s*&&\s*trimspace\(')
        skip_dirs = {".git", ".terraform", "__pycache__", ".pytest_cache", ".venv"}
        for tf_path in sorted(REPO_ROOT.rglob("*.tf")):
            if any(part in skip_dirs for part in tf_path.parts):
                continue
            rel = str(tf_path.relative_to(REPO_ROOT))
            text = tf_path.read_text(encoding="utf-8")
            with self.subTest(file=rel):
                self.assertIsNone(bad_coalesce.search(text), f"Empty-string coalesce in {rel}")
                self.assertIsNone(bad_trim_and.search(text), f"Unsafe null-and-trimspace in {rel}")


if __name__ == "__main__":
    unittest.main()
