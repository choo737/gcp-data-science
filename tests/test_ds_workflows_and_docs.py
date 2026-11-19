"""Hermetic tests for workflows, SQL, linker script, CI/CD, and README."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent

LOCKED_LEDGER_COLUMNS = (
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


class DsWorkflowsAndDocsTest(unittest.TestCase):
    """Verify Makefile, linker, workflows, SQL, KFP DAG, CI, and 10-section README."""

    def test_makefile_uses_data_science_backend_prefix(self) -> None:
        mk = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("data-science/$(strip $(DOMAIN_NAME))/$(NORMALIZED_STAGE)/$(ENV)", mk)
        self.assertIn("data-science/$(NORMALIZED_STAGE)/$(ENV)", mk)

    def test_readme_has_canonical_10_sections_and_plate(self) -> None:
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue(readme.startswith("# GCP Data Science"))
        self.assertIn("![Layer 04 — gcp-data-science](assets/cloud-city-data-science.webp)", readme)
        expected_headings = (
            "## Template Suite Overview",
            "## Create Your Private Repository (Do Not Fork Publicly)",
            "## Architecture & Capabilities",
            "## 6-Step Enterprise Operationalization",
            "## AI Coding Agent Quickstart",
            "## CI/CD Workflows",
            "## Repository Layout",
            "## Quick Verification",
            "## Documentation Index",
            "## License",
        )
        for heading in expected_headings:
            self.assertIn(heading, readme)

    def test_github_workflows_sha_pinned_no_expr_in_run_and_mrm_gate(self) -> None:
        workflows_dir = REPO_ROOT / ".github" / "workflows"
        files = sorted(workflows_dir.glob("*.yml"))
        self.assertEqual(
            {f.name for f in files},
            {"ci.yml", "build-cpr-and-scorer.yml", "terraform-deploy.yml"},
        )

        uses_re = re.compile(r"uses:\s+([^\s#]+)")
        sha_pin_re = re.compile(r"^[^@]+@[0-9a-f]{40}$")
        for wf in files:
            text = wf.read_text(encoding="utf-8")
            with self.subTest(workflow=wf.name):
                for action_ref in uses_re.findall(text):
                    self.assertRegex(action_ref, sha_pin_re, f"Unpinned action {action_ref} in {wf.name}")

                lines = text.splitlines()
                in_run = False
                run_indent = 0
                for line in lines:
                    stripped = line.lstrip()
                    indent = len(line) - len(stripped)
                    if in_run:
                        if stripped and indent <= run_indent:
                            in_run = False
                        else:
                            self.assertNotIn("${{", line, f"Expression injection risk in run block ({wf.name}): {line}")
                    if stripped.startswith("run:"):
                        self.assertNotIn("${{", line, f"Expression injection risk in run line ({wf.name}): {line}")
                        if stripped.endswith("|"):
                            in_run = True
                            run_indent = indent

        deploy_text = (workflows_dir / "terraform-deploy.yml").read_text(encoding="utf-8")
        self.assertIn("model-validator-gate:", deploy_text)
        self.assertIn("environment: model-validator", deploy_text)
        self.assertIn("ALLOW_INFRA_ONLY_PROD_APPLY", deploy_text)
        self.assertIn("MRM_EVAL_INPUT_JSON is missing or unreadable; prod apply blocked (fail-closed).", deploy_text)
        self.assertIn("gcloud run jobs execute", deploy_text)
        self.assertIn("GCP_DS_WIF_SA || vars.GCP_WIF_SERVICE_ACCOUNT", deploy_text)
        self.assertIn("data-science/data-science-outputs.json", deploy_text)
        self.assertIn('data-science/${DOMAIN_NAME}/data-science-outputs.json', deploy_text)
        self.assertEqual(deploy_text.count('--domain-name "${DOMAIN_NAME}" --tfstate-bucket "${TF_STATE_BUCKET}"'), 3)

    def test_bqml_workflow_yaml_and_sql_contracts(self) -> None:
        wf_text = (REPO_ROOT / "workflows" / "bqml_xai_pipeline.workflows.yaml").read_text(encoding="utf-8")
        self.assertNotIn("v1beta1", wf_text)
        for forbidden_param in ("@model_fqn", "@feature_fqn", "@ledger_fqn", "predicted_approved_flag_probs"):
            self.assertNotIn(forbidden_param, wf_text)
        for required in (
            "main:",
            "params: [input]",
            "text.match_regex",
            "googleapis.bigquery.v2.jobs.insert",
            "googleapis.bigquery.v2.jobs.get",
            "BOOSTED_TREE_CLASSIFIER",
            "enable_global_explain = TRUE",
            "model_registry = 'VERTEX_AI'",
            "ML.ENTITY_FEATURES_AT_TIME",
            "ML.FEATURES_AT_TIME",
            "ML.EXPLAIN_PREDICT",
            "FALSE AS approx_feature_contrib",
            "prediction_value",
            "JSON_OBJECT",
            "max_polls: 120",
            'parameterMode: "NAMED"',
            "useLegacySql: false",
        ):
            self.assertIn(required, wf_text)

        for expr in re.findall(r"'\$\{(.+)\}'", wf_text):
            stripped_strings = re.sub(r'"[^"]*"', '""', expr)
            self.assertNotIn("{", stripped_strings)
            self.assertNotIn("?", stripped_strings)

        wf_tf = (REPO_ROOT / "modules" / "ds" / "bqml-xai-pipeline" / "main.tf").read_text(encoding="utf-8")
        env_calls = set(re.findall(r'sys\.get_env\("([A-Z0-9_]+)"\)', wf_text))
        user_env_calls = [e for e in env_calls if not e.startswith("GOOGLE_CLOUD_WORKFLOW_")]
        self.assertLessEqual(len(user_env_calls), 20)
        for env_name in user_env_calls:
            self.assertIn(env_name, wf_tf)

        sql_text = (REPO_ROOT / "sql" / "track_a_bqml_xai.sql").read_text(encoding="utf-8")
        for col in LOCKED_LEDGER_COLUMNS:
            self.assertIn(col, sql_text)
        for feature_col in (
            "tenure_months",
            "avg_monthly_spend",
            "rolling_order_count",
            "days_since_last_order",
            "cancellation_rate_30d",
            "discount_utilization_ratio",
        ):
            self.assertIn(feature_col, sql_text)
            self.assertIn(feature_col, wf_text)
        self.assertIn("ML.ENTITY_FEATURES_AT_TIME", sql_text)
        self.assertIn("APPROX_FEATURE_CONTRIB", sql_text.upper())
        self.assertNotIn("predicted_approved_flag_probs", sql_text)
        self.assertNotIn("customer_id", sql_text)

    def test_kfp_components_ast_zero_closures_and_packages_to_install(self) -> None:
        import builtins

        kfp_path = REPO_ROOT / "src" / "pipelines" / "credit_scoring_kfp.py"
        source = kfp_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(kfp_path))

        builtin_names = set(dir(builtins))

        def _collect_target_names(node: ast.AST, out: set[str]) -> None:
            if isinstance(node, ast.Name):
                out.add(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    _collect_target_names(elt, out)
            elif isinstance(node, ast.Starred):
                _collect_target_names(node.value, out)

        component_funcs: list[ast.FunctionDef] = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "component":
                    component_funcs.append(node)
                    pkg_kw = [kw for kw in dec.keywords if kw.arg == "packages_to_install"]
                    self.assertEqual(len(pkg_kw), 1, f"{node.name} missing packages_to_install")
                    self.assertIsInstance(pkg_kw[0].value, ast.List)
                    self.assertGreater(len(pkg_kw[0].value.elts), 0, f"{node.name} has empty packages_to_install")

        self.assertEqual(
            {fn.name for fn in component_funcs},
            {"extract_pit_features", "train_xgboost_challenger", "evaluate_mrm_gate", "register_cpr_model"},
        )

        for fn in component_funcs:
            bound: set[str] = {arg.arg for arg in fn.args.args + fn.args.kwonlyargs}
            if fn.args.vararg:
                bound.add(fn.args.vararg.arg)
            if fn.args.kwarg:
                bound.add(fn.args.kwarg.arg)
            comp_vars: set[str] = set()
            loaded: set[str] = set()

            for stmt in fn.body:
                for sub in ast.walk(stmt):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        for alias in sub.names:
                            bound.add(alias.asname or alias.name.split(".")[0])
                    elif isinstance(sub, ast.Assign):
                        for tgt in sub.targets:
                            _collect_target_names(tgt, bound)
                    elif isinstance(sub, ast.AnnAssign):
                        _collect_target_names(sub.target, bound)
                    elif isinstance(sub, ast.AugAssign):
                        _collect_target_names(sub.target, bound)
                    elif isinstance(sub, ast.For):
                        _collect_target_names(sub.target, bound)
                    elif isinstance(sub, ast.With):
                        for item in sub.items:
                            if item.optional_vars is not None:
                                _collect_target_names(item.optional_vars, bound)
                    elif isinstance(sub, ast.ExceptHandler) and sub.name:
                        bound.add(sub.name)
                    elif isinstance(sub, ast.comprehension):
                        _collect_target_names(sub.target, comp_vars)
                    elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                        loaded.add(sub.id)

            unbound = loaded - bound - comp_vars - builtin_names
            self.assertEqual(unbound, set(), f"Component {fn.name} references unbound outer symbols: {sorted(unbound)}")

        try:
            from kfp import compiler  # type: ignore
            from src.pipelines.credit_scoring_kfp import credit_scoring_xai_pipeline
        except ImportError:
            return

        with tempfile.TemporaryDirectory() as td:
            pkg_path = Path(td) / "credit_scoring_xai_pipeline.yaml"
            compiler.Compiler().compile(pipeline_func=credit_scoring_xai_pipeline, package_path=str(pkg_path))
            compiled = pkg_path.read_text(encoding="utf-8")
            self.assertIn("evaluate-mrm-gate", compiled)
            self.assertIn("register-cpr-model", compiled)

    def test_enterprise_and_sandbox_ledger_schemas_contain_18_columns(self) -> None:
        for stage in ("enterprise", "sandbox"):
            tf_text = (REPO_ROOT / "stages" / stage / "main.tf").read_text(encoding="utf-8")
            with self.subTest(stage=stage):
                for col in LOCKED_LEDGER_COLUMNS:
                    self.assertIn(f'name = "{col}"', tf_text)
                for label_col in ("application_id", "entity_id", "label_timestamp", "approved_flag", "demographic_group"):
                    self.assertIn(f'name = "{label_col}"', tf_text)
                self.assertIn("require_partition_filter = true", tf_text)
        ent_text = (REPO_ROOT / "stages" / "enterprise" / "main.tf").read_text(encoding="utf-8")
        self.assertIn("count = var.create_ml_audit_ledger_table ? 1 : 0", ent_text)

    def test_feature_store_blocks_jakarta_online_serving(self) -> None:
        fs_tf = (REPO_ROOT / "modules" / "ds" / "feature-store" / "main.tf").read_text(encoding="utf-8")
        self.assertIn('var.region != "asia-southeast2"', fs_tf)
        self.assertIn("bigtable {", fs_tf)
        self.assertIn("cpu_utilization_target = 50", fs_tf)

    def test_link_stages_maps_upstream_aliases_and_preserves_shared_uris(self) -> None:
        script_path = REPO_ROOT / "scripts" / "link-stages.sh"
        self.assertTrue(script_path.is_file())
        self.assertTrue(script_path.stat().st_mode & stat.S_IXUSR)

        syntax = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True, check=False)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            gcs_store = tmp_path / "gcs_store"
            src = tmp_path / "foundation"
            dst = tmp_path / "stages" / "enterprise"
            for d in (bin_dir, gcs_store, src, dst):
                d.mkdir(parents=True)

            (dst / "variables.tf").write_text(
                'variable "core_project_id" { type = string }\n'
                'variable "egress_connector_subnet_id" { type = string }\n'
                'variable "alert_notification_channel_ids" { type = list(string) }\n'
                'variable "kms_key_id" { type = string }\n'
                'variable "vpc_id" { type = string }\n'
                'variable "environment" { type = string }\n',
                encoding="utf-8",
            )

            tf_shim = bin_dir / "terraform"
            tf_shim.write_text(
                "#!/usr/bin/env python3\n"
                "import json\n"
                "print(json.dumps({\n"
                "  'core_storage_project_id': {'value': 'proj-core-01'},\n"
                "  'subnet_id': {'value': 'projects/net/regions/asia-southeast1/subnetworks/sub-01'},\n"
                "  'notification_channel_ids': {'value': ['projects/ops/notificationChannels/1']},\n"
                "  'kms_key_name': {'value': 'projects/kms/locations/asia-southeast1/keyRings/k/cryptoKeys/c'},\n"
                "  'network_id': {'value': 'projects/net/global/networks/vpc-01'}\n"
                "}))\n",
                encoding="utf-8",
            )
            tf_shim.chmod(0o755)

            gcloud_shim = bin_dir / "gcloud"
            gcloud_shim.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import shutil, sys\n"
                f"store = Path({str(gcs_store)!r})\n"
                "args = sys.argv[1:]\n"
                "src_arg, dst_arg = args[2], args[3]\n"
                "if dst_arg.startswith('gs://'):\n"
                "    shutil.copyfile(src_arg, store / dst_arg.replace('gs://', '').replace('/', '__'))\n"
                "elif src_arg.startswith('gs://') and dst_arg == '-':\n"
                "    sys.stdout.write((store / src_arg.replace('gs://', '').replace('/', '__')).read_text(encoding='utf-8'))\n",
                encoding="utf-8",
            )
            gcloud_shim.chmod(0o755)

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            res = subprocess.run(
                [str(script_path), "--env", "staging", str(src), str(dst)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            linked = json.loads((dst / "00_linked.auto.tfvars.json").read_text(encoding="utf-8"))
            self.assertEqual(linked["core_project_id"], "proj-core-01")
            self.assertEqual(linked["egress_connector_subnet_id"], "projects/net/regions/asia-southeast1/subnetworks/sub-01")
            self.assertEqual(linked["alert_notification_channel_ids"], ["projects/ops/notificationChannels/1"])
            self.assertEqual(linked["kms_key_id"], "projects/kms/locations/asia-southeast1/keyRings/k/cryptoKeys/c")
            self.assertEqual(linked["vpc_id"], "projects/net/global/networks/vpc-01")
            self.assertEqual(linked["environment"], "staging")

            # Verify shared/ URI preservation on --export-gcs.
            exp_res = subprocess.run(
                [str(script_path), "--env", "dev", "--export-gcs", "gs://state-bkt/shared/org.json", str(src)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(exp_res.returncode, 0, exp_res.stderr)
            self.assertTrue((gcs_store / "state-bkt__shared__org.json").is_file())

            # Verify backend prefix resolution for external git repos vs internal stages.
            tf_log = tmp_path / "tf_invocations.log"
            tf_shim.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                f"log_path = {str(tf_log)!r}\n"
                "with open(log_path, 'a', encoding='utf-8') as f:\n"
                "    f.write(' '.join(sys.argv[1:]) + '\\n')\n"
                "if 'output' in sys.argv:\n"
                "    print(json.dumps({'core_storage_project_id': {'value': 'proj-core-02'}}))\n",
                encoding="utf-8",
            )
            tf_shim.chmod(0o755)

            ext_repo = tmp_path / "gcp-foundation"
            (ext_repo / ".git").mkdir(parents=True)
            ext_stage = ext_repo / "data" / "enterprise" / "03-data-science"
            (ext_stage / "envs").mkdir(parents=True)

            ext_res = subprocess.run(
                [
                    str(script_path),
                    "--env",
                    "staging",
                    "--tfstate-bucket",
                    "state-bkt",
                    "--from-local",
                    str(ext_stage),
                    str(dst),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(ext_res.returncode, 0, ext_res.stderr)

            int_stage_envs = REPO_ROOT / "stages" / "enterprise"
            int_res = subprocess.run(
                [
                    str(script_path),
                    "--env",
                    "staging",
                    "--tfstate-bucket",
                    "state-bkt",
                    str(int_stage_envs),
                    str(dst),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(int_res.returncode, 0, int_res.stderr)

            log_lines = tf_log.read_text(encoding="utf-8").splitlines()
            init_lines = [line for line in log_lines if " init " in line]
            self.assertTrue(any("-backend-config=prefix=data/enterprise/03-data-science/staging" in l for l in init_lines), init_lines)
            self.assertFalse(any("data-science/data/enterprise/03-data-science" in l for l in init_lines), init_lines)
            self.assertFalse(any(".." in l for l in init_lines), init_lines)
            self.assertTrue(any("-backend-config=prefix=data-science/enterprise/staging" in l for l in init_lines), init_lines)


if __name__ == "__main__":
    unittest.main()
