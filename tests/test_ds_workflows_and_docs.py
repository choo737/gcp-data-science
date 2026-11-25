"""Hermetic tests for workflows, SQL, linker script, CI/CD, and README."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite_doc_contract

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
        violations = suite_doc_contract.validate_repo_suite_contract(
            REPO_ROOT, "gcp-data-science"
        )
        self.assertEqual(violations, [], "\n".join(violations))
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("\n# GCP Data Science\n", readme)
        self.assertIn('<img src="assets/cloud-city-data-science.webp"', readme)
        for heading in suite_doc_contract.CANONICAL_README_H2_SECTIONS:
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
                try:
                    import yaml  # type: ignore

                    parsed_gh = yaml.safe_load(text)
                    self.assertIsInstance(parsed_gh, dict, f"{wf.name} did not parse to a YAML mapping")
                except ImportError:
                    pass
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

        build_text = (workflows_dir / "build-cpr-and-scorer.yml").read_text(encoding="utf-8")
        top_perms_block = build_text.split("jobs:", 1)[0]
        self.assertNotIn("id-token: write", top_perms_block)
        self.assertIn("id-token: write", build_text.split("verify-and-build:", 1)[1])
        self.assertIn("Validate WIF service account configuration", build_text)

        deploy_text = (workflows_dir / "terraform-deploy.yml").read_text(encoding="utf-8")
        self.assertIn("model-validator-gate:", deploy_text)
        self.assertIn("environment: model-validator", deploy_text)
        self.assertIn("ALLOW_INFRA_ONLY_PROD_APPLY", deploy_text)
        self.assertIn("MRM_EVAL_INPUT_JSON is missing or unreadable; prod apply blocked (fail-closed).", deploy_text)
        self.assertIn("gcloud run jobs execute", deploy_text)
        self.assertIn("gcloud ai endpoints describe", deploy_text)
        self.assertIn("gcloud run services update", deploy_text)
        self.assertIn("VERTEX_PSC_ENDPOINT_IP=", deploy_text)
        self.assertIn("GCP_DS_WIF_SA || vars.GCP_WIF_SERVICE_ACCOUNT", deploy_text)
        self.assertIn("data-science/data-science-outputs.json", deploy_text)
        self.assertIn('data-science/${DOMAIN_NAME}/data-science-outputs.json', deploy_text)
        self.assertEqual(deploy_text.count('--domain-name "${DOMAIN_NAME}" --tfstate-bucket "${TF_STATE_BUCKET}"'), 3)
        for uat_token in (
            "- uat",
            "apply-uat:",
            "environment: uat",
            "./scripts/link-stages.sh --env uat",
            'BACKEND_PREFIX="data-science/${DOMAIN_NAME}/enterprise/uat"',
            'BACKEND_PREFIX="data-science/enterprise/uat"',
            "envs/uat.tfvars",
        ):
            self.assertIn(uat_token, deploy_text)
        self.assertNotIn("envs/uat.tfvars.example", deploy_text)
        self.assertIn('[[ "${MRM_EVAL_INPUT_JSON}" == gs://* ]]', deploy_text)
        uat_example = (REPO_ROOT / "stages" / "enterprise" / "envs" / "uat.tfvars.example").read_text(encoding="utf-8")
        self.assertRegex(uat_example, r'environment\s*=\s*"uat"')
        self.assertIn("cpr_container_image", uat_example)

        linked_keys = (
            "core_project_id",
            "ds_exp_project_id",
            "ml_ops_project_id",
            "silver_dataset_id",
            "ml_audit_dataset_id",
            "ml_artifact_repo_id",
            "ml_worm_bucket_name",
            "vpc_id",
            "egress_connector_subnet_id",
            "kms_key_id",
        )
        tfvars_examples = [REPO_ROOT / "stages" / "enterprise" / "terraform.tfvars.example"] + sorted(
            (REPO_ROOT / "stages" / "enterprise" / "envs").glob("*.tfvars.example")
        )
        for ex_path in tfvars_examples:
            for line in ex_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                for lk in linked_keys:
                    self.assertNotRegex(
                        stripped,
                        rf"^{re.escape(lk)}\s*=",
                        f"Linked key {lk} must remain commented out in {ex_path.name}",
                    )

    def test_bqml_workflow_yaml_and_sql_contracts(self) -> None:
        wf_text = (REPO_ROOT / "workflows" / "bqml_xai_pipeline.workflows.yaml").read_text(encoding="utf-8")
        try:
            import yaml  # type: ignore

            parsed_wf = yaml.safe_load(wf_text)
            self.assertIsInstance(parsed_wf, dict)
            self.assertIn("main", parsed_wf)
            self.assertIn("run_bq_query", parsed_wf)
        except ImportError:
            pass

        # Stdlib YAML syntax guard: reject unquoted ': ' inside key: ${...} values outside block scalars.
        block_indent: int | None = None
        for lineno, line in enumerate(wf_text.splitlines(), start=1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if block_indent is not None:
                if stripped and indent <= block_indent:
                    block_indent = None
                else:
                    continue
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.endswith("|") or stripped.endswith(">"):
                block_indent = indent
                continue
            m_kv = re.match(r"^(?:-\s+)?[A-Za-z0-9_.-]+:\s+(.+)$", stripped)
            if m_kv:
                val = m_kv.group(1)
                if val.startswith("${"):
                    self.assertNotIn(
                        ": ",
                        val,
                        f"Unquoted ': ' in YAML expression at line {lineno}: {line}",
                    )
                    self.assertFalse(
                        val.endswith(":"),
                        f"Unquoted trailing ':' in YAML expression at line {lineno}: {line}",
                    )

        self.assertNotIn("v1beta1", wf_text)
        self.assertIsNone(re.search(r"\{\d+,\s*(?:100[1-9]|10[1-9]\d|1[1-9]\d{2}|[2-9]\d{3,})\}", wf_text))
        for forbidden_param in (
            "@model_fqn",
            "@feature_fqn",
            "@ledger_fqn",
            "predicted_approved_flag_probs",
            "vertex_ai_model_id = @model_id",
            "DATE(label_timestamp)",
        ):
            self.assertNotIn(forbidden_param, wf_text)
        for required in (
            "main:",
            "params: [input]",
            "text.match_regex",
            "len(silver_dataset_id) <= 1024",
            '^[A-Za-z0-9_]+$',
            'input != null and "train_model" in input',
            "googleapis.bigquery.v2.jobs.insert",
            "googleapis.bigquery.v2.jobs.get",
            "jobReference:",
            "location: ${location}",
            "retry: ${http.default_retry}",
            "BOOSTED_TREE_CLASSIFIER",
            "enable_global_explain = TRUE",
            "model_registry = 'VERTEX_AI'",
            "vertex_ai_model_id = '__MODEL_ID__'",
            "ML.ENTITY_FEATURES_AT_TIME",
            "ML.FEATURES_AT_TIME",
            "ML.EXPLAIN_PREDICT",
            "FALSE AS approx_feature_contrib",
            "prediction_value",
            "TreeSHAP additivity breach",
            "LN(GREATEST(1.0 - CAST(p.probability AS FLOAT64), 1e-15) / GREATEST(CAST(p.probability AS FLOAT64), 1e-15))",
            "LN(GREATEST(CAST(p.probability AS FLOAT64), 1e-15) / GREATEST(1.0 - CAST(p.probability AS FLOAT64), 1e-15))",
            "-CAST(p.baseline_prediction_value AS FLOAT64)",
            "'APPROVE' AS explained_class",
            "WHERE label_timestamp >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY))",
            "EXPORT DATA OPTIONS",
            "overwrite = false",
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
        self.assertIsNone(re.search(r"\{\d+,\s*(?:100[1-9]|10[1-9]\d|1[1-9]\d{2}|[2-9]\d{3,})\}", sql_text))
        self.assertIn("LENGTH(@silver_dataset_id) <= 1024", sql_text)
        self.assertIn("r'^[A-Za-z0-9_]+$'", sql_text)
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
        self.assertIn("WHERE label_timestamp >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY))", sql_text)
        self.assertIn("TreeSHAP additivity breach", sql_text)
        self.assertIn(
            "LN(GREATEST(1.0 - CAST(p.probability AS FLOAT64), 1e-15) / GREATEST(CAST(p.probability AS FLOAT64), 1e-15))",
            sql_text,
        )
        self.assertIn("-CAST(p.baseline_prediction_value AS FLOAT64)", sql_text)
        self.assertIn("vertex_ai_model_id = '%s'", sql_text)
        self.assertNotIn("vertex_ai_model_id = @model_id", sql_text)
        self.assertNotIn("DATE(label_timestamp)", sql_text)
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

        self.assertNotIn("DATE(label_timestamp)", source)
        self.assertIn("WHERE label_timestamp >= TIMESTAMP(DATE_SUB(CURRENT_DATE(), INTERVAL 730 DAY))", source)
        self.assertIn("shap_tolerance * max(1.0, abs(raw_margin))", source)
        self.assertIn("subgroup_ks", source)
        self.assertIn("max_fpr_gap", source)
        self.assertIn("encryption_spec_key_name", source)
        self.assertIn("MODEL_VERSION_ID", source)
        self.assertIn("python:3.13-slim@sha256:", source)
        self.assertEqual(source.count('"pyarrow==26.0.1"'), 3)
        self.assertNotIn("pyarrow==17.0.0", source)
        self.assertNotIn(">=", source.split("packages_to_install=[", 1)[1].split("def ", 1)[0])

        try:
            from kfp import compiler  # type: ignore
            from src.pipelines.credit_scoring_kfp import build_credit_scoring_pipeline, credit_scoring_xai_pipeline
        except ImportError:
            self.skipTest("kfp package not installed in bare stdlib test runner")

        with tempfile.TemporaryDirectory() as td:
            pkg_path = Path(td) / "credit_scoring_xai_pipeline.yaml"
            compiler.Compiler().compile(pipeline_func=credit_scoring_xai_pipeline, package_path=str(pkg_path))
            compiled = pkg_path.read_text(encoding="utf-8")
            self.assertIn("evaluate-mrm-gate", compiled)
            self.assertIn("register-cpr-model", compiled)
            self.assertIn("vw_ml_customer_features_online", compiled)
            self.assertIn("CONTAINER_IMAGE_DIGEST", compiled)
            self.assertIn("MODEL_VERSION_ID", compiled)
            self.assertIn("SHAP_ADDITIVITY_TOLERANCE", compiled)
            self.assertIn("encryption_spec_key_name", compiled)

            custom_pipeline = build_credit_scoring_pipeline(
                pipeline_name="custom-credit-xai",
                feature_columns=["tenure_months", "avg_monthly_spend"],
            )
            custom_path = Path(td) / "custom_pipeline.yaml"
            compiler.Compiler().compile(pipeline_func=custom_pipeline, package_path=str(custom_path))
            custom_yaml = custom_path.read_text(encoding="utf-8")
            self.assertIn("custom-credit-xai", custom_yaml)

    def test_enterprise_and_sandbox_ledger_schemas_contain_18_columns(self) -> None:
        for stage in ("enterprise", "sandbox"):
            tf_text = (REPO_ROOT / "stages" / stage / "main.tf").read_text(encoding="utf-8")
            with self.subTest(stage=stage):
                for col in LOCKED_LEDGER_COLUMNS:
                    self.assertIn(f'name = "{col}"', tf_text)
                for label_col in ("application_id", "entity_id", "label_timestamp", "approved_flag", "demographic_group"):
                    self.assertIn(f'name = "{label_col}"', tf_text)
                self.assertIn("require_partition_filter = true", tf_text)
                self.assertIn("aiplatform_audit_editor", tf_text)
                self.assertIn('role       = "roles/bigquery.dataEditor"', tf_text)
        ent_text = (REPO_ROOT / "stages" / "enterprise" / "main.tf").read_text(encoding="utf-8")
        self.assertIn("count = var.create_ml_audit_ledger_table ? 1 : 0", ent_text)
        self.assertIn("artifact_repo_project", ent_text)
        self.assertIn("project    = local.artifact_repo_project", ent_text)

        endpoint_tf = (REPO_ROOT / "modules" / "ds" / "xai-serving-endpoint" / "main.tf").read_text(encoding="utf-8")
        self.assertIn("aiplatform_model_viewer", endpoint_tf)
        self.assertIn("google_project_service_identity.aiplatform_agent", endpoint_tf)

        sbx_text = (REPO_ROOT / "stages" / "sandbox" / "main.tf").read_text(encoding="utf-8")
        self.assertIn("aiplatform_silver_viewer", sbx_text)
        self.assertIn('role       = "roles/bigquery.dataViewer"', sbx_text)
        self.assertNotIn("pubsub_agent", sbx_text)
        self.assertNotIn("pubsub_cmek", sbx_text)

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
                [str(script_path), "--env", "uat", str(src), str(dst)],
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
            self.assertEqual(linked["environment"], "uat")

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
            org_stage = ext_repo / "foundation" / "01-org-governance"
            (org_stage / "envs").mkdir(parents=True)

            ext_res = subprocess.run(
                [
                    str(script_path),
                    "--env",
                    "dev",
                    "--domain-name",
                    "retail",
                    "--tfstate-bucket",
                    "state-bkt",
                    "--from-local",
                    str(ext_stage),
                    str(org_stage),
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
                    "uat",
                    "--domain-name",
                    "retail",
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
            self.assertTrue(any("-backend-config=prefix=data/enterprise/03-data-science/dev" in l for l in init_lines), init_lines)
            self.assertTrue(any("-backend-config=prefix=foundation/01-org-governance/shared" in l for l in init_lines), init_lines)
            self.assertFalse(any("retail/data/enterprise/03-data-science" in l for l in init_lines), init_lines)
            self.assertFalse(any("data-science/data/enterprise/03-data-science" in l for l in init_lines), init_lines)
            self.assertFalse(any(".." in l for l in init_lines), init_lines)
            self.assertTrue(any("-backend-config=prefix=data-science/retail/enterprise/uat" in l for l in init_lines), init_lines)

            # Verify non-.git archive fallback resolves via parent directory of stages/foundation/data.
            nogit_repo = tmp_path / "extracted-archive"
            nogit_stage = nogit_repo / "data" / "enterprise" / "03-data-science"
            (nogit_stage / "envs").mkdir(parents=True)
            nogit_res = subprocess.run(
                [
                    str(script_path),
                    "--env",
                    "prod",
                    "--tfstate-bucket",
                    "state-bkt",
                    str(nogit_stage),
                    str(dst),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(nogit_res.returncode, 0, nogit_res.stderr)
            nogit_lines = tf_log.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any("-backend-config=prefix=data/enterprise/03-data-science/prod" in l for l in nogit_lines), nogit_lines)

            # Verify --env uat succeeds, non-.git apps/ fallback resolves, and invalid --env qa fails.
            uat_res = subprocess.run(
                [str(script_path), "--env", "uat", str(src), str(dst)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(uat_res.returncode, 0, uat_res.stderr)
            uat_linked = json.loads((dst / "00_linked.auto.tfvars.json").read_text(encoding="utf-8"))
            self.assertEqual(uat_linked["environment"], "uat")

            apps_nogit_stage = nogit_repo / "apps" / "enterprise"
            (apps_nogit_stage / "envs").mkdir(parents=True)
            apps_uat_res = subprocess.run(
                [
                    str(script_path),
                    "--env",
                    "uat",
                    "--tfstate-bucket",
                    "state-bkt",
                    str(apps_nogit_stage),
                    str(dst),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(apps_uat_res.returncode, 0, apps_uat_res.stderr)
            apps_lines = tf_log.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any("-backend-config=prefix=apps/enterprise/uat" in l for l in apps_lines), apps_lines)

            bad_env_res = subprocess.run(
                [str(script_path), "--env", "qa", str(src), str(dst)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(bad_env_res.returncode, 0)
            self.assertIn("Invalid --env value: qa", bad_env_res.stderr)

            # Verify external gcp-data-ingestion / gcp-data-science repos prefix before stripping stages/ and --export-gcs= works.
            ext_ing_repo = tmp_path / "gcp-data-ingestion"
            (ext_ing_repo / ".git").mkdir(parents=True)
            ext_ing_stage = ext_ing_repo / "stages" / "enterprise"
            (ext_ing_stage / "envs").mkdir(parents=True)
            ext_ds_repo = tmp_path / "gcp-data-science"
            (ext_ds_repo / ".git").mkdir(parents=True)
            ext_ds_stage = ext_ds_repo / "stages" / "enterprise"
            (ext_ds_stage / "envs").mkdir(parents=True)

            ext_pair_res = subprocess.run(
                [
                    str(script_path),
                    "--env=dev",
                    "--tfstate-bucket=state-bkt",
                    str(ext_ing_stage),
                    str(ext_ds_stage),
                    str(dst),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(ext_pair_res.returncode, 0, ext_pair_res.stderr)
            pair_lines = tf_log.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any("-backend-config=prefix=ingestion/enterprise/dev" in l for l in pair_lines), pair_lines)
            self.assertTrue(any("-backend-config=prefix=data-science/enterprise/dev" in l for l in pair_lines), pair_lines)

            eq_export_res = subprocess.run(
                [
                    str(script_path),
                    "--env=dev",
                    "--export-gcs=gs://state-bkt/shared/flag-eq.json",
                    str(src),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(eq_export_res.returncode, 0, eq_export_res.stderr)
            self.assertTrue((gcs_store / "state-bkt__shared__flag-eq.json").is_file())

    def test_link_stages_routes_ai_subnet_and_allows_ds_alias_override(self) -> None:
        script_path = REPO_ROOT / "scripts" / "link-stages.sh"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_dir = tmp_path / "bin"
            gcs_store = tmp_path / "gcs_store"
            net_stage = tmp_path / "02-security-network"
            ds_stage = tmp_path / "03-data-science"
            dst = tmp_path / "stages" / "enterprise"
            for d in (bin_dir, gcs_store, net_stage, ds_stage, dst):
                d.mkdir(parents=True)

            (dst / "variables.tf").write_text(
                'variable "egress_connector_subnet_id" { type = string }\n'
                'variable "subnet_id" { type = string }\n'
                'variable "environment" { type = string }\n',
                encoding="utf-8",
            )

            base_sub = "projects/net/regions/asia-southeast1/subnetworks/base-sub"
            data_sub = "projects/net/regions/asia-southeast1/subnetworks/data-sub"
            ai_sub = "projects/net/regions/asia-southeast1/subnetworks/ai-sub"
            ds_override_sub = "projects/net/regions/asia-southeast1/subnetworks/ds-override-sub"
            explicit_egress_sub = "projects/net/regions/asia-southeast1/subnetworks/explicit-egress-sub"

            tf_shim = bin_dir / "terraform"
            tf_shim.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "chdir = next(a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('-chdir='))\n"
                "mode = os.environ.get('DS_SUBNET_TEST_MODE', 'ai_subnet')\n"
                "if '02-security-network' in chdir:\n"
                "    payload = {\n"
                "        'environment': {'value': 'uat', 'sensitive': False},\n"
                f"        'subnet_id': {{'value': {base_sub!r}, 'sensitive': False}},\n"
                f"        'data_subnet_id': {{'value': {data_sub!r}, 'sensitive': False}},\n"
                "    }\n"
                "    if mode != 'alias_override_only':\n"
                f"        payload['ai_subnet_id'] = {{'value': {ai_sub!r}, 'sensitive': False}}\n"
                "    print(json.dumps(payload))\n"
                "else:\n"
                "    if mode == 'explicit_egress':\n"
                "        print(json.dumps({\n"
                "            'environment': {'value': 'uat', 'sensitive': False},\n"
                f"            'egress_connector_subnet_id': {{'value': {explicit_egress_sub!r}, 'sensitive': False}},\n"
                "        }))\n"
                "    else:\n"
                "        print(json.dumps({\n"
                "            'environment': {'value': 'uat', 'sensitive': False},\n"
                f"            'subnet_id': {{'value': {ds_override_sub!r}, 'sensitive': False}},\n"
                "        }))\n",
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

            # Case 1: ai_subnet_id present -> routes into both egress_connector_subnet_id and subnet_id (local + GCS).
            res_ai = subprocess.run(
                [str(script_path), "--env", "uat", str(net_stage), str(dst)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res_ai.returncode, 0, res_ai.stderr)
            linked_ai = json.loads((dst / "00_linked.auto.tfvars.json").read_text(encoding="utf-8"))
            self.assertEqual(linked_ai["egress_connector_subnet_id"], ai_sub)
            self.assertEqual(linked_ai["subnet_id"], ai_sub)

            gcs_uri = "gs://example-state/foundation-outputs.json"
            exp_res = subprocess.run(
                [str(script_path), "--env", "uat", "--export-gcs", gcs_uri, str(net_stage)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(exp_res.returncode, 0, exp_res.stderr)
            gcs_res = subprocess.run(
                [str(script_path), "--env", "uat", "--from-gcs", gcs_uri, str(dst)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(gcs_res.returncode, 0, gcs_res.stderr)
            linked_gcs = json.loads((dst / "00_linked.auto.tfvars.json").read_text(encoding="utf-8"))
            self.assertEqual(linked_gcs["egress_connector_subnet_id"], ai_sub)
            self.assertEqual(linked_gcs["subnet_id"], ai_sub)

            # Case 2: 03-data-science subnet_id overrides 02-security-network base subnet_id on OUTPUT_ALIASES.
            override_env = env.copy()
            override_env["DS_SUBNET_TEST_MODE"] = "alias_override_only"
            res_override = subprocess.run(
                [str(script_path), "--env", "uat", str(net_stage), str(ds_stage), str(dst)],
                env=override_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res_override.returncode, 0, res_override.stderr)
            linked_override = json.loads((dst / "00_linked.auto.tfvars.json").read_text(encoding="utf-8"))
            self.assertEqual(linked_override["egress_connector_subnet_id"], ds_override_sub)
            self.assertEqual(linked_override["subnet_id"], ds_override_sub)

            # Case 3: explicit egress_connector_subnet_id from 03-data-science is preserved even if ai_subnet_id exists.
            explicit_env = env.copy()
            explicit_env["DS_SUBNET_TEST_MODE"] = "explicit_egress"
            res_explicit = subprocess.run(
                [str(script_path), "--env", "uat", str(net_stage), str(ds_stage), str(dst)],
                env=explicit_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res_explicit.returncode, 0, res_explicit.stderr)
            linked_explicit = json.loads((dst / "00_linked.auto.tfvars.json").read_text(encoding="utf-8"))
            self.assertEqual(linked_explicit["egress_connector_subnet_id"], explicit_egress_sub)
            self.assertEqual(linked_explicit["subnet_id"], ai_sub)


if __name__ == "__main__":
    unittest.main()
