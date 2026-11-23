"""Score saved predictions in a separate process with the pinned upstream functions."""
from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from importlib.metadata import version
from types import SimpleNamespace

from .insightbench_assets import MANIFEST, asset

SCORER = "insightbench_upstream_rouge1_compat"


def _rouge():
    import numpy as np
    from rouge_score.rouge_scorer import RougeScorer

    rouge = RougeScorer(["rouge1"], use_stemmer=False)

    def compute(*, predictions, references, rouge_types):
        # Upstream compute_rouge_score calls the HF metric once per text pair.
        # Its singleton bootstrap aggregate equals that pair's F-measure.
        if len(predictions) != 1 or len(references) != 1 or rouge_types != ["rouge1"]:
            raise ValueError("Expected the pinned upstream single-pair ROUGE-1 call")
        return {"rouge1": rouge.score(references[0], predictions[0])["rouge1"].fmeasure}

    def load(metric):
        if metric != "rouge":
            raise ValueError("This compatibility scorer supports ROUGE-1 only")
        return SimpleNamespace(compute=compute)

    namespace = {"np": np, "defaultdict": defaultdict, "evaluate": SimpleNamespace(load=load)}
    functions = {
        "insightbench/utils/eval_utils.py": "compute_rouge_score",
        "insightbench/utils/metrics_utils.py": "score_insight",
        "insightbench/metrics.py": "compute_rouge",
        "insightbench/benchmarks.py": "evaluate_insights",
    }
    # Execute the exact function definitions from hash-checked upstream files.
    for path, name in functions.items():
        source = asset(path)
        tree = ast.parse(source.read_bytes(), filename=path)
        function, = (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module([function], type_ignores=[]), path, "exec"), namespace)
    namespace["metrics"] = SimpleNamespace(compute_rouge=namespace["compute_rouge"])
    namespace["mu"] = SimpleNamespace(score_insight=namespace["score_insight"])
    return namespace["evaluate_insights"]


def score(predictions: list[dict]) -> dict:
    evaluate = _rouge()
    if float(evaluate(["scorer self-check"], ["scorer self-check"], score_name="rouge1")) != 1.0:
        raise ValueError("Pinned upstream scorer self-check failed")
    rows = []
    for row in predictions:
        case = MANIFEST["cases"][row["case"]]
        official = json.loads(asset(case["reference"]).read_text())
        if case["goal"] != official["metadata"]["goal"] or case["csv"] != official["dataset_csv_path"]:
            raise ValueError("Official goal or CSV pointer does not match the pinned manifest")
        if row["data_sha256"] != MANIFEST["files"][case["csv"]]:
            raise ValueError("Saved predictions use an unexpected CSV")
        reference = official["insights"]
        text = row["insights"]
        rows.append({
            "case": row["case"], "strategy": row["strategy"],
            "rouge1": float(evaluate(text, reference, score_name="rouge1")) if text else 0.0,
            "references": len(reference), "prediction_count": len(text), "empty_prediction": not text,
        })
    return {"implementation": SCORER, "upstream_revision": MANIFEST["revision"],
            "entry_point": "insightbench.benchmarks.evaluate_insights",
            "rouge_score_version": version("rouge-score"), "numpy_version": version("numpy"),
            "compatibility": "Pinned upstream functions with local singleton HF ROUGE backend; empty predicts score zero.",
            "runs": rows}


if __name__ == "__main__":
    print(json.dumps(score(json.load(sys.stdin)), allow_nan=False))
