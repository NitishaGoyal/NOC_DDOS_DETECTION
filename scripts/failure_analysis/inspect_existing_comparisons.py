#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


DEFAULT_FILES = [
    Path("scripts/compare_v3_conv_vs_tcn_errors.py"),
    Path("scripts/compare_v3_conv_vs_meanpool_errors.py"),
]


METADATA_NAMES = [
    "run_id",
    "split",
    "profile",
    "active_cores",
    "attackers",
    "strength",
    "seed",
    "end_epoch",
    "feature_cols",
]


PREDICTION_NAMES = [
    "graph_prob",
    "node_prob",
    "y_graph",
    "y_node",
    "graph_logit",
    "node_logit",
    "sample_id",
    "exact_loc",
]


OUTPUT_PATTERNS = {
    "text_write": r"write_text|open\s*\(",
    "numpy_save": r"np\.save|np\.savez",
    "csv_export": r"to_csv|csv\.writer|DictWriter",
    "parquet_export": r"to_parquet|pyarrow",
    "json_export": r"json\.dump|jsonlines|jsonl",
    "pickle_export": r"pickle\.dump|to_pickle",
}


def find_relevant_lines(
    source: str,
    patterns: list[str],
) -> list[str]:
    compiled = re.compile("|".join(patterns))
    results: list[str] = []

    for line_number, line in enumerate(
        source.splitlines(),
        start=1,
    ):
        if compiled.search(line):
            results.append(f"{line_number}: {line.rstrip()}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect existing V3 comparison scripts."
    )
    parser.add_argument(
        "--file",
        action="append",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    files = args.file or DEFAULT_FILES

    for path in files:
        path = path.resolve()

        print("\n" + "=" * 100)
        print("FILE:", path)
        print("=" * 100)

        if not path.is_file():
            print("MISSING")
            continue

        source = path.read_text()

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            print("AST_PARSE_ERROR:", repr(exc))
            continue

        functions = [
            node.name
            for node in tree.body
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
        ]

        classes = [
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        ]

        print("functions:", functions)
        print("classes:", classes)

        print("\nMETADATA FIELDS REFERENCED")

        for name in METADATA_NAMES:
            print(
                f"{name}:",
                "YES" if repr(name) in source or f'"{name}"' in source else "NO",
            )

        print("\nPREDICTION FIELDS REFERENCED")

        for name in PREDICTION_NAMES:
            print(
                f"{name}:",
                "YES" if name in source else "NO",
            )

        print("\nOUTPUT METHODS")

        detected_output_methods = []

        for output_name, pattern in OUTPUT_PATTERNS.items():
            detected = bool(re.search(pattern, source))
            print(
                f"{output_name}:",
                "YES" if detected else "NO",
            )

            if detected:
                detected_output_methods.append(output_name)

        reusable_prediction_export = any(
            method in detected_output_methods
            for method in [
                "numpy_save",
                "csv_export",
                "parquet_export",
                "json_export",
                "pickle_export",
            ]
        )

        print(
            "reusable_sample_prediction_export_detected:",
            reusable_prediction_export,
        )

        print(
            "summary_text_output_detected:",
            "text_write" in detected_output_methods,
        )

        print(
            "loads_end_epoch:",
            "end_epoch" in source,
        )

        print(
            "contains_explicit_false_positive_mask:",
            bool(
                re.search(
                    r"false.?positive|fp_mask|pred.*==\s*1.*true.*==\s*0",
                    source,
                    re.IGNORECASE,
                )
            ),
        )

        print(
            "contains_explicit_false_negative_mask:",
            bool(
                re.search(
                    r"false.?negative|fn_mask|pred.*==\s*0.*true.*==\s*1",
                    source,
                    re.IGNORECASE,
                )
            ),
        )

        print(
            "contains_model_disagreement_logic:",
            bool(
                re.search(
                    r"disagree|only.*correct|conv_correct|tcn_correct|agreement",
                    source,
                    re.IGNORECASE,
                )
            ),
        )

        print(
            "contains_localization_error_taxonomy:",
            bool(
                re.search(
                    r"underprediction|overprediction|disjoint|mixed.?error|cardinality",
                    source,
                    re.IGNORECASE,
                )
            ),
        )

        print("\nRELEVANT SOURCE LINES")

        relevant_lines = find_relevant_lines(
            source,
            [
                r"def infer",
                r"return \{",
                r"graph_prob",
                r"node_prob",
                r"run_id",
                r"end_epoch",
                r"args\.out",
                r"write_text",
                r"np\.save",
                r"to_csv",
                r"to_parquet",
                r"false.?positive",
                r"false.?negative",
                r"disagree",
                r"exact_loc",
                r"group_values",
                r"group_metrics",
            ],
        )

        for line in relevant_lines:
            print(line)


if __name__ == "__main__":
    main()
