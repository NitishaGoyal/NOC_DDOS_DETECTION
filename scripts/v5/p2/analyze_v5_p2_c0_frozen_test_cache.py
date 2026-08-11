#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

STAGE = "V5_P2_C0_FROZEN_TEST_CACHE_FAILURE_ANALYSIS"
COMPLETE = f"{STAGE}_COMPLETE"
ROLES = ("source", "transit", "victim", "path")
EXPECTED_ITEMS = 11666
EXPECTED_PAIRS = 69
EXPECTED_RUNS = 138
EXPECTED_NODES = 16
EXPECTED_AUTH_ID = "95da82aebaf1b61b13ce4bf31346b08500037841369cfd3fe5dba0965dff24e2"
EXPECTED_CKPT_SHA = "7d4afae2214f67ecd9c65c6ff4614ba8238614234d7b07cdf1408b6d3efb07ef"
EXPECTED_THRESHOLD_SHA = "64bb23f1d0f36ec5f09236c4939317cddc96260f0ed4695036c4e802cd3056a7"
EXPECTED_THRESHOLDS = {
    "attack": 0.2,
    "source": 0.778,
    "transit": 0.824,
    "victim": 0.661,
    "path": 0.545,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True) + '\n')


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def div(a: int | float, b: int | float) -> float:
    return float(a) / float(b) if b else 0.0


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = y.astype(bool).reshape(-1)
    p = p.astype(bool).reshape(-1)
    tn = int((~y & ~p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    tp = int((y & p).sum())
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    f1 = div(2 * precision * recall, precision + recall)
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'accuracy': div(tp + tn, tp + tn + fp + fn),
        'balanced_accuracy': 0.5 * (recall + specificity),
        'precision': precision, 'recall': recall,
        'specificity': specificity, 'fpr': div(fp, fp + tn), 'f1': f1,
    }


def node_set_metrics(truth: np.ndarray, pred: np.ndarray, active: np.ndarray) -> dict[str, Any]:
    exact = (truth == pred).all(axis=1)
    return {
        'node': metrics(truth.reshape(-1), pred.reshape(-1)),
        'exact_all': float(exact.mean()),
        'exact_attack': float(exact[active].mean()),
        'exact_control': float(exact[~active].mean()),
    }


def nodes_string(v: np.ndarray) -> str:
    return ','.join(str(i) for i in np.flatnonzero(v))


def quantiles(v: np.ndarray) -> dict[str, float]:
    x = v.astype(np.float64).reshape(-1)
    return {
        'min': float(x.min()), 'p01': float(np.quantile(x, .01)),
        'p05': float(np.quantile(x, .05)), 'p10': float(np.quantile(x, .10)),
        'p25': float(np.quantile(x, .25)), 'median': float(np.quantile(x, .50)),
        'p75': float(np.quantile(x, .75)), 'p90': float(np.quantile(x, .90)),
        'p95': float(np.quantile(x, .95)), 'p99': float(np.quantile(x, .99)),
        'max': float(x.max()), 'mean': float(x.mean()), 'std': float(x.std()),
    }


def read_pair_manifest(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            result[int(row['pair_ordinal'])] = row['pair_key']
    if len(result) != EXPECTED_PAIRS or sorted(result) != list(range(EXPECTED_PAIRS)):
        raise RuntimeError('pair manifest is not exactly 69 contiguous pairs')
    return result


def episodes(error_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in error_rows:
        groups[(int(row['pair_ordinal']), int(row['mode']), str(row['graph_error_type']))].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r['window_start']))
        current = [rows[0]]
        chunks: list[list[dict[str, Any]]] = []
        for row in rows[1:]:
            if int(row['window_start']) == int(current[-1]['window_start']) + 8:
                current.append(row)
            else:
                chunks.append(current)
                current = [row]
        chunks.append(current)
        for chunk in chunks:
            starts = [int(r['window_start']) for r in chunk]
            scores = [float(r['graph_score']) for r in chunk]
            output.append({
                'pair_ordinal': key[0], 'pair_key': chunk[0]['pair_key'],
                'mode': key[1], 'mode_name': chunk[0]['mode_name'],
                'graph_error_type': key[2], 'episode_start': min(starts),
                'episode_end': max(starts), 'window_count': len(chunk),
                'mean_graph_score': float(np.mean(scores)),
                'min_graph_score': float(np.min(scores)),
                'max_graph_score': float(np.max(scores)),
            })
    return sorted(output, key=lambda r: (-int(r['window_count']), r['graph_error_type'], int(r['pair_ordinal'])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--b6-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    b6 = args.b6_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        print(f'STOP: output already exists: {out}', file=sys.stderr)
        return 2
    out.mkdir(parents=True)

    paths = {
        'report': b6 / 'V5_P2_B6_ONE_SHOT_TEST_EVALUATION.json',
        'lock': b6 / 'V5_P2_B6_ONE_SHOT_TEST_EVALUATION_LOCK.json',
        'complete': b6 / 'V5_P2_B6_ONE_SHOT_TEST_EVALUATION_COMPLETE',
        'cache': b6 / 'V5_P2_B6_LOCKED_TEST_PREDICTIONS_AND_TARGETS.npz',
        'pair_manifest': b6 / 'V5_P2_B6_TEST_PAIR_WINDOW_MANIFEST.csv',
    }
    failures = [f'missing {name}: {path}' for name, path in paths.items() if not path.is_file()]
    if failures:
        write_json(out / f'{STAGE}.json', {'stage': STAGE, 'status': 'HOLD', 'failures': failures})
        atomic_write(out / f'{STAGE}_HOLD', f'{STAGE}_HOLD\n')
        return 1

    report = load_json(paths['report'])
    lock = load_json(paths['lock'])
    checks = [
        (report.get('status') == 'COMPLETE', 'B6 report is not COMPLETE'),
        (lock.get('report_sha256') == sha256_file(paths['report']), 'B6 report SHA mismatch'),
        (lock.get('test_prediction_cache_sha256') == sha256_file(paths['cache']), 'cache SHA mismatch'),
        (lock.get('test_pair_window_manifest_sha256') == sha256_file(paths['pair_manifest']), 'pair manifest SHA mismatch'),
        (lock.get('authorization_id') == EXPECTED_AUTH_ID, 'authorization ID changed'),
        (lock.get('selected_checkpoint_sha256') == EXPECTED_CKPT_SHA, 'checkpoint SHA changed'),
        (lock.get('threshold_manifest_sha256') == EXPECTED_THRESHOLD_SHA, 'threshold manifest SHA changed'),
        (lock.get('thresholds') == EXPECTED_THRESHOLDS, 'thresholds changed'),
        (lock.get('test_evaluation_count') == 1, 'test evaluation count is not one'),
        (lock.get('successful_rerun_authorized') is False, 'rerun unexpectedly authorized'),
    ]
    failures = [message for passed, message in checks if not passed]
    if failures:
        write_json(out / f'{STAGE}.json', {'stage': STAGE, 'status': 'HOLD', 'failures': failures})
        atomic_write(out / f'{STAGE}_HOLD', f'{STAGE}_HOLD\n')
        for f in failures:
            print('FAIL:', f)
        return 1

    artifact_rows = [
        {'artifact': name, 'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': sha256_file(path)}
        for name, path in paths.items()
    ]
    artifact_path = out / 'V5_P2_FINAL_B6_ARTIFACT_HASH_ARCHIVE.csv'
    write_csv(artifact_path, artifact_rows)

    pair_keys = read_pair_manifest(paths['pair_manifest'])
    with np.load(paths['cache'], allow_pickle=False) as cache:
        a = {key: cache[key] for key in cache.files}

    required = {'graph_truth','graph_score','graph_prediction','count_truth_raw','count_prediction_raw','pair_ordinal','window_start','mode'}
    for role in ROLES:
        required |= {f'{role}_truth', f'{role}_score', f'{role}_prediction_ungated', f'{role}_prediction_gated'}
    missing = required - set(a)
    if missing:
        raise RuntimeError(f'cache missing keys: {sorted(missing)}')

    n = len(a['graph_truth'])
    if n != EXPECTED_ITEMS:
        raise RuntimeError(f'cache items={n}, expected {EXPECTED_ITEMS}')
    for key in ('graph_truth','graph_score','graph_prediction','count_truth_raw','count_prediction_raw','pair_ordinal','window_start','mode'):
        if a[key].shape != (n,):
            raise RuntimeError(f'{key} has shape {a[key].shape}')
    for role in ROLES:
        for suffix in ('truth','score','prediction_ungated','prediction_gated'):
            if a[f'{role}_{suffix}'].shape != (n, EXPECTED_NODES):
                raise RuntimeError(f'{role}_{suffix} has wrong shape')

    y = a['graph_truth'].astype(np.uint8)
    score = a['graph_score'].astype(np.float64)
    pred = a['graph_prediction'].astype(np.uint8)
    count_y = a['count_truth_raw'].astype(np.int64)
    count_p = a['count_prediction_raw'].astype(np.int64)
    ordinals = a['pair_ordinal'].astype(np.int64)
    starts = a['window_start'].astype(np.int64)
    mode = a['mode'].astype(np.uint8)
    active = y == 1
    control = ~active
    graph_correct = y == pred
    count_correct = np.ones(n, dtype=bool)
    count_correct[active] = count_y[active] == count_p[active]

    role_exact_gated = {}
    role_exact_ungated = {}
    for role in ROLES:
        truth = a[f'{role}_truth'].astype(np.uint8)
        gated = a[f'{role}_prediction_gated'].astype(np.uint8)
        ungated = a[f'{role}_prediction_ungated'].astype(np.uint8)
        role_exact_gated[role] = (truth == gated).all(axis=1)
        role_exact_ungated[role] = (truth == ungated).all(axis=1)

    all_exact = graph_correct & count_correct
    for role in ROLES:
        all_exact &= role_exact_gated[role]

    overall_graph = metrics(y, pred)

    pair_rows = []
    for ordinal in range(EXPECTED_PAIRS):
        s = ordinals == ordinal
        sa = s & active
        sc = s & control
        gm = metrics(y[s], pred[s])
        row = {
            'pair_ordinal': ordinal, 'pair_key': pair_keys[ordinal],
            'item_count': int(s.sum()), 'attack_item_count': int(sa.sum()), 'control_item_count': int(sc.sum()),
            **{f'graph_{k}': v for k, v in gm.items()},
            'mean_attack_score': float(score[sa].mean()), 'mean_control_score': float(score[sc].mean()),
            'median_attack_score': float(np.median(score[sa])), 'median_control_score': float(np.median(score[sc])),
            'count_accuracy_attack': float(count_correct[sa].mean()),
            'all_task_exact_attack': float(all_exact[sa].mean()),
            'all_task_exact_control': float(all_exact[sc].mean()),
            'all_task_exact_all': float(all_exact[s].mean()),
        }
        for role in ROLES:
            rm = node_set_metrics(a[f'{role}_truth'][s], a[f'{role}_prediction_gated'][s], y[s].astype(bool))
            row[f'{role}_gated_node_f1'] = rm['node']['f1']
            row[f'{role}_gated_exact_attack'] = rm['exact_attack']
            row[f'{role}_gated_exact_control'] = rm['exact_control']
        pair_rows.append(row)
    pair_path = out / 'V5_P2_C0_PER_PAIR_FAILURE_SUMMARY.csv'
    write_csv(pair_path, pair_rows)

    error_rows = []
    for i in np.flatnonzero(~graph_correct):
        row = {
            'cache_index': int(i), 'pair_ordinal': int(ordinals[i]), 'pair_key': pair_keys[int(ordinals[i])],
            'window_start': int(starts[i]), 'mode': int(mode[i]), 'mode_name': 'ATTACK' if mode[i] else 'CONTROL',
            'graph_error_type': 'FN' if y[i] else 'FP', 'graph_truth': int(y[i]), 'graph_prediction': int(pred[i]),
            'graph_score': float(score[i]), 'attack_threshold': EXPECTED_THRESHOLDS['attack'],
            'count_truth_raw': int(count_y[i]), 'count_prediction_raw': int(count_p[i]),
            'count_correct': bool(count_correct[i]), 'all_task_exact': bool(all_exact[i]),
        }
        for role in ROLES:
            row[f'{role}_truth_nodes'] = nodes_string(a[f'{role}_truth'][i])
            row[f'{role}_gated_prediction_nodes'] = nodes_string(a[f'{role}_prediction_gated'][i])
            row[f'{role}_gated_exact'] = bool(role_exact_gated[role][i])
        error_rows.append(row)
    errors_path = out / 'V5_P2_C0_GRAPH_ERROR_WINDOWS.csv'
    write_csv(errors_path, error_rows)

    episode_rows = episodes(error_rows)
    episodes_path = out / 'V5_P2_C0_CONSECUTIVE_GRAPH_ERROR_EPISODES.csv'
    write_csv(episodes_path, episode_rows)

    k_rows = []
    for k in (1,2,3,4):
        s = active & (count_y == k)
        row = {
            'attacker_count': k, 'attack_window_count': int(s.sum()),
            'graph_recall': float(pred[s].mean()), 'graph_false_negative_count': int((pred[s] == 0).sum()),
            'mean_graph_score': float(score[s].mean()), 'median_graph_score': float(np.median(score[s])),
            'count_accuracy': float(count_correct[s].mean()), 'all_task_exact_accuracy': float(all_exact[s].mean()),
        }
        for role in ROLES:
            row[f'{role}_gated_exact'] = float(role_exact_gated[role][s].mean())
            row[f'{role}_ungated_exact'] = float(role_exact_ungated[role][s].mean())
        k_rows.append(row)
    k_path = out / 'V5_P2_C0_K1_K2_K3_K4_ATTACK_SUMMARY.csv'
    write_csv(k_path, k_rows)

    node_rows = []
    for role in ROLES:
        truth = a[f'{role}_truth'].astype(np.uint8)
        gated = a[f'{role}_prediction_gated'].astype(np.uint8)
        for node in range(EXPECTED_NODES):
            m = metrics(truth[:,node], gated[:,node])
            node_rows.append({'role': role, 'node': node, 'truth_positive_windows': int(truth[:,node].sum()),
                              'predicted_positive_windows': int(gated[:,node].sum()), **m})
    node_path = out / 'V5_P2_C0_PER_NODE_ROLE_METRICS.csv'
    write_csv(node_path, node_rows)

    score_summary = {
        'attack_scores': quantiles(score[active]), 'control_scores': quantiles(score[control]),
        'true_positive_scores': quantiles(score[active & (pred == 1)]),
        'false_negative_scores': quantiles(score[active & (pred == 0)]),
        'true_negative_scores': quantiles(score[control & (pred == 0)]),
        'false_positive_scores': quantiles(score[control & (pred == 1)]),
        'frozen_attack_threshold': EXPECTED_THRESHOLDS['attack'],
    }
    score_path = out / 'V5_P2_C0_GRAPH_SCORE_DISTRIBUTIONS.json'
    write_json(score_path, score_summary)

    hard_fp = sorted(pair_rows, key=lambda r: (-int(r['graph_fp']), -float(r['graph_fpr']), int(r['pair_ordinal'])))[:15]
    hard_fn = sorted(pair_rows, key=lambda r: (-int(r['graph_fn']), float(r['graph_recall']), int(r['pair_ordinal'])))[:15]
    hard_multi = sorted(pair_rows, key=lambda r: (float(r['all_task_exact_attack']), int(r['pair_ordinal'])))[:15]

    role_overall = {
        role: node_set_metrics(a[f'{role}_truth'], a[f'{role}_prediction_gated'], active)
        for role in ROLES
    }
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    result = {
        'stage': STAGE, 'status': 'COMPLETE',
        'decision': 'FREEZE_POSTTEST_FAILURE_ANALYSIS_FROM_B6_CACHE_ONLY',
        'overall': {
            'test_items': n, 'test_pairs': EXPECTED_PAIRS, 'test_runs': EXPECTED_RUNS,
            'graph': overall_graph, 'count_accuracy_attack': float(count_correct[active].mean()),
            'all_task_exact_all': float(all_exact.mean()), 'all_task_exact_attack': float(all_exact[active].mean()),
            'all_task_exact_control': float(all_exact[control].mean()), 'roles': role_overall,
        },
        'failure_inventory': {
            'false_positive_windows': fp, 'false_negative_windows': fn,
            'graph_error_windows': len(error_rows), 'graph_error_episodes': len(episode_rows),
            'pairs_with_at_least_one_false_positive': sum(int(r['graph_fp']) > 0 for r in pair_rows),
            'pairs_with_at_least_one_false_negative': sum(int(r['graph_fn']) > 0 for r in pair_rows),
        },
        'ranked_hard_cases': {
            'highest_false_positive_pairs': hard_fp,
            'highest_false_negative_pairs': hard_fn,
            'lowest_attack_all_task_exact_pairs': hard_multi,
            'longest_graph_error_episodes': episode_rows[:20],
        },
        'k1_k2_k3_k4': k_rows, 'graph_score_distributions': score_summary,
        'frozen_provenance': {
            'authorization_id': EXPECTED_AUTH_ID, 'selected_checkpoint_sha256': EXPECTED_CKPT_SHA,
            'threshold_manifest_sha256': EXPECTED_THRESHOLD_SHA, 'thresholds': EXPECTED_THRESHOLDS,
            'b6_report_sha256': sha256_file(paths['report']), 'b6_lock_sha256': sha256_file(paths['lock']),
            'prediction_cache_sha256': sha256_file(paths['cache']), 'pair_manifest_sha256': sha256_file(paths['pair_manifest']),
        },
        'security_boundary': {
            'model_loaded': False, 'checkpoint_loaded': False, 'dataset_root_accessed': False,
            'test_directory_enumerated': False, 'test_tensor_files_opened': False,
            'test_tensor_contents_accessed': False, 'test_inference_rerun': False,
            'analysis_source': 'B6 frozen prediction cache and pair-window manifest only',
        },
        'failures': [], 'warnings': [],
        'next_stage': 'V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION',
    }
    result_path = out / f'{STAGE}.json'
    write_json(result_path, result)

    fp_share = div(fp, fp + fn)
    md = [
        '# V5 P2-C0 Frozen Test-Cache Failure Analysis', '',
        '## Integrity boundary', '',
        '- Analysis source: frozen B6 prediction cache and pair-window manifest only.',
        '- Model/checkpoint loaded: **no**.',
        '- Original test tensors opened or test inference rerun: **no**.', '',
        '## Primary diagnosis', '',
        f'- False-positive control windows: **{fp}**.',
        f'- False-negative attack windows: **{fn}**.',
        f'- False positives are **{100*fp_share:.2f}%** of graph errors.',
        f'- Pairs with at least one FP: **{result["failure_inventory"]["pairs_with_at_least_one_false_positive"]}/69**.',
        f'- Pairs with at least one FN: **{result["failure_inventory"]["pairs_with_at_least_one_false_negative"]}/69**.', '',
        'The dominant failure is normal/control traffic being classified as attack.', '',
        '## Highest false-positive pairs', '',
        '| Rank | Pair | FP | Control FPR | Mean control score |',
        '|---:|---|---:|---:|---:|',
    ]
    for rank, row in enumerate(hard_fp[:10], 1):
        md.append(f'| {rank} | `{row["pair_key"]}` | {row["graph_fp"]} | {row["graph_fpr"]:.4f} | {row["mean_control_score"]:.4f} |')
    md += ['', '## K1-K4 attack behavior', '',
           '| Count | Windows | Graph recall | Count accuracy | All-task exact | Source exact | Transit exact | Victim exact | Path exact |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in k_rows:
        md.append(f'| K{row["attacker_count"]} | {row["attack_window_count"]} | {row["graph_recall"]:.4f} | {row["count_accuracy"]:.4f} | {row["all_task_exact_accuracy"]:.4f} | {row["source_gated_exact"]:.4f} | {row["transit_gated_exact"]:.4f} | {row["victim_gated_exact"]:.4f} | {row["path_gated_exact"]:.4f} |')
    md += ['', '## Rule', '',
           'Do not alter the P2 checkpoint or thresholds from this analysis. Use it only to design the next dataset/model revision.']
    md_path = out / 'V5_P2_C0_FROZEN_CACHE_FAILURE_ANALYSIS.md'
    atomic_write(md_path, '\n'.join(md) + '\n')

    artifact_outputs = {
        'artifact_hash_archive': artifact_path, 'per_pair_summary': pair_path,
        'graph_error_windows': errors_path, 'graph_error_episodes': episodes_path,
        'k_summary': k_path, 'per_node_role_metrics': node_path,
        'graph_score_distributions': score_path, 'analysis_markdown': md_path,
    }
    lock_out = {
        'status': COMPLETE, 'decision': 'FREEZE_POSTTEST_FAILURE_ANALYSIS_FROM_B6_CACHE_ONLY',
        'report_sha256': sha256_file(result_path),
        'artifact_sha256': {name: sha256_file(path) for name, path in artifact_outputs.items()},
        'prediction_cache_sha256': sha256_file(paths['cache']),
        'pair_manifest_sha256': sha256_file(paths['pair_manifest']),
        'test_inference_rerun': False, 'test_tensor_contents_accessed': False,
        'checkpoint_loaded': False,
        'next_stage': 'V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION',
        'script_sha256': sha256_file(Path(__file__)),
    }
    write_json(out / f'{STAGE}_LOCK.json', lock_out)
    atomic_write(out / COMPLETE, COMPLETE + '\n')

    print('===== V5 P2-C0 FROZEN-CACHE FAILURE ANALYSIS =====')
    print('status: COMPLETE')
    print('decision: FREEZE_POSTTEST_FAILURE_ANALYSIS_FROM_B6_CACHE_ONLY')
    print('false_positive_windows:', fp)
    print('false_negative_windows:', fn)
    print('false_positive_share_of_graph_errors:', fp_share)
    print('pairs_with_false_positive:', result['failure_inventory']['pairs_with_at_least_one_false_positive'])
    print('pairs_with_false_negative:', result['failure_inventory']['pairs_with_at_least_one_false_negative'])
    print('graph_accuracy:', overall_graph['accuracy'])
    print('graph_balanced_accuracy:', overall_graph['balanced_accuracy'])
    print('graph_precision:', overall_graph['precision'])
    print('graph_recall:', overall_graph['recall'])
    print('graph_f1:', overall_graph['f1'])
    print('graph_fpr:', overall_graph['fpr'])
    print('count_accuracy_attack:', float(count_correct[active].mean()))
    print('all_task_exact_attack:', float(all_exact[active].mean()))
    print('model_loaded: false')
    print('checkpoint_loaded: false')
    print('test_directory_enumerated: false')
    print('test_tensor_contents_accessed: false')
    print('test_inference_rerun: false')
    print('failure_count: 0')
    print('warning_count: 0')
    print('next_stage: V5_P2_C1_SCENARIO_AND_ROOT_CAUSE_INTERPRETATION')
    print(COMPLETE)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
