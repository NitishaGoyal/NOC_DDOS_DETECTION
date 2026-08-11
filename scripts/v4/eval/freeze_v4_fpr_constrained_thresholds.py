#!/usr/bin/env python3
"""Freeze a validation-only graph threshold under an FPR cap, preserving node selection."""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def ensure_empty(path: Path) -> None:
    path.mkdir(parents=True,exist_ok=True)
    entries=list(path.iterdir())
    if entries: raise RuntimeError(f'Output directory must be empty: {path}; entries={[p.name for p in entries[:20]]}')


def numeric(row: dict[str,str]) -> dict[str,Any]:
    out={}
    for k,v in row.items():
        if k=='task': out[k]=v
        elif v not in ('',None): out[k]=float(v)
    return out


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-dir',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--fpr-cap',type=float,default=0.10)
    ap.add_argument('--experiment-designation',required=True)
    args=ap.parse_args()
    source=Path(args.validation_dir).resolve(); checkpoint=Path(args.checkpoint).resolve(); out=Path(args.out_dir).resolve()
    required=[source/'threshold_sweep.csv',source/'selected_thresholds.json',source/'validation_metrics.json',source/'validation_predictions.npz',checkpoint]
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError(missing)
    if not (0.0 < args.fpr_cap < 1.0): raise ValueError('--fpr-cap must be between 0 and 1')
    ensure_empty(out)
    source_selected=json.loads((source/'selected_thresholds.json').read_text())
    source_designation=source_selected.get('experiment_designation')
    if source_designation is not None and source_designation != args.experiment_designation:
        raise RuntimeError(f'Experiment designation mismatch: source={source_designation} cli={args.experiment_designation}')
    with (source/'threshold_sweep.csv').open(newline='',encoding='utf-8') as f:
        rows=[numeric(r) for r in csv.DictReader(f)]
    graph=[r for r in rows if r.get('task')=='graph' and r.get('fpr',1.0)<=args.fpr_cap]
    node=[r for r in rows if r.get('task')=='node']
    if not graph: raise RuntimeError(f'No graph threshold satisfies FPR <= {args.fpr_cap}')
    if not node: raise RuntimeError('No node threshold rows found')
    best_graph=max(graph,key=lambda r:(r['f1'],r['recall'],r['precision'],-abs(r['threshold']-0.5)))
    source_node_threshold=float(source_selected['node_threshold'])
    best_node=min(node,key=lambda r:abs(r['threshold']-source_node_threshold))
    checkpoint_hash=sha256_file(checkpoint)
    source_checkpoint_hash=source_selected.get('checkpoint_sha256')
    if source_checkpoint_hash and source_checkpoint_hash != checkpoint_hash:
        raise RuntimeError('Checkpoint hash differs from validation threshold-selection checkpoint')
    selected={
        **source_selected,
        'experiment_designation':args.experiment_designation,
        'checkpoint':str(checkpoint),
        'checkpoint_sha256':checkpoint_hash,
        'graph_threshold':float(best_graph['threshold']),
        'node_threshold':float(best_node['threshold']),
        'graph_fpr_constraint':float(args.fpr_cap),
        'graph_selection_rule':f'maximize validation graph F1 subject to validation FPR <= {args.fpr_cap:.6f}; tie max recall; tie max precision; tie closest to 0.50',
        'node_selection_rule':'preserve validation-selected maximum node F1 operating point',
        'split':'validation',
        'test_accessed':False,
        'supersedes_unconstrained_graph_threshold':float(source_selected['graph_threshold']),
    }
    metrics={
        'designation':'validation-only frozen FPR-constrained operating point',
        'graph':{k:v for k,v in best_graph.items() if k!='task'},
        'node':{k:v for k,v in best_node.items() if k!='task'},
        'selection':selected,
    }
    dump(out/'selected_thresholds.json',selected); dump(out/'validation_metrics.json',metrics)
    shutil.copy2(source/'threshold_sweep.csv',out/'threshold_sweep.csv')
    shutil.copy2(source/'validation_predictions.npz',out/'validation_predictions.npz')
    dump(out/'threshold_freeze_provenance.json',{
        'source_validation_directory':str(source),
        'source_selected_thresholds_sha256':sha256_file(source/'selected_thresholds.json'),
        'source_threshold_sweep_sha256':sha256_file(source/'threshold_sweep.csv'),
        'checkpoint_sha256':checkpoint_hash,
        'experiment_designation':args.experiment_designation,
        'fpr_cap':args.fpr_cap,
        'test_accessed':False,
        'manual_prediction_or_label_changes':False,
    })
    print('V4_FPR_CONSTRAINED_THRESHOLDS_FROZEN')
    print(json.dumps(metrics,indent=2))
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr); raise
