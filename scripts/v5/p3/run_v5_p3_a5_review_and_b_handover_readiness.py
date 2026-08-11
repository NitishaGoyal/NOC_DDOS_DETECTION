from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, math, os, random, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

STAGE = 'V5_P3_A5_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_REVIEW_AND_B_HANDOVER_READINESS'
LABEL = 'V5-P3-1500-D70 Tranche-A Preliminary Diagnostic'
SEED = 107
EXPECTED_PARAMS = 60_553
EXPECTED_VAL_ITEMS = 13_863
TOLERANCE = 1e-3


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--data-link', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--installed-script', required=True)
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='mergesort'); sorted_values = values[order]
    out = np.empty(len(values), dtype=np.float64); start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]: end += 1
        out[order[start:end]] = (start + 1 + end) / 2.0; start = end
    return out


def auroc(scores, targets) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1); y = np.asarray(targets, dtype=np.int64).reshape(-1)
    pos = y == 1; neg = y == 0; np_, nn_ = int(pos.sum()), int(neg.sum())
    if np_ == 0 or nn_ == 0: return float('nan')
    return float((ranks(s)[pos].sum() - np_ * (np_ + 1) / 2.0) / (np_ * nn_))


def ap(scores, targets) -> float:
    s = np.asarray(scores, dtype=np.float64).reshape(-1); y = np.asarray(targets, dtype=np.int64).reshape(-1)
    npos = int((y == 1).sum())
    if npos == 0: return float('nan')
    order = np.argsort(-s, kind='mergesort'); yy = y[order]
    precision = np.cumsum(yy == 1) / np.arange(1, len(yy) + 1)
    return float(precision[yy == 1].sum() / npos)


def threshold_metrics(probabilities, targets) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1); y = np.asarray(targets, dtype=np.int64).reshape(-1)
    pred = (p >= 0.5).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0; recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'accuracy': (tp+tn)/max(1,tp+fp+fn+tn), 'precision': precision, 'recall': recall,
            'f1': f1, 'fpr': fp/(fp+tn) if fp+tn else 0.0, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def macro_f1(predictions, targets) -> float:
    pred = np.asarray(predictions); y = np.asarray(targets); values = []
    for c in (1,2,3,4):
        pp, yy = pred == c, y == c
        tp = int((pp & yy).sum()); fp = int((pp & ~yy).sum()); fn = int((~pp & yy).sum())
        precision = tp/(tp+fp) if tp+fp else 0.0; recall = tp/(tp+fn) if tp+fn else 0.0
        values.append(2*precision*recall/(precision+recall) if precision+recall else 0.0)
    return float(np.mean(values))


def exact_active(probabilities, targets, active) -> float:
    pred = (np.asarray(probabilities) >= 0.5).astype(np.int64); y = np.asarray(targets, dtype=np.int64)
    return float(np.mean(np.all(pred[active] == y[active], axis=1)))


def move(batch, device):
    mapping = {'x':('x',torch.float32), 'mask':('physical_port_mask',torch.float32),
               'graph':('y_attack',torch.float32), 'count':('y_attacker_count',torch.long),
               'source':('y_source',torch.float32), 'transit':('y_transit',torch.float32),
               'victim':('y_victim',torch.float32), 'path':('y_attack_path',torch.float32)}
    out = {}
    for dst,(src,dtype) in mapping.items():
        if src not in batch: raise RuntimeError(f'missing {src}; keys={sorted(batch)}')
        value = batch[src].to(device=device, dtype=dtype, non_blocking=device.type=='cuda')
        out[dst] = value.reshape(-1) if dst in ('graph','count') else value
    return out


def main() -> int:
    a = args(); seed_all(SEED)
    repo = Path(a.repo).expanduser().resolve(); link = Path(a.data_link).expanduser(); root = link.resolve()
    out = Path(a.output_dir).expanduser().resolve(); installed = Path(a.installed_script).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    report_path = out/f'{STAGE}_REPORT.json'; lock_path = out/f'{STAGE}_LOCK.json'
    stable_path = out/'V5_P3_A5_STABLE_BEST_CHECKPOINT_VALIDATION_METRICS.json'
    complete = out/f'{STAGE}_COMPLETE'; hold = out/f'{STAGE}_HOLD'

    a4 = repo/'reports/v5/p3_a4_tranche_a_preliminary_diagnostic_seed107'
    a4_report_path = a4/'V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_REPORT.json'
    a4_lock_path = a4/'V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_LOCK.json'
    checkpoint_path = a4/'V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_BEST.pt'
    history_path = a4/'V5_P3_A4_TRANCHE_A_PRELIMINARY_DIAGNOSTIC_SEED107_EPOCHS.csv'
    a3 = repo/'reports/v5/p3_a3_preliminary_baseline_preflight'
    protocol_path = a3/'V5_P3_TRANCHE_A_PRELIMINARY_BASELINE_PROTOCOL.json'
    wrapper_path = repo/'src/data/v5_p3_tranche_a_guarded_loader.py'
    b3_path = repo/'src/models/v5_p2_b3_conv1d_only_count4.py'
    canonical_path = repo/'src/models/v5_p2_task_d_full_multitask_count4.py'
    dynamic_path = repo/'src/models/v6_p0_dynamic70_task_d_full_multitask_count4.py'
    required = [a4_report_path,a4_lock_path,checkpoint_path,history_path,protocol_path,wrapper_path,b3_path,canonical_path,dynamic_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing: raise RuntimeError(f'missing required files: {missing}')
    if not link.is_symlink(): raise RuntimeError(f'canonical data path is not symlink: {link}')

    a4_report = json.loads(a4_report_path.read_text()); a4_lock = json.loads(a4_lock_path.read_text()); protocol = json.loads(protocol_path.read_text())
    if a4_report.get('status') != 'PASS': raise RuntimeError('A4 report not PASS')
    if a4_lock.get('report_sha256') != sha(a4_report_path): raise RuntimeError('A4 report SHA mismatch')
    if a4_lock.get('best_checkpoint_sha256') != sha(checkpoint_path): raise RuntimeError('A4 checkpoint SHA mismatch')
    if a4_lock.get('history_csv_sha256') != sha(history_path): raise RuntimeError('A4 history SHA mismatch')
    if a4_report.get('sealed_test',{}).get('test_tensor_loaded'): raise RuntimeError('A4 reports test access')

    original_load = torch.load; loaded = []
    def guarded_load(file,*aa,**kk):
        try: candidate = Path(os.fspath(file)).expanduser().resolve()
        except TypeError: candidate = None
        if candidate is not None:
            loaded.append(str(candidate))
            if '/runs/test/' in str(candidate): raise PermissionError(f'A5 blocked test deserialization: {candidate}')
        return original_load(file,*aa,**kk)
    torch.load = guarded_load

    wrapper = import_file(wrapper_path,'_a5_loader'); b3 = import_file(b3_path,'_a5_b3')
    canonical = import_file(canonical_path,'_a5_canonical'); dynamic = import_file(dynamic_path,'_a5_dynamic')
    Dataset = wrapper.GuardedV5P3TrancheAPreliminaryDataset
    try: Dataset(root,'test')
    except wrapper.SealedTestAccessError: test_guard = True
    else: test_guard = False
    if not test_guard: raise RuntimeError('test guard failed')

    ds = Dataset(root,'validation',active_only=False)
    if len(ds) != EXPECTED_VAL_ITEMS: raise RuntimeError(f'validation length={len(ds)}')
    loader = DataLoader(ds,batch_size=int(protocol['optimization']['batch_size']),shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    edge = torch.as_tensor(ds[0]['edge_index']).long()
    ref = b3.P2B3Conv1DOnlyCount4(); canon_model = canonical.P2TaskDGraphConvCount4(ref,edge)
    model = dynamic.build_v6_p0_dynamic70_from_canonical_structure(canon_model,seed=SEED)
    params = sum(p.numel() for p in model.parameters())
    if params != EXPECTED_PARAMS: raise RuntimeError(f'parameter count={params}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model = model.to(device)
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)
    if int(checkpoint['epoch']) != int(a4_report['training']['best_epoch']): raise RuntimeError('best epoch mismatch')
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()

    acc = {k:[] for k in ['attack_logits','graph_targets','count_logits','count_targets','source_logits','source_targets','transit_logits','transit_targets','victim_logits','victim_targets','path_logits','path_targets']}
    with torch.no_grad():
        for i,batch in enumerate(loader,1):
            b = move(batch,device); outp = model(b['x'],b['mask'])
            acc['attack_logits'].append(outp['attack_logits'].cpu().numpy()); acc['graph_targets'].append(b['graph'].cpu().numpy())
            acc['count_logits'].append(outp['count_logits'].cpu().numpy()); acc['count_targets'].append(b['count'].cpu().numpy())
            for role in ('source','transit','victim','path'):
                acc[f'{role}_logits'].append(outp[f'{role}_logits'].cpu().numpy()); acc[f'{role}_targets'].append(b[role].cpu().numpy())
            if i % 20 == 0: print(f'validation_batches_replayed={i}/{len(loader)}')
    acc = {k:np.concatenate(v) for k,v in acc.items()}
    graph_y = acc['graph_targets'].astype(np.int64); active = graph_y == 1
    graph_prob = torch.sigmoid(torch.from_numpy(acc['attack_logits'])).numpy()
    graph = threshold_metrics(graph_prob,graph_y); graph['auroc_raw_logits'] = auroc(acc['attack_logits'],graph_y); graph['average_precision_raw_logits'] = ap(acc['attack_logits'],graph_y)
    count_pred = np.argmax(acc['count_logits'][active],axis=1)+1; count_y = acc['count_targets'][active].astype(np.int64)
    count = {'accuracy':float(np.mean(count_pred==count_y)),'macro_f1':macro_f1(count_pred,count_y),'active_items':int(active.sum())}
    roles = {}
    for role in ('source','transit','victim','path'):
        lg = acc[f'{role}_logits']; yy = acc[f'{role}_targets'].astype(np.int64); prob = torch.sigmoid(torch.from_numpy(lg)).numpy()
        m = threshold_metrics(prob.reshape(-1),yy.reshape(-1)); m['auroc_raw_logits'] = auroc(lg.reshape(-1),yy.reshape(-1)); m['average_precision_raw_logits'] = ap(lg.reshape(-1),yy.reshape(-1)); m['exact_active'] = exact_active(prob,yy,active)
        m['logit_min'] = float(lg.min()); m['logit_max'] = float(lg.max()); m['abs_ge_100'] = int((np.abs(lg)>=100).sum()); roles[role] = m
    components = [graph['auroc_raw_logits'],graph['average_precision_raw_logits'],count['macro_f1']] + [roles[r]['average_precision_raw_logits'] for r in ('source','transit','victim','path')]
    stable_score = float(np.mean(components))

    old = a4_report['selection']['best_validation_metrics']
    pairs = {
      'selection_score':(float(old['selection_score']),stable_score),
      'graph_auroc':(float(old['graph']['auroc']),graph['auroc_raw_logits']),
      'graph_ap':(float(old['graph']['average_precision']),graph['average_precision_raw_logits']),
      'count_macro_f1':(float(old['count_active']['macro_f1']),count['macro_f1'])}
    for role in ('source','transit','victim','path'):
        pairs[f'{role}_ap']=(float(old['roles'][role]['average_precision']),roles[role]['average_precision_raw_logits'])
        pairs[f'{role}_exact_active']=(float(old['roles'][role]['exact_active']),roles[role]['exact_active'])
    comparison = {k:{'a4':a,'stable':b,'absolute_delta':abs(a-b)} for k,(a,b) in pairs.items()}
    max_delta = max(v['absolute_delta'] for v in comparison.values()); consistency = max_delta <= TOLERANCE

    stable = {'stage':STAGE,'campaign_label':LABEL,'best_epoch':int(checkpoint['epoch']),'validation_items':len(ds),'device':str(device),
              'metric_policy':{'ranking':'raw_logits','threshold_probabilities':'torch_sigmoid','threshold':0.5,'threshold_tuning':False},
              'selection_score':stable_score,'graph':graph,'count_active':count,'roles':roles,'comparison_to_a4':comparison,
              'maximum_absolute_metric_delta':max_delta,'tolerance':TOLERANCE,'numerical_consistency_pass':consistency}
    atomic_json(stable_path,stable)

    with history_path.open(newline='',encoding='utf-8') as f: history = list(csv.DictReader(f))
    best_epoch = int(a4_report['training']['best_epoch']); best_row = next(r for r in history if int(r['epoch'])==best_epoch); final_row = history[-1]
    transitions=[]; previous=None
    for row in history:
        lr=float(row['learning_rate'])
        if previous is None or lr != previous: transitions.append({'epoch':int(row['epoch']),'learning_rate':lr}); previous=lr
    post_best=[float(r['selection_score']) for r in history if int(r['epoch'])>best_epoch]
    strongest_post=max(post_best) if post_best else float('nan')
    test_paths=[p for p in loaded if '/runs/test/' in p]
    finite=all(math.isfinite(x) for x in components); passed=consistency and finite and not test_paths

    report={'stage':STAGE,'status':'PASS' if passed else 'HOLD','created_utc':datetime.now(timezone.utc).isoformat(),'campaign_label':LABEL,
            'interpretation':'Tranche-A seed-107 train/validation preliminary diagnostic review; no sealed-test or final 1500-pair claim.',
            'stable_replay':{'metrics_path':str(stable_path),'metrics_sha256':sha(stable_path),'selection_score':stable_score,'maximum_absolute_delta_vs_a4':max_delta,'numerical_consistency_pass':consistency},
            'history_review':{'epochs':len(history),'best_epoch':best_epoch,'best_row':best_row,'final_row':final_row,'strongest_post_best_score':strongest_post,'learning_rate_transitions':transitions,'later_epochs_exceeded_best':strongest_post>float(best_row['selection_score'])},
            'scientific_review':{'count_task':'near-saturated on A_validation','graph_task':'moderate ranking performance; fixed-threshold FPR remains material','source_localization':'strongest role AP','transit_localization':'weakest exact-active role localization','victim_localization':'high exact-active bitmap accuracy but moderate AP','path_localization':'moderate AP and exact-active accuracy','architecture_change_supported_by_A_alone':False,'additional_A_seed_sweep_supported_before_B':False},
            'sealed_test':{'negative_guard_check':test_guard,'test_dataset_instantiated':False,'test_length_computed':False,'test_tensor_loaded':False,'evaluation_authorized':False},
            'decision':{'tranche_a_review_complete':passed,'tranche_a_preliminary_result_usable':passed,'architecture_change_authorized':False,'additional_tranche_a_seed_sweep_authorized':False,'tranche_b_handover_preflight_authorized':passed,'tranche_b_generation_authorized':False,'sealed_test_evaluation_authorized':False,'next_stage':'V5_P3_A6_TRANCHE_B_PRE_FROZEN_SPEC_DISCOVERY_AND_HANDOVER_PREFLIGHT' if passed else 'A5_HOLD_REVIEW'},
            'provenance':{'a4_report_sha256':sha(a4_report_path),'a4_lock_sha256':sha(a4_lock_path),'best_checkpoint_sha256':sha(checkpoint_path),'history_csv_sha256':sha(history_path),'protocol_sha256':sha(protocol_path),'guarded_loader_sha256':sha(wrapper_path),'dynamic70_model_sha256':sha(dynamic_path),'installed_script_sha256':sha(installed)},
            'source_modified':True,'certified_dataset_modified':False,'model_trained':False,'validation_replayed':True,'final_1500_pair_claim_authorized':False}
    atomic_json(report_path,report)
    lock={'stage':STAGE,'status':report['status'],'report_sha256':sha(report_path),'stable_metrics_sha256':sha(stable_path),'best_checkpoint_sha256':sha(checkpoint_path),'history_csv_sha256':sha(history_path),'stable_selection_score':stable_score,'maximum_absolute_metric_delta':max_delta,'best_epoch':best_epoch,'parameter_count':params,'input_features':70,'test_tensor_loaded':False}
    atomic_json(lock_path,lock)
    (complete if passed else hold).write_text(f'{STAGE}_{"COMPLETE" if passed else "HOLD"}\n')
    if passed and hold.exists(): hold.unlink()

    print(f'{STAGE}_{"COMPLETE" if passed else "HOLD"}')
    print(f'status={report["status"]}'); print(f'best_epoch={best_epoch}'); print(f'stable_validation_selection_score={stable_score:.8f}')
    print(f'maximum_absolute_metric_delta_vs_a4={max_delta:.10f}'); print(f'numerical_consistency_pass={consistency}')
    print(f'stable_graph_auroc={graph["auroc_raw_logits"]:.8f}'); print(f'stable_graph_average_precision={graph["average_precision_raw_logits"]:.8f}')
    print(f'stable_graph_f1_at_0p5={graph["f1"]:.8f}'); print(f'stable_graph_fpr_at_0p5={graph["fpr"]:.8f}'); print(f'stable_count_active_macro_f1={count["macro_f1"]:.8f}')
    for role in ('source','transit','victim','path'):
        print(f'stable_{role}_average_precision={roles[role]["average_precision_raw_logits"]:.8f}'); print(f'stable_{role}_exact_active_at_0p5={roles[role]["exact_active"]:.8f}')
    print('learning_rate_transitions='+json.dumps(transitions)); print('architecture_change_authorized=false'); print('additional_tranche_a_seed_sweep_authorized=false')
    print(f'tranche_b_handover_preflight_authorized={passed}'); print('tranche_b_generation_authorized=false'); print('test_tensor_loaded=false'); print('sealed_test_evaluation_authorized=false')
    print('next_stage='+report['decision']['next_stage']); print(f'stable_metrics={stable_path}'); print(f'report={report_path}'); print(f'lock={lock_path}')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
