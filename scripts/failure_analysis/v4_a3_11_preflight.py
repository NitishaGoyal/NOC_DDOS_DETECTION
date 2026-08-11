#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

REQ = {
    'graph_prob','node_prob','count_prob','y_graph','y_node','attacker_count',
    'global_index','run_index','split_id','attack_kind_id','profile_id'
}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

def div(a,b): return float(a/b) if b else 0.0

def bin_metrics(y,p):
    y=np.asarray(y,dtype=np.int8).ravel(); p=np.asarray(p,dtype=np.int8).ravel()
    tp=int(np.sum((y==1)&(p==1))); tn=int(np.sum((y==0)&(p==0)))
    fp=int(np.sum((y==0)&(p==1))); fn=int(np.sum((y==1)&(p==0)))
    prec=div(tp,tp+fp); rec=div(tp,tp+fn)
    return {'accuracy':div(tp+tn,tp+tn+fp+fn),'precision':prec,'recall':rec,
            'f1':div(2*prec*rec,prec+rec),'fpr':div(fp,fp+tn),'tnr':div(tn,tn+fp),
            'tn':tn,'fp':fp,'fn':fn,'tp':tp}

def exact(y,p,mask=None):
    ok=np.all(np.asarray(y,dtype=np.int8)==np.asarray(p,dtype=np.int8),axis=1)
    if mask is not None: ok=ok[np.asarray(mask,dtype=bool)]
    return float(np.mean(ok)) if ok.size else 0.0

def decode_topk(node_prob,count_prob):
    k=np.argmax(count_prob,axis=1).astype(np.int64)
    pred=np.zeros_like(node_prob,dtype=np.int8)
    order=np.argsort(-node_prob,axis=1,kind='stable')
    for kk in range(1,5):
        rows=np.flatnonzero(k==kk)
        if rows.size:
            pred[rows[:,None],order[rows,:kk]]=1
    return pred,k

def loc_metrics(y_graph,y_node,pred,true_count,pred_count):
    attack=np.asarray(y_graph,dtype=np.int8)==1
    overall=bin_metrics(y_node,pred); atk=bin_metrics(y_node[attack],pred[attack])
    normal=~attack
    return {
        'node_precision':overall['precision'],'node_recall':overall['recall'],'node_f1':overall['f1'],
        'node_tn':overall['tn'],'node_fp':overall['fp'],'node_fn':overall['fn'],'node_tp':overall['tp'],
        'attack_node_precision':atk['precision'],'attack_node_recall':atk['recall'],'attack_node_f1':atk['f1'],
        'attack_node_tn':atk['tn'],'attack_node_fp':atk['fp'],'attack_node_fn':atk['fn'],'attack_node_tp':atk['tp'],
        'exact_localization':exact(y_node,pred),
        'attack_exact_localization':exact(y_node,pred,attack),
        'count_accuracy':float(np.mean(np.asarray(true_count)==np.asarray(pred_count))),
        'count_mae':float(np.mean(np.abs(np.asarray(true_count)-np.asarray(pred_count)))),
        'attack_count_accuracy':float(np.mean(np.asarray(true_count)[attack]==np.asarray(pred_count)[attack])),
        'attack_count_mae':float(np.mean(np.abs(np.asarray(true_count)[attack]-np.asarray(pred_count)[attack]))),
        'attack_empty_prediction_rate':float(np.mean(np.sum(pred[attack],axis=1)==0)),
        'normal_false_isolation_rate':float(np.mean(np.sum(pred[normal],axis=1)>0)),
        'false_routers_per_attack_window':div(atk['fp'],int(np.sum(attack))),
        'missed_attackers_per_attack_window':div(atk['fn'],int(np.sum(attack))),
        'false_routers_per_normal_window':div(int(np.sum(pred[normal])),int(np.sum(normal))),
    }

def load_npz(path: Path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.array(z[k]) for k in z.files}

def audit_split(name,path,ds,fail):
    a=load_npz(path); missing=sorted(REQ-set(a))
    if missing: fail.append(f'{name}: missing keys {missing}')
    n=len(a['global_index']); gi=a['global_index'].astype(np.int64)
    for k in REQ:
        if k in a and a[k].shape[0]!=n: fail.append(f'{name}: row mismatch {k}')
    if a['graph_prob'].shape!=(n,): fail.append(f'{name}: graph_prob shape {a["graph_prob"].shape}')
    if a['node_prob'].shape!=(n,16): fail.append(f'{name}: node_prob shape {a["node_prob"].shape}')
    if a['count_prob'].shape!=(n,5): fail.append(f'{name}: count_prob shape {a["count_prob"].shape}')
    for k in ('graph_prob','node_prob','count_prob'):
        x=a[k]
        if not np.all(np.isfinite(x)): fail.append(f'{name}: nonfinite {k}')
        if float(np.min(x)) < -1e-7 or float(np.max(x)) > 1+1e-7: fail.append(f'{name}: bad range {k}')
    if float(np.max(np.abs(np.sum(a['count_prob'],axis=1)-1)))>1e-4: fail.append(f'{name}: count_prob rows not normalized')
    total=len(ds['run_index'])
    if np.any(gi<0) or np.any(gi>=total): fail.append(f'{name}: global_index out of bounds')
    if np.unique(gi).size!=gi.size: fail.append(f'{name}: duplicate global_index')
    checks=(('run_index',np.int64),('y_graph',np.int64),('y_node',np.float32),('attacker_count',np.int64),('attack_kind_id',np.int64))
    for k,dtype in checks:
        if not np.array_equal(a[k].astype(dtype),ds[k][gi].astype(dtype)): fail.append(f'{name}: {k} alignment mismatch')
    epoch=np.asarray(ds['end_epoch'][gi],dtype=np.int64)
    order=np.lexsort((epoch,a['run_index'].astype(np.int64)))
    r=a['run_index'].astype(np.int64)[order]; e=epoch[order]
    same=r[1:]==r[:-1]; gaps=np.diff(e)[same]
    nonunit=int(np.sum(gaps!=1)); nonpos=int(np.sum(gaps<=0))
    if nonpos: fail.append(f'{name}: {nonpos} non-increasing within-run epochs')
    if nonunit: fail.append(f'{name}: {nonunit} non-unit within-run epoch gaps')
    runs,counts=np.unique(a['run_index'],return_counts=True)
    summary={'samples':n,'runs':int(runs.size),'min_samples_per_run':int(counts.min()),'max_samples_per_run':int(counts.max()),
             'split_ids':sorted(map(int,np.unique(a['split_id']))),'attack_kind_ids':sorted(map(int,np.unique(a['attack_kind_id']))),
             'profile_ids':sorted(map(int,np.unique(a['profile_id']))),'global_index_min':int(gi.min()),'global_index_max':int(gi.max()),
             'end_epoch_min':int(epoch.min()),'end_epoch_max':int(epoch.max()),'nonunit_gaps':nonunit}
    return a,summary

def h1(a,gthr,nthr):
    yg=a['y_graph'].astype(np.int8); yn=a['y_node'].astype(np.int8); tc=a['attacker_count'].astype(np.int64)
    gp=(a['graph_prob']>=gthr).astype(np.int8)
    nt=(a['node_prob']>=nthr).astype(np.int8)
    topk,k=decode_topk(a['node_prob'],a['count_prob'])
    return {'sample_count':int(len(yg)),'attack_samples':int(np.sum(yg==1)),'normal_samples':int(np.sum(yg==0)),
            'graph_threshold':gthr,'node_threshold':nthr,'graph':bin_metrics(yg,gp),
            'node_threshold_decoder':loc_metrics(yg,yn,nt,tc,np.sum(nt,axis=1)),
            'count_conditioned_topk':loc_metrics(yg,yn,topk,tc,k)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--validation-predictions',type=Path,required=True); p.add_argument('--test-predictions',type=Path,required=True)
    p.add_argument('--thresholds',type=Path,required=True); p.add_argument('--test-metrics',type=Path,required=True)
    p.add_argument('--test-lock',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--expected-checkpoint-sha256',default='f61c1add6c057f7f53dd34fb1f9f4e95b01cefd5053e0e42bc100c76dac7f923')
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    fail=[]; warn=[]
    for x in (a.data_dir,a.checkpoint,a.validation_predictions,a.test_predictions,a.thresholds,a.test_metrics,a.test_lock):
        if not x.exists(): fail.append(f'missing: {x}')
    if fail:
        print(json.dumps({'status':'FAIL','failures':fail},indent=2)); return 2
    ck=sha256(a.checkpoint); th=sha256(a.thresholds); vp=sha256(a.validation_predictions); tp=sha256(a.test_predictions); tm=sha256(a.test_metrics)
    thresholds=json.loads(a.thresholds.read_text()); metrics=json.loads(a.test_metrics.read_text()); lock=json.loads(a.test_lock.read_text())
    if ck!=a.expected_checkpoint_sha256: fail.append('checkpoint sha256 mismatch')
    if thresholds.get('checkpoint_sha256')!=ck: fail.append('threshold checkpoint mismatch')
    if thresholds.get('selection_split')!='validation': fail.append('selection_split is not validation')
    if thresholds.get('test_accessed') is not False: fail.append('threshold package says test accessed')
    if metrics.get('thresholds_selected_on_test') is not False: fail.append('test metrics says thresholds selected on test')
    if lock.get('checkpoint_sha256')!=ck: fail.append('test lock checkpoint mismatch')
    if lock.get('test_predictions_sha256')!=tp: fail.append('test lock predictions mismatch')
    if lock.get('test_metrics_sha256')!=tm: fail.append('test lock metrics mismatch')
    if lock.get('threshold_package_sha256')!=th: fail.append('test lock threshold mismatch')
    ds={}
    for k in ('run_index','end_epoch','y_graph','y_node','attacker_count','strength','attack_kind_id'):
        f=a.data_dir/f'{k}.npy'
        if not f.is_file(): fail.append(f'missing dataset array: {f}')
        else: ds[k]=np.load(f,mmap_mode='r',allow_pickle=False)
    if fail:
        report={'status':'FAIL','failures':fail,'warnings':warn}; (a.output_dir/'preflight_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); return 1
    val,vsum=audit_split('validation',a.validation_predictions,ds,fail)
    test,tsum=audit_split('development_test',a.test_predictions,ds,fail)
    overlap=int(np.intersect1d(val['global_index'],test['global_index']).size)
    if overlap: fail.append(f'validation/test global_index overlap={overlap}')
    gthr=float(thresholds['graph_threshold']); nthr=float(thresholds['node_threshold'])
    h1v=h1(val,gthr,nthr); h1t=h1(test,gthr,nthr)
    prov={'checkpoint':str(a.checkpoint),'checkpoint_sha256':ck,'thresholds_sha256':th,'validation_predictions_sha256':vp,
          'test_predictions_sha256':tp,'test_metrics_sha256':tm,'graph_threshold':gthr,'node_threshold':nthr}
    report={'status':'PASS' if not fail else 'FAIL','failures':fail,'warnings':warn,'provenance':prov,
            'alignment':{'validation':vsum,'development_test':tsum,'validation_test_global_index_overlap':overlap},
            'h1_reproduction':{'validation':h1v,'development_test':h1t},
            'notes':['Prediction archives contain probabilities, not raw logits.',
                     'Graph/node logits can be reconstructed with clipped inverse sigmoid.',
                     'For count evidence use mean probability and mean log-probability; original softmax logits are not uniquely recoverable.',
                     'No aggregation policy was selected in this preflight.']}
    (a.output_dir/'preflight_report.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    (a.output_dir/'h1_reproduction_metrics.json').write_text(json.dumps(report['h1_reproduction'],indent=2,sort_keys=True))
    (a.output_dir/'input_provenance.json').write_text(json.dumps(prov,indent=2,sort_keys=True))
    print(json.dumps(report,indent=2,sort_keys=True))
    if fail:
        print('V4_A3_11_PREFLIGHT_FAIL',file=sys.stderr); return 1
    (a.output_dir/'V4_A3_11_PREFLIGHT_PASS').write_text('V4_A3_11_PREFLIGHT_PASS\n')
    print('V4_A3_11_PREFLIGHT_PASS'); return 0

if __name__=='__main__': raise SystemExit(main())
