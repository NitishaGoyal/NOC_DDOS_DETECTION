# GNN-2d DS2 4×4 Port-Feature Reports

These files document the experiment after the previous aggregate-feature baseline.

## Files in this zip

- `ml_portfeat_ds2_4x4_results_report.md`  
  Main detailed report. Use this as the primary documentation.

- `ml_portfeat_ds2_4x4_status_summary.txt`  
  Short summary of the result.

- `ml_portfeat_ds2_4x4_artifact_index.md`  
  Where to find datasets, scripts, models, and logs.

- `ml_portfeat_ds2_4x4_next_steps.md`  
  What to do next, including fixed 0.50 threshold logs and ablations.

## Copy into Ubuntu project

After extracting this zip, run:

```bash
cd ~/research/projects/GNN-2d
mkdir -p reports

cp ~/Downloads/gnn2d_ds2_portfeat_reports/ml_portfeat_ds2_4x4_results_report.md reports/
cp ~/Downloads/gnn2d_ds2_portfeat_reports/ml_portfeat_ds2_4x4_status_summary.txt reports/
cp ~/Downloads/gnn2d_ds2_portfeat_reports/ml_portfeat_ds2_4x4_artifact_index.md reports/
cp ~/Downloads/gnn2d_ds2_portfeat_reports/ml_portfeat_ds2_4x4_next_steps.md reports/

ls -lh reports
```

If your browser extracts somewhere else, replace `~/Downloads/gnn2d_ds2_portfeat_reports/` with the actual extracted folder path.


Additional file in v2:

- `ml_portfeat_ds2_4x4_summary_json_snapshot.md`  
  Exact saved `summary.json` outputs for prototype and placement port-feature models.

Copy command:

```bash
cp ~/Downloads/gnn2d_ds2_portfeat_reports/ml_portfeat_ds2_4x4_summary_json_snapshot.md reports/
```
