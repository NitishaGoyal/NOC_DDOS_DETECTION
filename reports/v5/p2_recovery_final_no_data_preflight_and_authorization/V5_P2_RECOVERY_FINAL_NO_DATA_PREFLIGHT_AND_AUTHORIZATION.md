# V5 P2 Recovery Final No-Data Preflight and Authorization

## Status

- Status: **COMPLETE**
- Decision: **AUTHORIZE EXACTLY ONE PROSPECTIVE V5 P2 RECOVERY Raw/A0/A1 EXECUTION**
- Recovery authorization ID: `V5P2-RECOVERY-1066fc8903ea19ba256b0924caf0be7b`
- Ready for recovery test: **true**

## Required disclosure

The original authorization `V5P2-caaea91e4eb6772ace0dc1b43b7e28c4` was consumed by a
pre-inference manifest-wiring abort. No original test result was obtained. This
new execution is a prospectively amended recovery and must be reported
separately.

## Bound execution

- Evaluator: `/home/zira/research/projects/GNN-2d/scripts/v5/p2/evaluate_v5_p2_recovery_raw_a0_a1.py`
- Evaluator SHA-256: `770ad8937427c32680c1cced7b3040bf6055662615c88edbf544e8805d9fe7c9`
- Recovery amendment SHA-256: `b2be725493603328bf7ae56a7ba0b8d6bdb3dc5aadc9ffdc52e3c3fc3cfde095`
- Data-root text: `/home/zira/research/projects/GNN-2d/data/processed/v5/p2_complete558_dynamic_graph`
- Output directory: `/home/zira/research/projects/GNN-2d/reports/v5/p2_recovery_raw_a0_a1_blind_evaluation`
- Launcher: `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_recovery_authorized_one_shot.sh`
- Launcher SHA-256: `6fe6a2e785a4ef82178e25252aba3d75bba63a4088d1a51f1ffdaf05b10a84c8`

## Authorization

- Contract: `/home/zira/research/projects/GNN-2d/artifacts/v5/p2_recovery_final_no_data_preflight_and_authorization/V5_P2_RECOVERY_ONE_SHOT_AUTHORIZATION.json`
- Contract SHA-256: `f51323d5496aa40dab5f73c504f79cdbb1e5b66789a46ed82d74fbf617840d7e`
- Token: `/home/zira/research/projects/GNN-2d/artifacts/v5/p2_recovery_final_no_data_preflight_and_authorization/V5_P2_RECOVERY_ONE_SHOT_AUTHORIZATION_TOKEN.json`
- Token SHA-256: `b0ae1535938594cf5e05e595f9c4b5dc52aa21ee102598b7ba5d54f906178f67`
- Single-use: **true**
- Evaluation count: **1**
- Successful rerun authorized: **false**

## No-data boundary

- Frozen hashes: **PASS**
- Nested recovery evaluator contract: **PASS**
- Fresh source-only integration test: **PASS**
- Real test path resolved: **false**
- Test directory checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Recovery evaluation performed: **false**

Once the recovery token is consumed, any failure is irreversible and does not
authorize another execution.
