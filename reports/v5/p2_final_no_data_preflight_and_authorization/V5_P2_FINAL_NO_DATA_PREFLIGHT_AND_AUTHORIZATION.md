# V5 P2 Final No-Data Preflight and Authorization

## Status

- Status: **COMPLETE**
- Decision: **AUTHORIZE EXACTLY ONE V5 P2 Raw/A0/A1 blind-test execution**
- Authorization ID: `V5P2-caaea91e4eb6772ace0dc1b43b7e28c4`
- Ready for one-shot test: **true**

## Bound execution

- Evaluator: `/home/zira/research/projects/GNN-2d/scripts/v5/p2/evaluate_v5_p2_final_raw_a0_a1_one_shot.py`
- Evaluator SHA-256: `f6f061177b1621e3e7d8b3cbb32a401f83cdbf0488235125c62f3be17501ee44`
- Data-root text: `/home/zira/research/projects/GNN-2d/data/processed/v5/p2_complete558_dynamic_graph`
- Output directory: `/home/zira/research/projects/GNN-2d/reports/v5/p2_final_raw_a0_a1_one_shot_blind_evaluation`
- Launcher: `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_final_authorized_one_shot.sh`
- Launcher SHA-256: `569e38f315a2018c437cb73a22837cdfa610a3facb539c9bc6cc2d78d8dfe798`

## Authorization

- Contract: `/home/zira/research/projects/GNN-2d/artifacts/v5/p2_final_no_data_preflight_and_authorization/V5_P2_FINAL_ONE_SHOT_AUTHORIZATION.json`
- Contract SHA-256: `e0f62d2c83e77825ee3e39da351a8347cda1857eb4bc3f4295756d928bf9dfc1`
- Token: `/home/zira/research/projects/GNN-2d/artifacts/v5/p2_final_no_data_preflight_and_authorization/V5_P2_FINAL_ONE_SHOT_AUTHORIZATION_TOKEN.json`
- Token SHA-256: `2f374c0894f315a759c6ec08acb58c091c9ef185d6f7aef2670e19960e7d2186`
- Single-use: **true**
- Evaluation count: **1**
- Successful rerun authorized: **false**

## Official result groups

1. **Raw neural outputs**
2. **A0 lightweight decoder**
3. **A1 exact structured decoder**

A2 remains rejected and is not executed.

## No-data boundary

- All frozen hashes verified: **PASS**
- Evaluator AST audit: **PASS**
- Fresh source-only integration test: **PASS**
- Test path resolved: **false**
- Test directory existence checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Test evaluation performed: **false**

Once the token is consumed, any failure is irreversible and does not authorize a rerun.
