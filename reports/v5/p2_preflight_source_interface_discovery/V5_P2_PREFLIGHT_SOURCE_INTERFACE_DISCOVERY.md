# V5 P2 Preflight Source and Interface Discovery

## Status

- Discovery status: **DISCOVERY_COMPLETE**
- Source files scanned: **213**
- Ranked candidates retained: **42**
- Complete batch-key candidates: **7**
- Canonical-alias candidates: **4**
- Model output keys missing: **none**

## Frozen expected interfaces

### Loader batch

`x, physical_port_mask, role_mask, y_attack, y_attack_path, y_attacker_count, y_source, y_transit, y_victim`

Canonical aliases:

- `y_graph <- y_attack`
- `y_path <- y_attack_path`

### Model output

`attack_logits, count_logits, source_logits, transit_logits, victim_logits, path_logits`

## Highest-ranked source candidates

- Score **53** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_a4_loader_integration_tiny_overfit.py`
  - batch keys: x, physical_port_mask, role_mask, y_attack, y_attack_path, y_attacker_count, y_source, y_transit, y_victim
  - output keys: attack_logits, count_logits, source_logits, transit_logits, victim_logits, path_logits
  - aliases: {"y_path": "y_attack_path"}
- Score **53** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/train_v5_p2_b2_single_seed.py`
  - batch keys: x, physical_port_mask, y_attack, y_attack_path, y_attacker_count, y_source, y_transit, y_victim
  - output keys: attack_logits, count_logits, source_logits, transit_logits, victim_logits, path_logits
  - aliases: {"y_path": "y_attack_path"}
- Score **52** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/run_v5_p2_b0_r2_count_head_expansion_a4_repeat.py`
  - batch keys: x, physical_port_mask, role_mask, y_attack, y_attack_path, y_attacker_count, y_source, y_transit, y_victim
  - output keys: attack_logits, count_logits, source_logits, transit_logits, victim_logits, path_logits
  - aliases: {"y_path": "y_attack_path"}
- Score **47** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/tune_v5_p2_b4_validation_thresholds.py`
  - batch keys: x, physical_port_mask, y_attack, y_attack_path, y_attacker_count, y_source, y_transit, y_victim
  - output keys: attack_logits, count_logits, source_logits, transit_logits, victim_logits, path_logits
  - aliases: {}
- Score **46** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/evaluate_v5_p2_b6_one_shot_test.py`
  - batch keys: x, physical_port_mask, role_mask, y_attack, y_attack_path, y_attacker_count, y_source, y_transit, y_victim
  - output keys: attack_logits, count_logits, source_logits, transit_logits, victim_logits, path_logits
  - aliases: {}
- Score **46** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/train_v5_p2_g3_role_aware_multilabel_single_run.py`
  - batch keys: x, physical_port_mask, y_attack, y_attack_path, y_source, y_transit, y_victim
  - output keys: source_logits, transit_logits, victim_logits, path_logits
  - aliases: {"y_path": "y_attack_path"}
- Score **44** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/audit_v5_p2_b0_r3_corrected_nontest_label_shortcuts.py`
  - batch keys: x, physical_port_mask, role_mask, y_attack, y_attack_path, y_attacker_count, y_source, y_transit, y_victim
  - output keys: count_logits
  - aliases: {"y_graph": "y_attack", "y_path": "y_attack_path"}
- Score **42** — `/home/zira/research/projects/GNN-2d/scripts/v5/p2/audit_v5_p2_b0_nontest_label_shortcuts.py`
  - batch keys: x, physical_port_mask, role_mask, y_attack, y_attack_path, y_attacker_count, y_source, y_transit, y_victim
  - output keys: none
  - aliases: {"y_graph": "y_attack", "y_path": "y_attack_path"}

## Boundary

- Dataset builder imported: **false**
- Dataset builder executed: **false**
- Test path resolved: **false**
- Test directory existence checked: **false**
- Test directory enumerated: **false**
- Test tensors deserialized: **false**
- Evaluator source frozen: **false**
- Ready for one-shot test: **false**

The next stage uses this report to freeze one concrete loader callable and the
immutable one-shot evaluator source. It still performs source-only import and
interface checks before any blind-test path is opened.
