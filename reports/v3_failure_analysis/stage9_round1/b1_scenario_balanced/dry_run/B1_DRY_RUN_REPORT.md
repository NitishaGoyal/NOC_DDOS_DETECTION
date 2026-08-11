# Stage 9 B1 Sampler Dry Run

## Verdict

**PASS**

## Intervention

The sole optimization change is graph-class-balanced,
class-conditional run-uniform sampling with replacement.

- Model seed: `7`
- Sampler seed: `7001`
- Samples per epoch: `148185`
- Batch size: `256`
- Batches per epoch: `579`
- Replacement: `True`

## Distribution

- Normal draws: `73991`
- Attack draws: `74194`
- Normal fraction: `0.499315`
- Attack fraction: `0.500685`
- Expected draws per normal run: `5292.321`
- Expected draws per attack run: `2390.081`

## Frozen historical objective interaction

B1 retains graph positive weight `0.451612903226` while sampling
approximately 50/50 graph classes. Ignoring prediction difficulty:

- normal contribution: `0.5 × 1.0 = 0.5`
- attack contribution: `0.5 × 0.451612903226 = 0.225806451613`

B1 is balanced sampling under the frozen historical loss, not a fully
balanced objective.

## Train metrics

`train_loader` is optimization-only. `train_eval_loader` traverses the
complete original training cohort without replacement for diagnostic train
metrics. Validation and test remain the comparison/model-selection sources.

## Safety

- GPU transfer performed: `False`
- Forward pass performed: `False`
- Backpropagation performed: `False`
- Optimizer update performed: `False`
- Checkpoint created: `False`
