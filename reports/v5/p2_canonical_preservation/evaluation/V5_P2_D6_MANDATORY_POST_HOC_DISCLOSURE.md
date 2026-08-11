# Mandatory P2 Post-Hoc Disclosure

## Required result label

**P2 Post-Hoc Test-Informed Numerical-Recovery Evaluation**

## Required disclosure

The original P2 blind one-shot evaluation did not produce a complete official
Raw/A0/A1 test result. A single authorized recovery execution completed neural
inference for 11,666/11,666 items and then aborted during A1 exact decoding
after at least 6,200 items because a machine-scale reported MIP-gap residue
exceeded the original 64-epsilon reporting gate.

The numerical-certification policy, certified decoder, and staged evaluator
were subsequently repaired and validated using validation and synthetic data
only. Any later P2 execution is therefore test-informed and post hoc. It is not
an untouched blind test and is not the original one-shot evaluation.

Raw, A0, and A1 results must be reported separately. Raw and A0 artifacts must
be committed immutably before A1 begins.
