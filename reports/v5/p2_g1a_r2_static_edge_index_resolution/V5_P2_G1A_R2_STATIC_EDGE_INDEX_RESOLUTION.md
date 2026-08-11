# V5 P2-G1A-R2 Static Edge-Index Resolution

- Status: **HOLD**
- Dataset class: `V5P2PairAlignedPrimary58Dataset`
- Train manifest: `/home/zira/research/projects/GNN-2d/reports/v5/p2_a1_r2_pair_aligned_window_contract/V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv`
- Validation manifest: `/home/zira/research/projects/GNN-2d/reports/v5/p2_a1_r2_pair_aligned_window_contract/V5_P2_A1_R2_PAIR_ALIGNED_WINDOW_MANIFEST.csv`
- Selected topology source: `None`

The P2 loader does not need to duplicate a fixed 4x4 topology in every sample. R2 resolves the static edge_index from the dataset, loader module, or a separate data-root artifact and freezes it for G1.

No P2 test access, checkpoint loading, training, threshold tuning, quantization, RTL generation, or Legal NoC decoder implementation was performed.
