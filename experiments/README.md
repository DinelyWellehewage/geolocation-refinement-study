# Experiments

This directory contains development and ablation training runners used during model selection.

These files are not part of the final training/inference pipeline.

## Included experiments

- `main_cluster_country.py`
  - K=128 geographic cluster + auxiliary country validation training.

- `main_cluster_country_residual_xyz.py`
  - Joint spherical XYZ residual refinement experiments.

- `main_cluster_country_residual_xyz_frozen.py`
  - Frozen-baseline residual-head ablation.

The final submission pipeline is located in `src/` and uses:

- `src/main_cluster_country_final.py`
- `src/main_cluster_country_residual_xyz_final.py`
- `src/predict_holdout_residual_xyz.py`
