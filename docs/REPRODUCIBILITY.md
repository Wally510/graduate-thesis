# Reproducibility and interpretation

## What this release verifies

- Public Python files are copied without algorithm edits; their original paths and SHA-256 hashes are recorded in `SOURCE_MANIFEST.json`.
- Python sources pass static syntax parsing. Selected sources were checked for common embedded credential patterns.
- The complete project and weight archives were validated during publication using file size and SHA-256.
- No training, inference, metric recalculation, checkpoint selection or new figure generation was performed for this source release.

These checks establish source and archive integrity, not scientific reproducibility or numerical equivalence across environments.

## Execution requirements

The core sources import Python packages including PyTorch, NumPy, SciPy, pandas and scikit-learn. A universal pinned environment has not been validated for this selection. Check individual imports and the relevant archived run environment; the archived Windows dependency copies are not a portable environment specification.

The scripts contain original cluster paths under `/data-ai/…`. Before running an experiment, bind all dataset roots, manifests, model source paths, checkpoints and output directories to the intended environment, preserving its protocol. Do not treat historical default checkpoint paths as a selection recommendation.

| Source family | Additional requirements beyond this Git checkout |
|---|---|
| Pretraining | Compatible packed data/manifests, the original schedule/configuration and suitable compute; the packing script imports the external `train_dual_view_virtual_r_alignment` module |
| Downstream | Dataset-specific readers/manifests, trained backbone, fold definitions; optional AnyPPG/CSFM/PulsePPG adapters are external to this selection |
| Reconstruction | Modules such as `evaluate_real_beat_reconstruction` and `train_phase8_maskadapter_frozen`, common training code, fixed mask manifests and adapter checkpoints |
| Denoising | Phase-base/refiner/token-dense modules dynamically loaded from external paths, matched training data and checkpoints |
| Evaluation | Corresponding training packages, run outputs and frozen evaluation manifests |

Restore the complete archive and inspect its run-specific dependency checks. If a required module, dataset or saved prediction head is still unavailable, that run remains blocked; do not replace it with a different head/checkpoint and report the result as a reproduction. Data access may require obtaining the dataset from its original provider under its terms.

## Scientific boundaries

- Original epoch-14 downstream models and epoch-10 reconstruction/denoising adaptations serve different purposes. Their checkpoints and metrics are not interchangeable.
- The reconstruction evaluator's eight within-beat regions are equal-width bins, not validated physiological landmarks. Its documented second-based missingness uses oracle beat segmentation; it does not demonstrate end-to-end recovery with R-peak detection after corruption.
- Native-protocol denoising comparisons use each model's own signal processing/coordinates. They do not automatically establish a fair absolute-RMSE ranking on identical raw tensors.
- Fixed-model inference-time interventions are not matched from-start retraining ablations. Some historical ep14 intervention packages are explicitly blocked by missing original fold heads and standardizers.
- The root project status files describe a historical state. The archive also contains later work. Use the corresponding run contract, inputs, completion evidence and numerical outputs together; avoid combining different snapshots into a single task count or leaderboard claim.
- FP-46 job11590 and FP-47/48/49 derivatives are marked `REJECTED_DO_NOT_USE` by the project. Archival presence is not endorsement. Cross-modal translation is not presented here as a completed validated result.

## Archive layout and restoration

The Git tree is a source browsing view. Original project paths are preserved inside the separate Releases. Restore result files before model weights, using the instructions in each Release's `RESTORE.md`. Directory links were not followed during archival. ZIP member contents are recoverable, but recreated ZIP container metadata and bytes may differ.

The copies in `tools/` are provided for inspection; execute the restore scripts in their complete extracted archive layout, where the necessary manifests and payloads are present.
