# Source guide

This is a curated source snapshot for browsing, not an independently validated end-to-end distribution. The Python files are byte-identical to their originals. `../SOURCE_MANIFEST.json` records where each file came from in the complete project archive.

| Directory | Source family | Scope |
|---|---|---|
| `pretraining/` | `fp53/pretrain_hparam_audit/foundation_stage2_ppg2ecg_aux_bundle_20260701-26` | Archived merged model, dual-view packing/training, Stage 2 DDP reconstruction pretraining |
| `downstream/` | `fp53/code` | BUT-PPG/general downstream wrappers, new4 benchmark, direct BP implementation |
| `reconstruction/` | `server_packages/shared_evaluation/splitenc8_phase8_vtac_external_eval_20260725` | Phase8 target-beat and long-gap external evaluation |
| `denoising/` | `result_review/splitenc8_token_dense_denoising_fullfinetune_20260729` | Denoising training and its DDP helper |
| `evaluation/` | `server_packages/shared_evaluation/splitenc8_token_dense_vtac_all_epochs_recon_denoise_20260730` | Epoch-wise evaluation, aggregation and output audits |

Suggested reading order: model definitions → data preparation → training loop → downstream benchmarks → reconstruction/denoising evaluators. The similarly named model copies are historical implementations with different interfaces; do not interchange them solely by class name.

The pretraining model file is the archived counterpart of the module filename referenced by the reconstruction evaluator. This release does not independently prove that the local source hash equals the source used for every historical checkpoint. Consult the run-specific provenance before reproducing a result.

No cluster submission package is generated here. Historical server launchers and full package structure remain in the project Release. This source selection does not include rejected FP-46/47/48/49 experimental branches or vendored third-party model repositories.

See [reproducibility notes](../docs/REPRODUCIBILITY.md) before execution.
