# VisionGuard

Code for **"VisionGuard: Evaluating the Adversarial Robustness of Post-Training Quantised Vision Transformers and ConvNeXt Models"**.

It measures how FP16 and INT8 post-training quantisation change the adversarial vulnerability of ViT-S/16 and ConvNeXt-T on CIFAR-100 and Tiny-ImageNet, using FGSM and PGD (ℓ∞) in three threat models (A0 FP32 white-box, A1 quantised white-box, A2 transfer from FP32), with gradient-masking diagnostics, tipping points and latency.

## Contents
| File | Purpose |
|---|---|
| `visionguard.py` | The experiment library: fine-tuning, INT8 PTQ (FX graph mode, oneDNN), attacks, diagnostics, efficiency, analysis |
| `VisionGuard_experiments.ipynb` | Colab notebook that configures and runs the study end to end |
| `requirements.txt` | Package versions used for the reported results |

## How to run
1. Open `VisionGuard_experiments.ipynb` in Google Colab with a GPU runtime (an A100 was used).
2. Run Sections 1–2 (set `MODE = "quick"` first for a dry run), then Section 3.
3. Section 5 builds the tables and plots; Section 6 runs the FP16-underflow and APGD checks.

All results, fine-tuned weights and calibrated quantisation scales are saved to Google Drive after every unit of work, so interrupted runs resume where they stopped. Per-sample predictions for every attack are stored as `.npz`, so every statistic can be recomputed without re-attacking.

## Notes
* INT8 runs real integer kernels on CPU through PyTorch FX graph mode (oneDNN backend); gradients for A1 come from a fake-quantised proxy that shares scales with the integer model.
* FP16 white-box attacks use loss scaling (`FP16_LOSS_SCALE`), because unscaled half-precision gradients underflow and the attack silently fails.

## Citation
<!-- add the paper reference here after publication -->
