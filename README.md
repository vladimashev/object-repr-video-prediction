# Evaluating Image Representations for Video Prediction

Transformer-based video frame prediction on **MOVi-C**, comparing patch-based RGB representations with object-centric mask-based representations.

---

## Overview

The project studies how different image representations affect future-frame prediction.

The pipeline consists of two stages:

1. an **autoencoder** compresses video frames into latent representations;
2. a **Transformer predictor** models their temporal dynamics and predicts future representations, which are decoded back into RGB frames.

We compare:

- **Patch-based representation** — ViT-style patch encoder/decoder;
- **Object-centric representation** — masked object slots with Transformer refinement;
- **Auto-regressive (AR)** and **Teacher Forcing (TF)** predictor training.

---

## Architecture

### Patch-based Autoencoder

Frames are split into patches, encoded as Transformer tokens, and reconstructed with a ViT-style decoder.

<!-- Architecture figure: Patch-based Autoencoder -->
<p align="center">
  <img src="assets/PatchAE.png" alt="Patch-based autoencoder architecture" width="800">
</p>

### Object-centric Autoencoder

Instance masks isolate individual objects. Each object is encoded into a slot representation, refined with self-attention, and decoded back into the full frame.

<!-- Architecture figure: Object-centric Autoencoder -->
<p align="center">
  <img src="assets/MAE_scheme.png" alt="Object-centric autoencoder architecture" width="800">
</p>

### Spatial-Temporal Predictor

The predictor receives embeddings from five seed frames and applies spatial and temporal attention to predict future latent representations.

<!-- Architecture figure: Predictor -->
<p align="center">
  <img src="assets/Predictor.png" alt="Spatial-temporal predictor architecture" width="750">
</p>

---

## Dataset

Experiments are performed on **MOVi-C**, a synthetic object-centric video dataset containing moving 3D-scanned objects on realistic backgrounds.

For this project, RGB frames are used by both approaches, while instance segmentation masks are additionally used by the object-centric model.

---

## Results

The best reconstruction models were:

- **Patch Autoencoder:** embedding dimension `256`;
- **Object-centric Autoencoder:** embedding dimension `512`.

For video prediction, four configurations were evaluated:

| Model | MSE ↓ | SSIM ↓ | PSNR ↑ | LPIPS ↓ | FVD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Pred Obj AR** | **0.083** | 0.415 | **19.083** | 0.542 | 79.676 |
| Pred Obj TF | 0.088 | 0.409 | 18.454 | 0.457 | 68.932 |
| Pred RGB AR | 0.121 | **0.421** | 16.841 | 0.514 | 73.307 |
| Pred RGB TF | 0.145 | 0.376 | 15.607 | **0.447** | **63.237** |

Main observations:

- **Object-centric AR** achieves the best pixel-level accuracy.
- **RGB AR** obtains the highest SSIM.
- **RGB TF** achieves the best LPIPS and FVD.
- Object-centric representations preserve object motion and identity more reliably over long rollouts.

---

## Qualitative Results

### Auto-Regressive Rollout

<table>
  <tr>
    <th align="center">Ground Truth</th>
    <th align="center">Patch Representation</th>
    <th align="center">Object-Centric Representation</th>
  </tr>
  <tr>
    <td align="center">
      <img src="results/Pred_RGB_AR/validation_000008/rollout_gt_000000.gif" alt="Ground-truth sequence" width="280">
    </td>
    <td align="center">
      <img src="results/Pred_RGB_AR/validation_000008/rollout_val_000000.gif" alt="Auto-Regressive rollout with patch representation" width="280">
    </td>
    <td align="center">
      <img src="results/Pred_Obj_AR/validation_000008/rollout_gt_000000.gif" alt="Auto-Regressive rollout with object-centric representation" width="280">
    </td>
  </tr>
</table>

---

### Teacher-Forcing Rollout

<table>
  <tr>
    <th align="center">Ground Truth</th>
    <th align="center">Patch Representation</th>
    <th align="center">Object-Centric Representation</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/rollout_tf_gt.gif" alt="Ground-truth sequence" width="280">
    </td>
    <td align="center">
      <img src="assets/rollout_tf_patch.gif" alt="Teacher-Forcing rollout with patch representation" width="280">
    </td>
    <td align="center">
      <img src="assets/rollout_tf_obj.gif" alt="Teacher-Forcing rollout with object-centric representation" width="280">
    </td>
  </tr>
</table>

---
### Teacher-Forcing Rollout

<table>
  <tr>
    <th align="center">Patch Representation</th>
    <th align="center">Object-Centric Representation</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/rollout_tf_patch.gif" alt="Teacher-Forcing rollout with patch representation" width="320">
    </td>
    <td align="center">
      <img src="assets/rollout_tf_obj.gif" alt="Teacher-Forcing rollout with object-centric representation" width="320">
    </td>
  </tr>
</table>

---

## Key Findings

Object-centric representations provide a stronger inductive bias for tracking objects and maintaining motion. RGB-based models remain competitive on perceptual metrics, but their dynamics tend to weaken during longer rollouts.

The experiments also show that pixel accuracy, structural similarity, and perceptual video quality do not necessarily favor the same model.

---


## Results

Qualitative results of video prediction are available in the `results/` folder

Interactive demos of training/evaluation are available in the notebooks:

- `test_reconstruction.ipynb` — reconstruction demo for the autoencoders (bothe patch and mask)
- `test.ipynb` — predictor training demo (Obj/RGB × TF/AR), Patch Autoencoder training
