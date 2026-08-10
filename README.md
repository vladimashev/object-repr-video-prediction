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

## Results

Qualitative results of video prediction are available in the `results/` folder

Interactive demos of training/evaluation are available in the notebooks:

- `test_reconstruction.ipynb` — reconstruction demo for the autoencoders (bothe patch and mask)
- `test.ipynb` — predictor training demo (Obj/RGB × TF/AR), Patch Autoencoder training
