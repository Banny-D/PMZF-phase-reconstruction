# PMZF DHM Phase Reconstruction Demo

This repository contains the core code for digital holographic microscopy (DHM)-based phase reconstruction and Zernike-fitting-based phase correction used in our paper.

## Overview

The main script is `dhm.py`, which implements the DHM processing pipeline.

The file `globals.py` stores global variables and program settings.

The file `train.py` is provided for training the U-Net model. You may download the released dataset and use it for training. Before training, please modify the dataset file paths in the code according to your local environment.

A runnable demo is included in the `sample/` folder. It provides a wrapped phase image as input. Phase unwrapping and subsequent Zernike fitting are then performed to obtain the final phase results, which can be used to verify the PMZF and FMZF methods.

## Repository Structure

- `dhm.py` – main DHM processing script
- `globals.py` – global variables and settings
- `train.py` – U-Net training script
- `sample/` – demo files for PMZF and FMZF verification

## Dataset

The dataset for U-Net training can be downloaded from:

[Dataset Download Link](http://micronanorobot-of-bit.quickconnect.cn/d/s/17UGA4zZxkyfhJxWtdCQb8zZr0smm4Yl/ZJdy5kLRpHAyKKjErFvPaFSjspi-GksU-X75APTf6DQ0)

After downloading the dataset, please modify the file paths in `train.py` before running the training script.

## Quick Start

Install requirements: (python version = 3.9)

```bash
pip install -r requirements.txt
```

Run the main program with:

```bash
python dhm.py
```

To train the U-Net model, run:

```bash
python train.py
```

## Notes

- The `sample/` folder is intended for quick verification of the PMZF and FMZF results.
- The provided demo starts from a wrapped phase image rather than the raw hologram.
- Additional local path configuration may be required before running the code.

<!-- 
## Citation

If you use this code or dataset in your research, please cite our paper [*"Real-Time Holography Guided 3D Printing for Photocurable Hydrogel Microstructures with Tailored Mechanical and Morphological Properties"*](https://example.com). 
-->