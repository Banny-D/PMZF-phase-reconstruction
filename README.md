# PMZF DHM Phase Reconstruction Demo

This repository contains the core code for digital holographic microscopy (DHM)-based phase reconstruction and Zernike-fitting-based phase correction used in our paper.

## Related Paper

**[Real-Time Holography Guided 3D Printing for Photocurable Hydrogel Microstructures With Tailored Morphology and Stiffness](https://doi.org/10.1109/TMECH.2026.3694211)**

Xinyi Dong, Yanfeng Zhao, Kaijun Lin, Haotian Yang, Yaozhen Hou, Qing Shi, Qiang Huang, Toshio Fukuda, and Huaping Wang.

*IEEE/ASME Transactions on Mechatronics*, 2026. DOI: [10.1109/TMECH.2026.3694211](https://doi.org/10.1109/TMECH.2026.3694211).

The paper integrates DHM with digital light processing (DLP) to provide real-time feedback for printing photocurable hydrogel microstructures with controlled morphology and stiffness. Its partial matrix Zernike fitting (PMZF) method estimates optical phase distortion using only background pixels identified by U-Net segmentation. Compared with full matrix Zernike fitting (FMZF), which includes object pixels, PMZF reduces the influence of the object's phase on background distortion estimation. The paper reports phase reconstruction at 5 frames per second in the experimental system.

This repository provides the phase reconstruction and correction components of that workflow, including phase unwrapping, PMZF/FMZF comparison, and U-Net training. The wrapped-phase demo corresponds to the phase correction workflow described in Section II-B and Fig. 2 of the paper.

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

## Citation

If you use this code or dataset in your research, please cite the following paper:

```bibtex
@article{dong2026realtime,
  author  = {Dong, Xinyi and Zhao, Yanfeng and Lin, Kaijun and Yang, Haotian and Hou, Yaozhen and Shi, Qing and Huang, Qiang and Fukuda, Toshio and Wang, Huaping},
  title   = {Real-Time Holography Guided {3D} Printing for Photocurable Hydrogel Microstructures With Tailored Morphology and Stiffness},
  journal = {IEEE/ASME Transactions on Mechatronics},
  year    = {2026},
  doi     = {10.1109/TMECH.2026.3694211},
  url     = {https://doi.org/10.1109/TMECH.2026.3694211}
}
```
