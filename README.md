# Frequency-domain augmentation for cross-device OCT intraretinal fluid segmentation

Code, frozen-file hashes and trained model weights for the preregistered study:

> X. Yang, G. Yue, G. Zhou. Frequency-domain augmentation for cross-device intraretinal fluid segmentation on optical coherence tomography: a preregistered single-source study. *Submitted to Biomedical Signal Processing and Control*, 2026.

- Preregistration: Open Science Framework, [osf.io/peums](https://osf.io/peums) (registered 12 November 2026; embargoed until 28 February 2027).
- Archived release: Zenodo, DOI_PLACEHOLDER (code) and DOI_WEIGHTS_PLACEHOLDER (weights).

## What this repository contains

| Path | Content |
|---|---|
| `nnunet_trainers/nnUNetTrainer_Freq.py` | Baseline with seed control (`nnUNetTrainer_FreqBase`) and frequency-domain trainers: `RASS`, `RAS` (RASS + random mask shuffle), `WTConv`, `FFC`, `RAS_WTConv` (also `SpecTilt`, registered but not run) |
| `nnunet_trainers/nnUNetTrainer_PDUNet.py` | Re-implemented PDU-Net (exploratory configuration) |
| `gpu/run_freq.sh` | Smoke test, training and single-run external inference |
| `s12_freq_eval.py` | Frozen statistical analysis (v0.3.1): fixed-sequence H1 → H2 → H3, exploratory comparisons, sensitivity analyses S1–S7 |
| `s12b_h3_eye.py` | Eye-level H3 analysis registered in OSF Update 3 |
| `s13_retouch.py` | Conversion of RETOUCH volumes (.mhd) to resampled 8-bit PNG B-scans |
| `s01_prepare_towardpi.py`, `s01b_axial_check.py`, `s02_prepare_external.py`, `config.py`, `common.py`, `s06_evaluate.py` | Shared preprocessing and per-image scoring modules used by the scripts above. `config.py` and `s06_evaluate.py` are shared with the authors' other registered study; the parameters and functions belonging to that study have been removed, which does not affect any analysis in this paper. The complete versions will be released when that study is published. All code called by this study is unchanged. |
| `tests/` | Self-tests on simulated data (no real images needed) |
| `work/tables/freq_sap_params.json` | Registered analysis parameters |
| `work/splits/towardpi_splits_final.json`, `work/tables/towardpi_cases.csv` | Five-fold split of the TowardPi dataset (grouped by eye; fold 0 was used) and the per-B-scan case table derived from the public TowardPi data |
| `results/` | Aggregate result tables (hypotheses, exploratory comparisons, per-device results, sensitivity analyses) and Fig. 2 source data |
| `HASHES.sha256` | SHA-256 of every file in this release |
| `weights/` (Zenodo only) | Final checkpoints of all eleven model instances (see below) |

`nnUNetTrainer_FT100.py`, which appears in the registered code freeze, belongs to a separate study of the same project and is not included; it will be released with that study.

Most code comments are in Chinese; function names, arguments and outputs are in English.

**Not included:** no RETOUCH, Duke DME or OCTDL images, labels or predictions, and no per-image metrics of the external test sets. RETOUCH is used under a data confidentiality agreement and must be requested from the challenge organisers; Duke DME and OCTDL are available from their original publications.

## Verifying the frozen code

The analysis was run with code whose SHA-256 hashes were registered on OSF before training (Update 1) and before external inference (Update 3). To check a local copy:

```bash
sha256sum -c HASHES.sha256
```

Twelve of the 13 files registered on OSF are included and must match the registered hashes exactly (Supplementary Table S2 of the paper): `nnunet_trainers/nnUNetTrainer_Freq.py`, `nnunet_trainers/nnUNetTrainer_PDUNet.py`, `gpu/run_freq.sh`, `s12_freq_eval.py`, `s12b_h3_eye.py`, `s13_retouch.py`, `tests/test_freq.py`, `tests/test_pdunet.py`, `tests/test_freq_eval.py`, `tests/test_retouch.py`, `tests/test_s12b.py` and `work/tables/freq_sap_params.json`. The thirteenth, `nnUNetTrainer_FT100.py`, belongs to the separate study and is not included.

## Environment used in the study

- NVIDIA A10 GPU (Alibaba Cloud PAI-DSW), NGC PyTorch 25.03 image: PyTorch 2.7, Python 3.12.3
- nnU-Net v2.6.4
- pandas 2.2.3, NumPy 1.26.4, SciPy 1.15.2, statsmodels 0.15.0
- Preprocessing additionally uses Pillow, tifffile, scikit-learn (versions not recorded)

GPUs with compute capability below 7.5 (for example V100) are not supported by the PyTorch build of this image.

## Quick start

### 1. Install the trainers

Copy both files from `nnunet_trainers/` into `nnunetv2/training/nnUNetTrainer/` of your nnU-Net installation (needed for both training and inference).

### 2. Self-tests (CPU, simulated data)

```bash
python tests/test_freq.py
python tests/test_pdunet.py
python tests/test_freq_eval.py
python tests/test_retouch.py
python tests/test_s12b.py
```

### 3. Prepare data

The default dataset paths in `config.py` and `s13_retouch.py` are the authors' local folders; set your own in `config.py` (or through the environment variables `RAW_TOWARDPI`, `RAW_DUKE`, `RAW_OCTDL`, `RAW_RETOUCH`), then:

```bash
python s01_prepare_towardpi.py          # TowardPi → nnU-Net Dataset101_TowardPi + split file
python s02_prepare_external.py          # Duke DME and OCTDL
python s13_retouch.py --inspect         # check RETOUCH headers first
python s13_retouch.py                   # convert RETOUCH (after obtaining access)
```

All images are resampled to 3.9 μm (axial) × 11.7 μm (lateral) per pixel; the TowardPi axial spacing of 2.7 μm per pixel was calibrated empirically (see the paper, Section 2.3).

### 4. Inference with the released weights

```bash
export nnUNet_results=/path/to/weights/seed1      # seed directory of the model you want
nnUNetv2_predict -i INPUT_DIR -o OUTPUT_DIR -d 101 -c 2d -f 0 \
    -tr nnUNetTrainer_RAS -chk checkpoint_final.pth
```

Input images must be single-channel PNG B-scans named `<case>_0000.png` and resampled to the common spacing above.

### 5. Training and the full pipeline

`gpu/run_freq.sh smoke | train | predict` reproduces the study pipeline; `FREQ_RUNS` selects trainers and seeds (for example `FREQ_RUNS="nnUNetTrainer_RAS:1"`). The `predict` stage asks for confirmation because, in the study, each external set was run only once.

## Model weights (Zenodo)

| Configuration | Trainer | Seeds |
|---|---|---|
| Baseline | `nnUNetTrainer_250epochs` (seed 1, trained earlier with identical settings) and `nnUNetTrainer_FreqBase` (seeds 2, 3) | 1, 2, 3 |
| RAS (primary) | `nnUNetTrainer_RAS` | 1, 2, 3 |
| RASS | `nnUNetTrainer_RASS` | 1 |
| WTConv | `nnUNetTrainer_WTConv` | 1 |
| FFC | `nnUNetTrainer_FFC` | 1 |
| RAS + WTConv | `nnUNetTrainer_RAS_WTConv` | 1 |
| PDU-Net | `nnUNetTrainer_PDUNet` | 1 |

All models were trained on fold 0 of the public TowardPi dataset only (250 epochs × 250 iterations, final checkpoint). They are research models: absolute accuracy on unseen devices was low (mean volume IRF Dice 0.377 for RAS on RETOUCH), and they must not be used for clinical decisions.

## Licence

- Code: MIT License (see `LICENSE`).
- Model weights: CC BY 4.0, the licence of the TowardPi dataset on which they were trained (figshare, doi:10.6084/m9.figshare.30582035). Please also cite the TowardPi dataset when using them.

## Citation

If you use this code or the weights, please cite the paper above and the datasets: TowardPi (Huang et al., Sci. Data 2026), RETOUCH (Bogunović et al., IEEE Trans. Med. Imaging 2019), Duke DME (Chiu et al., Biomed. Opt. Express 2015) and OCTDL (Kulyabin et al., Sci. Data 2024). The frequency-domain components follow RAS4DG (Qiao et al., MICCAI 2024), WTConv (Finder et al., ECCV 2024) and FFC (Chi et al., NeurIPS 2020).

## Contact

Xiaowei Yang (yxwjsd2020@163.com); corresponding author Guohong Zhou (guohongzhou2016@163.com).
