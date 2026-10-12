# Third-party sources and research provenance

This project is internal research code and intentionally has no public software license. No source code was copied from SSDU, K-band, Noisier2Noise, FreMIM, FRCNet, or the 2025 kidney uncertainty repository.

Conceptual provenance:

- Yaman et al., *Self-supervised learning of physics-guided reconstruction neural networks without fully sampled reference data*, MRM 2020 (SSDU): mutually exclusive information partitions.
- Wang et al., *k-band: Self-supervised MRI Reconstruction via Stochastic Gradient Descent over K-space Subsets*, 2023: random directional bands and inverse sampling-probability weighting.
- Millard and Chiew, *A theoretical framework for self-supervised MR image reconstruction using sub-sampling via variable density Noisier2Noise*, 2022: distribution-matched masking and density correction.
- Wang et al., *FreMIM: Fourier Transform Meets Masked Image Modeling for Medical Image Segmentation*, WACV 2024: frequency-domain medical segmentation pretraining context.
- He et al., *FRCNet: Frequency and Region Consistency for Semi-supervised Medical Image Segmentation*, MICCAI 2024: prior frequency-domain semi-supervised consistency.
- Damiano et al., *The Impact of Uncertainty Estimation on Radiomic Segmentation Reproducibility and Scan-Rescan Repeatability in Kidney MRI*, Medical Physics 2025: kidney MRI MCD/TTA uncertainty and repeatability boundary.

Local upstream repositories remain separate and unmodified. Their license constraints do not transfer a right to redistribute their code here: SSDU states educational/non-profit research use, Noisier2Noise_for_recon is GPL-3.0, and the K-band repository has no root-wide license (its DeepInPy subtree is MIT).

Dataset: Daniel et al., [T2-weighted Kidney MRI Segmentation](https://doi.org/10.5281/zenodo.5153568), with its record-level terms and associated paper governing data use.

## ReliableSSL v2 additions

- Semi-URF, IEEE JBHI 2026, DOI `10.1109/JBHI.2026.3690043`; official repository `senseiyang/Semi-URF`. The local benchmark configuration is an explicitly marked clean-room, abstract-level adapter (output UDAT and Haar boundary proxy), not an exact reproduction of BUCE/FEFF and contains no copied source.
- RandDG, Medical Physics 2025, DOI `10.1002/mp.70118`. The local adapter uses input-level low-frequency amplitude randomization with the common backbone; it does not claim to reproduce the paper's GUNet and feature-level ULoFT implementation.
- FRCNet, MICCAI 2024, DOI `10.1007/978-3-031-72111-3_29`. The official MICCAI paper page listed no code repository on 2026-08-02. The local output-level DCT/region-similarity adapter is therefore a declared approximation.
- nnU-Net v2 3D full-resolution uses the official Apache-2.0 `nnunetv2==2.8.1` package. V2 converts binary source masks to canonical RAS left/right classes in an isolated nnU-Net workspace; source NIfTI files remain unchanged.

External datasets are not part of the software repository:

- CHAOS training data, Zenodo DOI `10.5281/zenodo.3431873`, CC-BY-NC-SA-4.0. Only the 20 labeled T2-SPIR training cases are prepared locally; challenge test ground truth is not available.
- AMOS22, Zenodo DOI `10.5281/zenodo.7262581`, CC-BY-4.0 as declared by its record. The external manifest excludes CT and maps official label 3 to left kidney and label 2 to right kidney.

Download archives, derived NIfTI files, manifests, predictions, checkpoints, and experiment logs stay outside Git under `D:/30 医学论文/01/data` and `D:/30 医学论文/01/experiments`.

## SCI benchmark provenance

The following entries document what was consulted and what is actually executed. Except for the locked `nnunetv2` dependency, upstream source files are not vendored or copied.

| Registered family | Paper / upstream checked 2026-08-02 | Executed implementation and material deviation |
|---|---|---|
| Daniel-UNet | MRM 2021, DOI `10.1002/mrm.28768`; `alexdaniel654/Renal_Segmentor` HEAD `4b0c57465ba0b11abc57ee20595770e8dc3b0988` | Clean-room 2D U-Net and Dice loss. Uses the common RAS/z-score preprocessing, 25% subject labels, 100-epoch budget and registered LR grid rather than the paper's slice windowing, 150-epoch SGD and original 50-subject training split. |
| Damiano-UNet/DicePlus | Medical Physics 2025, DOI `10.1002/mp.17995`; Zenodo software v1.0 `10.5281/zenodo.15188948` | Clean-room 2D U-Net, dropout 0.25 and Dice++ (`gamma=2`). The same checkpoint is evaluated deterministically, by 20-pass MCD and by dihedral TTA, and counts once. Common split/budget replaces the original TensorFlow protocol. No Zenodo code or weights are included. |
| nnU-Net v2 2D | Nature Methods 2021, DOI `10.1038/s41592-020-01008-z`; PyPI `nnunetv2==2.8.1`; repository HEAD `0e495086eb108ff79afe106291e8c15bd2f2bc3a` | Official Apache-2.0 package. Runs both `nnUNetTrainer_100epochs` for development context and official default `nnUNetTrainer` for confirmatory comparison. Dataset conversion hard-links/copies NIfTI files without changing the originals and verifies ID/geometry reversibility. |
| UNet++ 2.5D | IEEE TMI 2020, DOI `10.1109/TMI.2019.2959609`; official repository HEAD `db124279549ed983b9dbce1ef50831aaeecf6452` | Clean-room nested dense skip paths and deep supervision-inspired averaged heads, adapted to three adjacent slices and the common budget. No upstream source is copied. |
| TCSM | IEEE TNNLS 2021, DOI `10.1109/TNNLS.2020.2995319` | Clean-room EMA teacher/student with rotation, flip and center-scale equivariance. Uses the common U-Net, confidence filter and ramp; therefore it is an adaptation, not a bitwise reproduction of the paper's task-specific networks. |
| CMMT-Net | Medical Image Analysis 2024, DOI `10.1016/j.media.2024.103302`; official repository HEAD `485c2507c3df5fdd99426bd672e45efe51c7118a` (no repository license found) | Clean-room shared encoder/dual decoder, cross-head EMA pseudo supervision, cross-set CutMix-style batch mixing and one-step input MVAT adaptation. The original 3D task-specific networks and code are not copied; results must be named a clean-room adaptation. |

License note: the UNet++ upstream repository exposes a license file, but this project does not import its source. The CMMT-Net repository showed no root license at the pinned inspection point, so it was read only for structural verification. Damiano's Zenodo record is open software but did not expose an explicit reusable software license in the inspected metadata; none of its code or weights are redistributed.

### v2.1 matched-input comparison variants

The formal v2.1 comparison registers five-slice input variants for UNet++, TCSM,
CMMT-Net, FRCNet, Semi-URF, and RandDG so that common 2.5D methods receive the
same adjacent-slice context as the selected proposed model. This is a controlled
matched-input adaptation and not a claim that the cited papers used five-slice
kidney MRI inputs. Daniel-UNet and Damiano-UNet retain their paper-motivated
single-slice input, while nnU-Net remains the official three-dimensional package.

Only `nnunetv2==2.8.1` is an official upstream executable in this comparison.
FRCNet, Semi-URF, and RandDG remain explicitly labeled proxy implementations in
the suite metadata, protocol lock, result tables, and manuscript-facing reports.

### v2.2 CSD-only selection

The v2.2 formal candidate reuses the project's clean-room CSD-MT implementation
after the preregistered v2.1 development screen retained CSD and rejected
UABand. No third-party source code was added or copied for this selection.
UABand and DZ-BiMorph remain documented development ablations and are not
silently included in the v2.2 proposed model.
