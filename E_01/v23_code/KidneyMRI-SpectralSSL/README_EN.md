# KidneyBand-ReliableSSL v2: Label-efficient T2 kidney MRI segmentation and reliable TKV

The active development schema is v2: three-class bilateral output, UABand-HT, CSD-MT, DZ-BiMorph, stable BF16/FP32 losses, resumable checkpoints, source-only external evaluation, and a fail-closed Q2 evidence gate. V1 configurations and results remain readable but are not mixed into v2 confirmatory statistics.

```powershell
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\q2_v2.yaml --phase screen --resume
.\.venv\Scripts\kidney-seg.exe lock-protocol --suite configs\benchmark\q2_v2.yaml
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\q2_v2.yaml --phase internal --resume
.\.venv\Scripts\kidney-seg.exe prepare-external --dataset chaos --data-root "D:\30 医学论文\01\data\CHAOS"
.\.venv\Scripts\kidney-seg.exe prepare-external --dataset amos --data-root "D:\30 医学论文\01\data\AMOS22"
```

See `docs/novelty_matrix_v2.md` and `docs/q2_protocol_v2.md` for contribution boundaries and preregistered retention rules.

This is an independent clean-room PyTorch/MONAI research implementation for the 60-subject, 100-acquisition [T2-weighted Kidney MRI Segmentation dataset](https://zenodo.org/records/5153568). It combines a 2.5D U-Net with three independently switchable candidate contributions: an inverse-probability-weighted random directional spectral-band loss, distribution-matched complementary Fourier-domain image views for an EMA Mean Teacher, and registration-free scan–rescan soft-TKV consistency.

All Fourier operations are applied to reconstructed magnitude images, not scanner-acquired k-space. The mechanisms are candidate contributions pending a complete literature review and confirmatory experiments; this repository makes no priority claim.

## Commands

```powershell
./scripts/bootstrap.ps1
.\.venv\Scripts\kidney-seg.exe download --data-root "D:\30 医学论文\01\data\T2KidneyMRI"
.\.venv\Scripts\kidney-seg.exe audit --data-root "D:\30 医学论文\01\data\T2KidneyMRI"
.\.venv\Scripts\kidney-seg.exe make-folds --manifest "D:\30 医学论文\01\data\T2KidneyMRI\audit\manifest.csv" --seed 20260802
.\.venv\Scripts\kidney-seg.exe pilot
```

The bootstrap script syncs locked third-party dependencies without an editable project install, then installs the local project as a regular wheel. This avoids Windows locale errors caused by an editable `.pth` containing a non-ASCII project path.


Subject-level split enforcement prevents scan–rescan leakage. Unlabeled Dataset items contain no mask tensor or mask path. Evaluation is acquisition- and subject-based and includes Dice, HD95, ASD, TKV error, within-subject repeat CV, ICC(2,1), and Bland–Altman summaries.

## Fair SCI benchmark

The registered suite contains Daniel-UNet, Damiano-UNet/DicePlus (deterministic, MCD and TTA evaluations count as one family), nnU-Net v2 2D, UNet++ 2.5D, TCSM, CMMT-Net clean-room adaptation, and B0–B3 ablations. Development uses only fold 0 from split seed `20260802`. A SHA-256 protocol lock is required before the `20260803` split can be used for the confirmatory 5-fold, 3-seed experiment.

```powershell
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\sci_suite.yaml --phase dev --resume
.\.venv\Scripts\kidney-seg.exe benchmark --suite configs\benchmark\sci_suite.yaml --phase confirmatory --resume
```

Superiority is gated by a positive subject-level paired Dice 95% bootstrap CI, Holm-adjusted permutation `p<0.05`, and an upper 95% CI no larger than `0.02` for the proposed-minus-baseline TKV relative-error difference. Incomplete folds, seeds, or subjects can never produce a superiority claim.

See `docs/experiment_protocol.md`, `docs/SCI_算法与投稿评估.md`, `docs/novelty_matrix.md`, and `THIRD_PARTY_NOTICES.md` for scope and provenance.
