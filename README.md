# PulseGuard

Implementation of **Event-preserving residual filtering for PPG**. A residual candidate is trained with waveform, pulse-timing, pulse-mass and identity objectives. A separate gate predicts whether that candidate improves pulse evidence relative to raw bypass. Subject-wise cross-fitting supplies gate targets.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Use measured reference PPG from [BIDMC](https://physionet.org/content/bidmc/1.0.0/) for controlled contamination pairs. Natural-motion evaluation uses PPG-DaLiA or WESAD wrist PPG with chest ECG rate references. Obtain datasets from their maintainers and export recordings to NumPy archives:

| Key | Shape | Meaning |
| --- | --- | --- |
| `ppg` | `(T,)` | PPG at the declared sampling frequency |
| `fs` | scalar | Sampling frequency in Hz |
| `acc` | `(3,T)` | Optional synchronized acceleration; absent channels become zeros |
| `pulse_times` | `(P,)` | Peripheral PPG event times in seconds, required for controlled references |
| `ecg_peak_times` | `(R,)` | ECG R-peak times in seconds, required for natural recordings |

Resample PPG and acceleration onto a common timeline before preparation; event timestamps remain in seconds. Do not use ECG R peaks as exact peripheral pulse timestamps. A natural ECG reference provides HR, not clean PPG.

Create `data/manifest.json` with paths relative to the manifest:

```json
[
  {"subject": "bidmc01", "split": "train", "track": "controlled", "file": "raw/bidmc01.npz"},
  {"subject": "bidmc02", "split": "train", "track": "controlled", "file": "raw/bidmc02.npz"},
  {"subject": "bidmc03", "split": "train", "track": "controlled", "file": "raw/bidmc03.npz"},
  {"subject": "bidmc04", "split": "dev", "track": "controlled", "file": "raw/bidmc04.npz"},
  {"subject": "bidmc05", "split": "test", "track": "controlled", "file": "raw/bidmc05.npz"},
  {"subject": "dalia01", "split": "test", "track": "natural", "file": "raw/dalia01.npz", "severity": 1}
]
```

Use complete subject-disjoint partitions and include the subjects selected by your protocol. Natural data must not be included in fitting or threshold selection. Assign natural severity strata using acquisition activity or motion information before evaluating prediction errors.

Preparation creates nonoverlapping segments, applies per-segment normalization, adds controlled white noise, random-walk drift and acceleration contamination, and retains reference peripheral events. `--severity` controls contamination amplitude; a manifest row can override it. `--seed` fixes contamination and subject-fold ordering. Natural measured PPG is kept as measured apart from normalization.

```bash
pulseguard prepare --manifest data/manifest.json --fs 64 --seconds 8 --output data/prepared --seed 0
```

## Training and inference

```bash
pulseguard train --data data/prepared --folds 3 --output runs/train --seed 0
```

## Evaluation

```bash
pulseguard evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

## Outputs

`model.pt` stores candidate/gate/calibration state. Evaluation writes `metrics.json`, `segments.json`, `waveforms.npz` and `waveform.png`.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
