# PulseGuard

Implementation of **Event-preserving residual filtering for PPG**. A residual candidate is trained with waveform, pulse-timing, pulse-mass and identity objectives. A separate gate predicts whether that candidate improves pulse evidence relative to raw bypass. Subject-wise cross-fitting supplies gate targets.

## Install

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.10+ and activate the environment before installation. Commands default to the CPU; a compatible PyTorch GPU installation supports `--device cuda`.

## Data

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

```bash
pulseguard prepare --manifest data/manifest.json --fs 64 --seconds 8 --output data/prepared
pulseguard train --data data/prepared --folds 3 --output runs/train
pulseguard evaluate --data data/prepared --checkpoint runs/train/model.pt --output runs/full
```

Preparation creates nonoverlapping segments, applies per-segment normalization, adds controlled white noise, random-walk drift and acceleration contamination, and retains reference peripheral events. `--severity` controls contamination amplitude; a manifest row can override it. `--seed` fixes contamination and subject-fold ordering. Natural measured PPG is kept as measured apart from normalization.

## Comparisons

```bash
pulseguard evaluate --method raw --output runs/raw
pulseguard evaluate --method bandpass --output runs/bandpass
pulseguard evaluate --method forced --output runs/forced
pulseguard train --event-weight 0 --output runs/waveform
pulseguard evaluate --checkpoint runs/waveform/model.pt --method forced --output runs/waveform_eval
pulseguard train --no-time --output runs/no_time
pulseguard train --no-count --output runs/no_count
pulseguard train --no-identity --output runs/no_identity
pulseguard train --quality-gate --output runs/quality_gate
pulseguard evaluate --checkpoint runs/quality_gate/model.pt --output runs/quality_gate_eval
```

Evaluate every newly trained ablation with its own checkpoint and output directory. `--epochs` fits each candidate, including the final candidate; `--gate-epochs` fits the small benefit regressor. `--lr`, `--batch-size`, `--event-weight` and `--identity-weight` control optimization. At least two subject folds are required. Gate fitting receives only outputs from candidates that did not train on the held-out subject. Development subjects select the correction threshold by event loss after final candidate fitting.

## Metrics and outputs

`model.pt` stores the candidate, gate, gate feature normalization, frozen threshold and fitting-subject identities. Evaluation writes `metrics.json`, per-segment `segments.json`, corrected `waveforms.npz` and an example `waveform.png`.

Controlled-reference metrics include insertions and deletions per 100 reference pulses using one-to-one hard event matching. `--tolerance` sets the matching tolerance in seconds, default 0.15; freeze it before comparing methods. HR MAE and RMSE are in beats/min. Natural HR uses ECG interbeat intervals as its reference. `hr_output_fraction` records segments with enough detected pulses; unavailable rate estimates remain missing instead of contributing zero error. `correction_fraction` records how often the filter is selected. Per-segment severity allows motion-stratified analysis without ranking test errors.

## Implementation choices

The residual network is a compact dilated convolutional filter. The fixed differentiable event proxy uses smoothed local prominence and refractory suppression; it has no trainable parameters and cannot co-adapt with the candidate. Normalized cumulative event evidence supplies timing loss, while absolute total-mass difference supplies the complementary count loss. Timing is disabled when either signal has insufficient event evidence. Final hard pulse detection is separate from this soft training proxy. The binary gate implements the raw-bypass and full-correction endpoints of the paper's output equation.

The quality-gate control predicts controlled artifact energy; the benefit gate predicts reduction in event loss. The waveform-only candidate, bandpass and forced-correction comparisons are directly executable. The compact candidate is not named after an external published architecture. Clean measured references and controlled contamination define the training target; the natural track is evaluated only with the rate information available from ECG.
