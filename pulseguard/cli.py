import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.nn import functional as F

from .method import BenefitGate, ResidualFilter, event_loss, gate_features, pulse_errors, pulses


def prepare(args):
    manifest = Path(args.manifest)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    seen, packed = {}, {}
    rng = np.random.default_rng(args.seed)
    for row in rows:
        subject, split = str(row["subject"]), row["split"]
        if subject in seen and seen[subject] != split:
            raise ValueError("A subject cannot occur in multiple partitions")
        seen[subject] = split
        raw = np.load(manifest.resolve().parent / row["file"], allow_pickle=False)
        fs = float(raw["fs"])
        if fs != args.fs:
            raise ValueError("Resample signals and event timestamps consistently before preparation")
        ppg = raw["ppg"].reshape(-1)
        acc = raw["acc"] if "acc" in raw else np.zeros((3, len(ppg)))
        events = raw["pulse_times"] if "pulse_times" in raw else None
        ecg = raw["ecg_peak_times"] if "ecg_peak_times" in raw else None
        if row["track"] == "controlled" and events is None:
            raise ValueError("Controlled references require peripheral pulse_times")
        if row["track"] == "natural" and ecg is None:
            raise ValueError("Natural recordings require ECG peak times for HR reference")
        size = round(args.seconds * fs)
        for start in range(0, len(ppg) - size + 1, size):
            end = start + size
            clean = ppg[start:end].astype(np.float32)
            mean, std = float(clean.mean()), max(float(clean.std()), 1e-5)
            clean = (clean - mean) / std
            a = acc[:, start:end].astype(np.float32)
            a = (a - a.mean(-1, keepdims=True)) / np.maximum(a.std(-1, keepdims=True), 1e-5)
            track = row["track"]
            severity = float(row.get("severity", args.severity))
            x = clean.copy()
            if track == "controlled":
                white = rng.normal(size=size)
                drift = np.cumsum(rng.normal(size=size))
                drift = drift / max(drift.std(), 1e-5)
                x = clean + severity * (0.5 * white + 0.3 * drift + 0.2 * a[0])
                event = events[(events >= start / fs) & (events < end / fs)] - start / fs
                hr = 60 / np.median(np.diff(event)) if len(event) >= 2 else np.nan
                mask = np.zeros(size, dtype=np.uint8)
                mask[np.minimum((event * fs).astype(int), size - 1)] = 1
            else:
                selected = ecg[(ecg >= start / fs) & (ecg < end / fs)]
                hr = 60 / np.median(np.diff(selected)) if len(selected) >= 2 else np.nan
                mask = np.zeros(size, dtype=np.uint8)
            if not np.isfinite(hr):
                continue
            packed.setdefault(split, []).append((subject, x, clean, a, mask, hr, track, severity))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for split, samples in packed.items():
        fields = list(zip(*samples))
        np.savez_compressed(out / f"{split}.npz", subject=fields[0], x=np.array(fields[1], np.float32),
                            reference=np.array(fields[2], np.float32), acc=np.array(fields[3], np.float32),
                            events=np.array(fields[4]), hr=fields[5], track=fields[6], severity=fields[7], fs=args.fs)


def tensors(data, index, device):
    return (torch.as_tensor(data["x"][index], device=device)[:, None],
            torch.as_tensor(data["reference"][index], device=device)[:, None],
            torch.as_tensor(data["acc"][index], device=device))


def fit_filter(data, index, args):
    if not len(index):
        raise ValueError("A cross-fitting fold has no training subjects")
    net = ResidualFilter().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    for _ in range(args.epochs):
        order = rng.permutation(index)
        for begin in range(0, len(order), args.batch_size):
            x, reference, acc = tensors(data, order[begin:begin + args.batch_size], args.device)
            candidate = net(x, acc)
            loss = F.mse_loss(candidate, reference)
            loss = loss + args.event_weight * event_loss(candidate, reference, float(data["fs"]), not args.no_time, not args.no_count).mean()
            if not args.no_identity:
                loss = loss + args.identity_weight * F.mse_loss(net(reference, acc), reference)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return net.eval()


def train(args):
    torch.manual_seed(args.seed)
    data = np.load(Path(args.data) / "train.npz", allow_pickle=False)
    if np.any(data["track"] != "controlled"):
        raise ValueError("Candidate and gate training require controlled reference pairs")
    subjects = np.unique(data["subject"])
    if len(subjects) < args.folds or args.folds < 2:
        raise ValueError("Cross-fitting needs at least two nonempty subject folds")
    shuffled = np.random.default_rng(args.seed).permutation(subjects)
    features, targets = [], []
    for heldout in np.array_split(shuffled, args.folds):
        hold = np.isin(data["subject"], heldout)
        net = fit_filter(data, np.flatnonzero(~hold), args)
        with torch.no_grad():
            for index in np.array_split(np.flatnonzero(hold), max(1, int(np.ceil(hold.sum() / args.batch_size)))):
                x, reference, acc = tensors(data, index, args.device)
                candidate = net(x, acc)
                features.append(gate_features(x, candidate, acc, float(data["fs"])).cpu())
                if args.quality_gate:
                    target = (x - reference).square().mean((1, 2)) - 0.1
                else:
                    target = event_loss(x, reference, float(data["fs"])) - event_loss(candidate, reference, float(data["fs"]))
                targets.append(target.cpu())
    f = torch.cat(features).to(args.device)
    y = torch.cat(targets).to(args.device)
    mean, scale = f.mean(0), f.std(0).clamp_min(1e-5)
    gate = BenefitGate().to(args.device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=args.lr)
    for _ in range(args.gate_epochs):
        loss = F.mse_loss(gate((f - mean) / scale), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = fit_filter(data, np.arange(len(data["x"])), args)
    dev = np.load(Path(args.data) / "dev.npz", allow_pickle=False)
    if set(data["subject"]) & set(dev["subject"]):
        raise ValueError("Development subjects overlap training")
    if np.any(dev["track"] != "controlled"):
        raise ValueError("Gate threshold selection requires development reference pairs")
    with torch.no_grad():
        x, reference, acc = tensors(dev, np.arange(len(dev["x"])), args.device)
        candidate = final(x, acc)
        scores = gate((gate_features(x, candidate, acc, float(dev["fs"])) - mean) / scale)
        raw_loss = event_loss(x, reference, float(dev["fs"]))
        filtered_loss = event_loss(candidate, reference, float(dev["fs"]))
        thresholds = torch.cat([scores.min()[None] - 1, torch.quantile(scores, torch.linspace(0, 1, 51, device=args.device)), scores.max()[None] + 1])
        costs = [torch.where(scores > t, filtered_loss, raw_loss).mean().item() for t in thresholds]
        threshold = float(thresholds[int(np.argmin(costs))])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"filter": final.state_dict(), "gate": gate.state_dict(), "mean": mean.cpu(),
                "scale": scale.cpu(), "threshold": threshold, "fs": float(data["fs"]),
                "train_subjects": subjects.tolist(), "dev_subjects": np.unique(dev["subject"]).tolist()}, out / "model.pt")


def evaluate(args):
    data = np.load(Path(args.data) / "test.npz", allow_pickle=False)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if (set(saved["train_subjects"]) | set(saved["dev_subjects"])) & set(data["subject"]):
        raise ValueError("Test subjects overlap fitting subjects")
    fs = float(data["fs"])
    if fs != saved["fs"]:
        raise ValueError("Test sampling frequency differs from fitted model")
    net, gate = ResidualFilter().to(args.device), BenefitGate().to(args.device)
    net.load_state_dict(saved["filter"])
    gate.load_state_dict(saved["gate"])
    net.eval()
    gate.eval()
    results, outputs = [], []
    for start in range(0, len(data["x"]), args.batch_size):
        index = np.arange(start, min(start + args.batch_size, len(data["x"])))
        with torch.no_grad():
            x, _, acc = tensors(data, index, args.device)
            candidate = net(x, acc)
            scores = gate((gate_features(x, candidate, acc, fs) - saved["mean"].to(args.device)) / saved["scale"].to(args.device))
            selected = scores > saved["threshold"]
            if args.method == "forced":
                selected = torch.ones_like(selected)
            if args.method in {"raw", "bandpass"}:
                selected = torch.zeros_like(selected)
            corrected = torch.where(selected[:, None, None], candidate, x).cpu().numpy()[:, 0]
        if args.method == "bandpass":
            corrected = sosfiltfilt(butter(3, [0.5, min(5, fs * 0.45)], fs=fs, btype="bandpass", output="sos"), corrected, axis=-1)
        outputs.extend(corrected)
        for j, i in enumerate(index):
            events = pulses(corrected[j], fs)
            hr = 60 / np.median(np.diff(events)) if len(events) >= 2 else None
            inserted, deleted = None, None
            reference = np.flatnonzero(data["events"][i]) / fs
            if data["track"][i] == "controlled":
                inserted, deleted = pulse_errors(events, reference, args.tolerance)
            results.append({"subject": str(data["subject"][i]), "track": str(data["track"][i]),
                            "severity": float(data["severity"][i]), "hr": hr, "reference_hr": float(data["hr"][i]),
                            "inserted": inserted, "deleted": deleted, "reference_pulses": len(reference),
                            "corrected": bool(selected[j])})
    metrics = {}
    for track in sorted({r["track"] for r in results}):
        rows = [r for r in results if r["track"] == track]
        valid = [r for r in rows if r["hr"] is not None]
        errors = np.array([r["hr"] - r["reference_hr"] for r in valid])
        metrics[track] = {"hr_mae_bpm": float(np.abs(errors).mean()) if len(errors) else None,
                          "hr_rmse_bpm": float(np.sqrt((errors ** 2).mean())) if len(errors) else None,
                          "hr_output_fraction": len(valid) / len(rows),
                          "correction_fraction": float(np.mean([r["corrected"] for r in rows]))}
        if track == "controlled":
            count = sum(r["reference_pulses"] for r in rows)
            metrics[track].update({key + "_per_100": 100 * sum(r[key] for r in rows) / count if count else None for key in ["inserted", "deleted"]})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (out / "segments.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    np.savez_compressed(out / "waveforms.npz", corrected=np.array(outputs))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(len(outputs[0])) / fs, data["x"][0], label="Measured")
    ax.plot(np.arange(len(outputs[0])) / fs, outputs[0], label="Output", alpha=0.8)
    ax.set(xlabel="Time (s)", ylabel="Normalized PPG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "waveform.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--fs", type=float, default=64)
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("--severity", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(function=prepare)
    for name in ["train", "evaluate"]:
        p = commands.add_parser(name)
        p.add_argument("--data", default="data/prepared")
        p.add_argument("--output", default=f"runs/{name}")
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch-size", type=int, default=32)
        if name == "train":
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--gate-epochs", type=int, default=200)
            p.add_argument("--folds", type=int, default=3)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--event-weight", type=float, default=1)
            p.add_argument("--identity-weight", type=float, default=0.2)
            for flag in ["no-time", "no-count", "no-identity", "quality-gate"]:
                p.add_argument("--" + flag, action="store_true")
            p.set_defaults(function=train)
        else:
            p.add_argument("--checkpoint", default="runs/train/model.pt")
            p.add_argument("--method", choices=["full", "forced", "raw", "bandpass"], default="full")
            p.add_argument("--tolerance", type=float, default=0.15)
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)
