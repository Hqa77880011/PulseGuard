import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, find_peaks, sosfiltfilt
from torch import nn
from torch.nn import functional as F


class ResidualFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(4, 32, 9, padding=4), nn.SiLU(),
            nn.Conv1d(32, 32, 9, padding=8, dilation=2), nn.SiLU(),
            nn.Conv1d(32, 32, 9, padding=16, dilation=4), nn.SiLU(),
            nn.Conv1d(32, 1, 9, padding=4),
        )

    def forward(self, x, acc):
        return x - self.network(torch.cat([x, acc], 1))


def event_evidence(x, fs):
    width = max(3, round(0.12 * fs) | 1)
    smooth = F.avg_pool1d(x, width, stride=1, padding=width // 2)
    baseline = F.avg_pool1d(smooth, 2 * width + 1, stride=1, padding=width)
    prominence = F.softplus(8 * (smooth - baseline - 0.15)) / 8
    refractory = max(3, round(0.3 * fs) | 1)
    local = F.max_pool1d(prominence, refractory, stride=1, padding=refractory // 2)
    return prominence * torch.exp(-20 * (local - prominence)) / max(1, fs * 0.1)


def event_loss(candidate, reference, fs, timing=True, count=True):
    a, b = event_evidence(candidate, fs), event_evidence(reference, fs)
    mass_a, mass_b = a.sum(-1), b.sum(-1)
    fa = a.cumsum(-1) / (mass_a[..., None] + 1e-6)
    fb = b.cumsum(-1) / (mass_b[..., None] + 1e-6)
    usable = ((mass_a > 0.1) & (mass_b > 0.1)).float()
    loss = torch.zeros_like(mass_a)
    if timing:
        loss = loss + (fa - fb).abs().mean(-1) * usable
    if count:
        loss = loss + 0.1 * (mass_a - mass_b).abs()
    return loss.flatten(1).mean(1)


def gate_features(x, candidate, acc, fs):
    difference = candidate - x
    return torch.stack([
        x.std((1, 2)), candidate.std((1, 2)), difference.square().mean((1, 2)),
        acc.square().mean((1, 2)), event_evidence(x, fs).sum((1, 2)),
        event_evidence(candidate, fs).sum((1, 2)),
    ], -1)


class BenefitGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pulses(x, fs):
    sos = butter(3, [0.5, min(5, fs * 0.45)], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, x)
    peaks, _ = find_peaks(filtered, distance=max(1, round(fs * 0.3)), prominence=max(1e-6, 0.25 * np.std(filtered)))
    return peaks / fs


def pulse_errors(predicted, reference, tolerance):
    if not len(predicted) or not len(reference):
        return len(predicted), len(reference)
    cost = np.abs(predicted[:, None] - reference[None])
    penalty = (max(cost.shape) + 1) * max(tolerance, 1)
    weighted = np.where(cost <= tolerance, cost, penalty)
    row, col = linear_sum_assignment(weighted)
    matches = int(np.sum(cost[row, col] <= tolerance))
    return len(predicted) - matches, len(reference) - matches
