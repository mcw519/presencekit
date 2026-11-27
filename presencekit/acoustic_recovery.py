"""Short raw-onset recovery for the experimental cross-check."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

RISE_DB = 8
TILT_DB = -12
HISTORY_S = 0.2
HOLD_S = 0.12


def onset_features(attack: np.ndarray, history: np.ndarray, sample_rate: int) -> tuple[float, float]:
    def level(signal):
        return 10 * math.log10(max(1e-12, float(np.mean(signal.astype(np.float64) ** 2))))

    freq = np.fft.rfftfreq(len(attack), 1 / sample_rate)
    power = np.abs(np.fft.rfft(attack * np.hanning(len(attack)))) ** 2
    low = power[(freq >= 150) & (freq < 1800)].sum()
    high = power[(freq >= 1800) & (freq < 6000)].sum()
    return level(attack) - level(history), 10 * math.log10(max(high, 1e-12) / max(low, 1e-12))


class AcousticRecovery:
    def __init__(self, sample_rate: int, hop: int, min_on: float):
        self.sample_rate = sample_rate
        self.hop = hop
        self.attack_hops = math.ceil(min_on * sample_rate / hop)
        self.history_hops = round(HISTORY_S * sample_rate / hop)
        self.recent: deque[np.ndarray] = deque(maxlen=self.history_hops + self.attack_hops)
        self.onset: int | None = None

    def step(self, samples: np.ndarray, end_sample: int, onset: bool) -> None:
        self.recent.append(samples.copy())
        if onset:
            self.onset = None
            if len(self.recent) == self.recent.maxlen:
                history = np.concatenate(self.recent)
                split = self.history_hops * self.hop
                rise, tilt = onset_features(history[split:], history[:split], self.sample_rate)
                if rise >= RISE_DB and tilt <= TILT_DB:
                    self.onset = end_sample - self.attack_hops * self.hop

    def allows(self, output_start: int, output_bound: int) -> bool:
        onset = self.onset
        return onset is not None and output_bound > onset and output_start < onset + round(HOLD_S * self.sample_rate)
