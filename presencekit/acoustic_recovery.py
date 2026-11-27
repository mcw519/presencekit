"""Short raw-onset recovery for the experimental cross-check."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

RISE_DB = 8
TILT_DB = -12
BROADBAND_RISE_DB = 18.0
BROADBAND_LOW_RISE_DB = 15.0
BROADBAND_TILT_DB = 0.0
HISTORY_S = 0.2
HOLD_S = 0.16


def _band_power(signal: np.ndarray, sample_rate: int, lo_hz: float, hi_hz: float) -> float:
    win = np.hanning(len(signal))
    denom = max(1e-12, float(np.sum(win**2)) * len(signal))
    freq = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    power = np.abs(np.fft.rfft(signal * win)) ** 2 / denom
    return float(power[(freq >= lo_hz) & (freq < hi_hz)].sum())


def onset_features(attack: np.ndarray, history: np.ndarray, sample_rate: int) -> tuple[float, float]:
    def level(signal):
        return 10 * math.log10(max(1e-12, float(np.mean(signal.astype(np.float64) ** 2))))

    freq = np.fft.rfftfreq(len(attack), 1 / sample_rate)
    power = np.abs(np.fft.rfft(attack * np.hanning(len(attack)))) ** 2
    low = power[(freq >= 150) & (freq < 1800)].sum()
    high = power[(freq >= 1800) & (freq < 6000)].sum()
    return level(attack) - level(history), 10 * math.log10(max(high, 1e-12) / max(low, 1e-12))


def low_band_rise_db(attack: np.ndarray, history: np.ndarray, sample_rate: int) -> float:
    att_low = _band_power(attack, sample_rate, 150.0, 1800.0)
    his_low = _band_power(history, sample_rate, 150.0, 1800.0)
    return 10 * math.log10(max(att_low, 1e-18) / max(his_low, 1e-18))


def qualifies(attack: np.ndarray, history: np.ndarray, sample_rate: int) -> bool:
    rise, tilt = onset_features(attack, history, sample_rate)
    if rise >= RISE_DB and tilt <= TILT_DB:
        return True
    if rise >= BROADBAND_RISE_DB and tilt <= BROADBAND_TILT_DB:
        return low_band_rise_db(attack, history, sample_rate) >= BROADBAND_LOW_RISE_DB
    return False


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
                if qualifies(history[split:], history[:split], self.sample_rate):
                    self.onset = end_sample - self.attack_hops * self.hop

    def allows(self, output_start: int, output_bound: int) -> bool:
        onset = self.onset
        return onset is not None and output_bound > onset and output_start < onset + round(HOLD_S * self.sample_rate)
