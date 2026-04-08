"""
ECG-Specific Data Augmentation Pipeline.

Domain-specific augmentations are critical for ECG deep learning because:
1. PTB-XL has only ~21k records — augmentation effectively expands the dataset
2. ECG signals have structured noise patterns (electrode noise, motion artifact)
   that differ from natural image noise
3. Clinical variability (body position, electrode placement variations)
   should be captured in training

All augmentations are designed to:
- Preserve arrhythmia-defining features (R-peak timing, P-QRS-T morphology)
- Simulate realistic acquisition artifacts
- NOT introduce physiologically impossible patterns

Augmentation strategy follows the PhysioNet Challenge 2021 winners
(Natarajan et al., 2022) and CLOCS contrastive learning approach
(Kiyasseh et al., 2021).

References:
    Natarajan et al. (2022). Convolution-free ECG classification.
        PhysioNet Challenge Report.
    Kiyasseh et al. (2021). CLOCS: Contrastive learning of cardiac signals.
        ICML.
    Hannun et al. (2019). Cardiologist-level arrhythmia detection.
        Nature Medicine.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal


# ─────────────────────────────────────────────
#  Base Augmentation
# ─────────────────────────────────────────────

class ECGAugmentation:
    """Base class for ECG augmentations."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return self.apply(x)
        return x

    def apply(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ─────────────────────────────────────────────
#  Noise Augmentations
# ─────────────────────────────────────────────

class GaussianNoise(ECGAugmentation):
    """Add Gaussian (white) noise to simulate ADC quantization noise.

    Typical ECG SNR: 20–40 dB. Scale noise relative to signal amplitude.
    """

    def __init__(self, snr_db_range: Tuple[float, float] = (20.0, 40.0), p: float = 0.5) -> None:
        super().__init__(p)
        self.snr_min, self.snr_max = snr_db_range

    def apply(self, x: np.ndarray) -> np.ndarray:
        snr_db = random.uniform(self.snr_min, self.snr_max)
        signal_power = np.mean(x ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.randn(*x.shape).astype(np.float32) * np.sqrt(noise_power)
        return x + noise


class PowerlineNoise(ECGAugmentation):
    """Simulate powerline interference (50 or 60 Hz).

    Powerline noise is one of the most common ECG artifacts.
    Amplitude varies 0.01–0.2 mV depending on electrode setup.
    """

    def __init__(
        self,
        freq: float = 60.0,
        amplitude_range: Tuple[float, float] = (0.01, 0.15),
        fs: float = 500.0,
        p: float = 0.3,
    ) -> None:
        super().__init__(p)
        self.freq = freq
        self.amp_min, self.amp_max = amplitude_range
        self.fs = fs

    def apply(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[-1]
        t = np.arange(T) / self.fs
        amplitude = random.uniform(self.amp_min, self.amp_max)
        phase = random.uniform(0, 2 * np.pi)
        noise = amplitude * np.sin(2 * np.pi * self.freq * t + phase)
        return (x + noise).astype(np.float32)


class EMGNoise(ECGAugmentation):
    """Simulate muscle artifact (electromyogram / EMG noise).

    EMG appears as high-frequency burst noise (50–500 Hz).
    Simulated as bandlimited white noise with random ON/OFF bursts.
    """

    def __init__(
        self,
        amplitude_range: Tuple[float, float] = (0.05, 0.3),
        burst_duration_range: Tuple[float, float] = (0.5, 3.0),
        fs: float = 500.0,
        p: float = 0.2,
    ) -> None:
        super().__init__(p)
        self.amp_min, self.amp_max = amplitude_range
        self.burst_min, self.burst_max = burst_duration_range
        self.fs = fs

    def apply(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[-1]
        amplitude = random.uniform(self.amp_min, self.amp_max)

        # Generate burst noise
        noise = np.random.randn(*x.shape).astype(np.float32) * amplitude

        # Bandpass to EMG frequency range (50–200 Hz)
        nyq = self.fs / 2.0
        if nyq > 200:
            sos = sp_signal.butter(4, [50.0 / nyq, 200.0 / nyq], btype="bandpass", output="sos")
            noise = sp_signal.sosfilt(sos, noise, axis=-1).astype(np.float32)

        # Apply burst envelope (random on/off)
        burst_samples = int(random.uniform(self.burst_min, self.burst_max) * self.fs)
        burst_start = random.randint(0, max(0, T - burst_samples))
        mask = np.zeros(T, dtype=np.float32)
        mask[burst_start : burst_start + burst_samples] = 1.0

        return x + noise * mask


# ─────────────────────────────────────────────
#  Baseline Augmentations
# ─────────────────────────────────────────────

class BaselineWander(ECGAugmentation):
    """Simulate baseline wander due to respiration and body movement.

    Real baseline wander:
    - Respiration: 0.15–0.4 Hz, amplitude 0.1–0.5 mV
    - Low-frequency body movement: 0.01–0.3 Hz
    """

    def __init__(
        self,
        freq_range: Tuple[float, float] = (0.05, 0.8),
        amplitude_range: Tuple[float, float] = (0.05, 0.4),
        n_components: int = 3,
        fs: float = 500.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.freq_min, self.freq_max = freq_range
        self.amp_min, self.amp_max = amplitude_range
        self.n_components = n_components
        self.fs = fs

    def apply(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[-1]
        t = np.arange(T) / self.fs
        wander = np.zeros(T, dtype=np.float32)

        for _ in range(self.n_components):
            freq = random.uniform(self.freq_min, self.freq_max)
            amp = random.uniform(self.amp_min, self.amp_max)
            phase = random.uniform(0, 2 * np.pi)
            wander += amp * np.sin(2 * np.pi * freq * t + phase)

        return (x + wander).astype(np.float32)


# ─────────────────────────────────────────────
#  Structural Augmentations
# ─────────────────────────────────────────────

class AmplitudeScaling(ECGAugmentation):
    """Scale signal amplitude to simulate electrode contact variation.

    Electrode-skin contact impedance varies with sweat, pressure,
    and skin condition, causing amplitude variations.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.7, 1.3),
        per_lead: bool = True,
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.scale_min, self.scale_max = scale_range
        self.per_lead = per_lead

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.per_lead and x.ndim == 2:
            # Different scale per lead
            scales = np.random.uniform(
                self.scale_min, self.scale_max, size=(x.shape[0], 1)
            ).astype(np.float32)
        else:
            scales = float(random.uniform(self.scale_min, self.scale_max))
        return (x * scales).astype(np.float32)


class LeadDropout(ECGAugmentation):
    """Randomly zero out entire ECG leads.

    Simulates:
    - Lead disconnection during monitoring
    - Missing leads in single/3-lead devices
    - Training the model to be robust to missing leads

    This is important for device deployment where not all 12 leads
    may always be available (e.g., patch ECG devices capture V1–V3 only).
    """

    def __init__(
        self,
        n_leads_to_drop_range: Tuple[int, int] = (1, 3),
        replace_with_noise: bool = False,
        p: float = 0.3,
    ) -> None:
        super().__init__(p)
        self.drop_min, self.drop_max = n_leads_to_drop_range
        self.replace_with_noise = replace_with_noise

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.ndim < 2:
            return x  # Single lead — skip
        x = x.copy()
        n_leads = x.shape[0]
        n_drop = random.randint(self.drop_min, min(self.drop_max, n_leads - 1))
        drop_leads = random.sample(range(n_leads), n_drop)

        for lead_idx in drop_leads:
            if self.replace_with_noise:
                x[lead_idx] = np.random.randn(x.shape[-1]) * 0.02
            else:
                x[lead_idx] = 0.0
        return x


class TimeShift(ECGAugmentation):
    """Random temporal shift (circular).

    Simulates recording start offset — the model should be
    invariant to absolute time position within a window.
    """

    def __init__(
        self,
        max_shift_ms: float = 200.0,
        fs: float = 500.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p)
        self.max_shift = int(max_shift_ms / 1000 * fs)

    def apply(self, x: np.ndarray) -> np.ndarray:
        shift = random.randint(-self.max_shift, self.max_shift)
        return np.roll(x, shift, axis=-1).astype(np.float32)


class TimeWarping(ECGAugmentation):
    """Local temporal warping to simulate heart rate variation.

    Stretches/compresses different regions of the ECG locally,
    simulating beat-to-beat HR variability and ectopic beats.

    Uses piecewise linear interpolation with random control points.
    """

    def __init__(
        self,
        n_knots: int = 4,
        warp_std: float = 0.1,
        fs: float = 500.0,
        p: float = 0.3,
    ) -> None:
        super().__init__(p)
        self.n_knots = n_knots
        self.warp_std = warp_std
        self.fs = fs

    def apply(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[-1]

        # Create random warp map
        knot_positions = np.linspace(0, T - 1, self.n_knots + 2)
        knot_warps = np.zeros(self.n_knots + 2)
        knot_warps[1:-1] = np.random.randn(self.n_knots) * self.warp_std * T

        # Smooth warp field
        warp_field = np.interp(
            np.arange(T), knot_positions, knot_positions + knot_warps
        )
        warp_field = np.clip(warp_field, 0, T - 1)

        # Apply warp via interpolation
        if x.ndim == 1:
            warped = np.interp(warp_field, np.arange(T), x)
        else:
            warped = np.array([
                np.interp(warp_field, np.arange(T), x[i])
                for i in range(x.shape[0])
            ])

        return warped.astype(np.float32)


class FrequencyMasking(ECGAugmentation):
    """Mask a frequency band in the ECG spectrum.

    Inspired by SpecAugment for audio, adapted for ECG.
    Encourages the model to not over-rely on specific frequency bands.
    """

    def __init__(
        self,
        max_mask_hz: float = 5.0,
        fs: float = 500.0,
        p: float = 0.2,
    ) -> None:
        super().__init__(p)
        self.max_mask_hz = max_mask_hz
        self.fs = fs

    def apply(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[-1]
        freqs = np.fft.rfftfreq(T, 1.0 / self.fs)

        # Choose random frequency band to zero out
        mask_width = random.uniform(0, self.max_mask_hz)
        mask_center = random.uniform(1.0, self.fs / 4)
        mask = (np.abs(freqs - mask_center) < mask_width / 2)

        # Apply mask in frequency domain
        X = np.fft.rfft(x, axis=-1)
        X[..., mask] = 0
        return np.fft.irfft(X, n=T, axis=-1).astype(np.float32)


class MixUp(ECGAugmentation):
    """MixUp augmentation for ECG (linearly interpolate two samples).

    While originally designed for images, MixUp has been shown effective
    for ECG classification (Cheng et al., 2021).
    """

    def __init__(self, alpha: float = 0.2, p: float = 0.2) -> None:
        super().__init__(p)
        self.alpha = alpha

    def apply_with_pair(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y1: np.ndarray,
        y2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        lam = np.random.beta(self.alpha, self.alpha)
        x_mix = lam * x1 + (1 - lam) * x2
        y_mix = lam * y1 + (1 - lam) * y2
        return x_mix.astype(np.float32), y_mix.astype(np.float32)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return x  # Requires pair — use apply_with_pair instead


# ─────────────────────────────────────────────
#  Augmentation Pipeline
# ─────────────────────────────────────────────

class ECGAugmentationPipeline:
    """Composed augmentation pipeline for ECG training.

    Provides two preset modes:
    - 'train': Strong augmentation for training
    - 'val': No augmentation (identity)
    - 'light': Light augmentation for semi-supervised or fine-tuning

    Can also be used to create contrastive views (two augmented versions
    of the same signal) for self-supervised pre-training (SimCLR, CLOCS).
    """

    PRESETS = {
        "train": [
            # Noise augmentations (moderate probability)
            ("gaussian_noise", GaussianNoise, {"snr_db_range": (25, 40), "p": 0.5}),
            ("powerline", PowerlineNoise, {"freq": 60.0, "p": 0.3}),
            ("emg_noise", EMGNoise, {"p": 0.2}),
            # Structural augmentations
            ("baseline_wander", BaselineWander, {"p": 0.5}),
            ("amplitude_scale", AmplitudeScaling, {"scale_range": (0.75, 1.25), "p": 0.5}),
            ("time_shift", TimeShift, {"max_shift_ms": 150.0, "p": 0.4}),
            ("time_warp", TimeWarping, {"p": 0.3}),
            ("lead_dropout", LeadDropout, {"n_leads_to_drop_range": (0, 2), "p": 0.25}),
            ("freq_mask", FrequencyMasking, {"p": 0.15}),
        ],
        "light": [
            ("gaussian_noise", GaussianNoise, {"snr_db_range": (30, 45), "p": 0.3}),
            ("baseline_wander", BaselineWander, {"p": 0.3}),
            ("amplitude_scale", AmplitudeScaling, {"scale_range": (0.85, 1.15), "p": 0.3}),
        ],
        "val": [],
    }

    def __init__(self, mode: str = "train", fs: float = 500.0) -> None:
        assert mode in self.PRESETS
        self.mode = mode
        self.fs = fs
        self.augmentations: List[ECGAugmentation] = []

        for name, cls, kwargs in self.PRESETS[mode]:
            if "fs" in cls.__init__.__code__.co_varnames:
                kwargs = {**kwargs, "fs": fs}
            self.augmentations.append(cls(**kwargs))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for aug in self.augmentations:
            x = aug(x)
        return x

    def get_two_views(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two independently augmented views of the same signal.

        Used for contrastive learning (CLOCS, SimCLR-ECG).
        """
        view1 = self(x.copy())
        view2 = self(x.copy())
        return view1, view2

    def __repr__(self) -> str:
        aug_names = [type(a).__name__ for a in self.augmentations]
        return f"ECGAugmentationPipeline(mode={self.mode}, augs={aug_names})"


if __name__ == "__main__":
    fs = 500.0
    # Create synthetic 12-lead ECG
    x = np.random.randn(12, 5000).astype(np.float32)

    pipeline = ECGAugmentationPipeline(mode="train", fs=fs)
    x_aug = pipeline(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {x_aug.shape}")
    print(f"Pipeline: {pipeline}")

    # Test contrastive views
    v1, v2 = pipeline.get_two_views(x)
    corr = np.corrcoef(v1.flatten(), v2.flatten())[0, 1]
    print(f"View correlation: {corr:.3f} (should be high but <1.0)")
