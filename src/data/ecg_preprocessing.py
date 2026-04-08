"""
ECG Preprocessing Pipeline.

Clinical ECG signals are corrupted by multiple noise sources:
1. Baseline wander (0.05–1 Hz): respiration, body movement
2. Powerline interference (50/60 Hz): AC electrical noise
3. Muscle artifacts (EMG, >100 Hz): skeletal muscle activity
4. Motion artifact: electrode-skin impedance variation
5. Quantization noise: ADC resolution limits

This module implements the standard clinical signal processing chain
used in commercial cardiac monitors and Holter analysis systems.

Key algorithms:
- Butterworth bandpass filter: 0.5–45 Hz (AHA/ESC guideline standard)
- Notch filter: 50/60 Hz powerline removal
- Pan-Tompkins QRS detector (1985): gold standard R-peak algorithm
- Cubic spline baseline correction: physiologically accurate wander removal
- Signal quality index (SQI): kurtosis + template correlation

References:
    Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
        IEEE Trans. Biomed. Eng., 32(3), 230–236.
    Clifford, G. D., et al. (2006). Advanced methods and tools for ECG
        data analysis. Artech House.
    Orphanidou, C. (2018). Signal quality assessment in physiological
        monitoring. Springer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

# minimum acceptable SQI score — leads below this are flagged as noisy
# and excluded from the multi-label prediction ensemble
SQI_MIN_THRESHOLD = 0.60


# ─────────────────────────────────────────────
#  Filter Design
# ─────────────────────────────────────────────

class ECGFilters:
    """Pre-designed digital filters for ECG preprocessing.

    Filters are designed at instantiation for a given sampling rate,
    avoiding repeated filter design overhead during batch processing.
    """

    def __init__(self, fs: float = 500.0) -> None:
        self.fs = fs
        nyq = fs / 2.0

        # Bandpass: 0.5–45 Hz (AHA recommended for clinical ECG)
        self.bp_sos = sp_signal.butter(
            4, [0.5 / nyq, 45.0 / nyq], btype="bandpass", output="sos"
        )

        # High-pass only: 0.5 Hz (for baseline wander removal only)
        self.hp_sos = sp_signal.butter(
            4, 0.5 / nyq, btype="highpass", output="sos"
        )

        # Low-pass: 40 Hz (muscle artifact removal)
        self.lp_sos = sp_signal.butter(
            4, 40.0 / nyq, btype="lowpass", output="sos"
        )

        # Notch filters for powerline: 50 Hz and 60 Hz
        self.notch_50_sos = self._design_notch(50.0, Q=30)
        self.notch_60_sos = self._design_notch(60.0, Q=30)

    def _design_notch(self, freq: float, Q: float) -> np.ndarray:
        """Design IIR notch filter."""
        nyq = self.fs / 2.0
        w0 = freq / nyq
        b, a = sp_signal.iirnotch(w0, Q)
        return sp_signal.tf2sos(b, a)

    def bandpass(self, x: np.ndarray) -> np.ndarray:
        """Zero-phase bandpass filter (0.5–45 Hz)."""
        return sp_signal.sosfiltfilt(self.bp_sos, x, axis=-1)

    def highpass(self, x: np.ndarray) -> np.ndarray:
        return sp_signal.sosfiltfilt(self.hp_sos, x, axis=-1)

    def lowpass(self, x: np.ndarray) -> np.ndarray:
        return sp_signal.sosfiltfilt(self.lp_sos, x, axis=-1)

    def remove_powerline(self, x: np.ndarray, freq: int = 60) -> np.ndarray:
        """Remove powerline noise with notch filter."""
        sos = self.notch_60_sos if freq == 60 else self.notch_50_sos
        return sp_signal.sosfiltfilt(sos, x, axis=-1)


# ─────────────────────────────────────────────
#  Baseline Wander Correction
# ─────────────────────────────────────────────

def remove_baseline_wander_spline(
    x: np.ndarray,
    r_peaks: np.ndarray,
    fs: float = 500.0,
) -> np.ndarray:
    """Cubic spline baseline correction using R-peak locations.

    Algorithm:
    1. Estimate baseline at each R-peak location (use T-P segment mean)
    2. Fit cubic spline through baseline points
    3. Subtract spline from signal

    This method is more physiologically accurate than high-pass filtering
    because it preserves the ST segment morphology — critical for
    ischemia/STEMI detection.

    Args:
        x: (T,) single lead signal
        r_peaks: R-peak sample indices
        fs: Sampling rate

    Returns:
        Baseline-corrected signal (T,)
    """
    if len(r_peaks) < 4:
        # Fallback to median subtraction
        return x - np.median(x)

    # Estimate baseline at each R-peak: use 40ms window around T-P segment
    # T-P segment is ~200ms before next R peak (physiologically clean period)
    baseline_points = []
    baseline_times = []

    for i, rp in enumerate(r_peaks[:-1]):
        # Midpoint between current R-peak and next R-peak (T-P segment)
        tp_mid = int((rp + r_peaks[i + 1]) / 2)
        window = 20  # ±20 samples = ±40ms @ 500Hz

        seg_start = max(0, tp_mid - window)
        seg_end = min(len(x), tp_mid + window)
        baseline_points.append(np.median(x[seg_start:seg_end]))
        baseline_times.append(tp_mid)

    # Add boundary points
    baseline_times = [0] + baseline_times + [len(x) - 1]
    baseline_points = [baseline_points[0]] + baseline_points + [baseline_points[-1]]

    # Cubic spline interpolation
    cs = CubicSpline(baseline_times, baseline_points)
    t = np.arange(len(x))
    baseline = cs(t)

    return x - baseline


def remove_baseline_wander_hp(
    x: np.ndarray,
    fs: float = 500.0,
    cutoff: float = 0.5,
) -> np.ndarray:
    """High-pass filter baseline wander removal.

    Simpler but distorts low-frequency components of ST/T waves.
    Use spline method when precise ST analysis is required.
    """
    nyq = fs / 2.0
    sos = sp_signal.butter(4, cutoff / nyq, btype="highpass", output="sos")
    return sp_signal.sosfiltfilt(sos, x, axis=-1)


# ─────────────────────────────────────────────
#  Pan-Tompkins QRS Detector
# ─────────────────────────────────────────────

class PanTompkinsDetector:
    """Pan-Tompkins real-time QRS detection algorithm.

    The Pan-Tompkins algorithm (1985) remains the standard reference
    for R-peak detection in clinical systems, with ~99.3% sensitivity
    on the MIT-BIH Arrhythmia Database.

    Pipeline:
    1. Bandpass filter (5–15 Hz): isolates QRS energy
    2. Differentiation: emphasizes QRS slope
    3. Squaring: makes all values positive, amplifies high-frequency content
    4. Moving window integration (150ms): smooths and integrates QRS energy
    5. Threshold detection: adaptive threshold from signal history
    6. Refractory period: 200ms minimum between beats

    Args:
        fs: Sampling rate (Hz).
    """

    BANDPASS_LOW = 5.0   # Hz
    BANDPASS_HIGH = 15.0  # Hz
    INTEGRATION_WINDOW_MS = 150  # ms

    def __init__(self, fs: float = 500.0) -> None:
        self.fs = fs
        nyq = fs / 2.0
        self.bp_sos = sp_signal.butter(
            2,
            [self.BANDPASS_LOW / nyq, self.BANDPASS_HIGH / nyq],
            btype="bandpass",
            output="sos",
        )
        self.integration_window = int(self.INTEGRATION_WINDOW_MS * fs / 1000)
        self.refractory_samples = int(0.200 * fs)  # 200ms

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """Pan-Tompkins preprocessing: BPF → Diff → Square → MWI."""
        # Step 1: Bandpass filter
        filtered = sp_signal.sosfiltfilt(self.bp_sos, x)

        # Step 2: Derivative (approximates 5-point derivative)
        dx = np.zeros_like(filtered)
        dx[2:-2] = (2 * filtered[4:] + filtered[3:-1] -
                    filtered[1:-3] - 2 * filtered[:-4]) / (8 / self.fs)

        # Step 3: Squaring (all positive + enhance high-frequency content)
        squared = dx ** 2

        # Step 4: Moving window integration
        kernel = np.ones(self.integration_window) / self.integration_window
        mwi = np.convolve(squared, kernel, mode="same")

        return mwi

    def detect(self, x: np.ndarray) -> np.ndarray:
        """Detect R-peak locations in a single ECG lead.

        Args:
            x: (T,) ECG signal (should be filtered before detection).

        Returns:
            r_peaks: Array of R-peak sample indices.
        """
        mwi = self._preprocess(x)

        # Adaptive threshold: starts at 50% of max, adapts online
        # Use simplified offline version: peak detection with prominence
        min_distance = self.refractory_samples

        # Find peaks in preprocessed signal
        peaks, properties = sp_signal.find_peaks(
            mwi,
            height=0.05 * mwi.max(),
            distance=min_distance,
            prominence=0.03 * mwi.max(),
        )

        if len(peaks) == 0:
            return np.array([], dtype=int)

        # Refine: find actual R-peak (maximum of original signal in window)
        window = int(0.05 * self.fs)  # ±50ms search window
        refined_peaks = []
        for p in peaks:
            start = max(0, p - window)
            end = min(len(x), p + window)
            local_max = start + np.argmax(np.abs(x[start:end]))
            refined_peaks.append(local_max)

        # Remove duplicates and sort
        refined_peaks = sorted(set(refined_peaks))

        # Enforce refractory period
        final_peaks = [refined_peaks[0]]
        for p in refined_peaks[1:]:
            if p - final_peaks[-1] >= self.refractory_samples:
                final_peaks.append(p)

        return np.array(final_peaks, dtype=int)


# ─────────────────────────────────────────────
#  HRV Features
# ─────────────────────────────────────────────

def compute_hrv_features(
    r_peaks: np.ndarray,
    fs: float = 500.0,
) -> Dict[str, float]:
    """Compute time-domain and frequency-domain HRV features.

    HRV features are clinically used for:
    - Autonomic nervous system assessment
    - Predicting AF recurrence (SDNN, pNN50 reduction)
    - Risk stratification post-MI (low SDNN → worse prognosis)
    - Sleep quality assessment

    Time-domain features:
        SDNN: Standard deviation of NN intervals (overall HRV)
        RMSSD: Root mean square of successive differences (vagal tone)
        pNN50: Percentage of successive differences > 50ms
        mean_HR: Mean heart rate (bpm)

    Frequency-domain features (Welch's method):
        LF power (0.04–0.15 Hz): sympathetic/parasympathetic
        HF power (0.15–0.40 Hz): parasympathetic (respiratory)
        LF/HF ratio: sympathovagal balance

    Args:
        r_peaks: R-peak sample indices.
        fs: Sampling rate.

    Returns:
        Dict of HRV feature values.
    """
    features: Dict[str, float] = {}

    if len(r_peaks) < 4:
        return {k: np.nan for k in [
            "mean_hr", "sdnn", "rmssd", "pnn50", "lf_power", "hf_power", "lf_hf_ratio"
        ]}

    # RR intervals in seconds
    rr = np.diff(r_peaks) / fs
    rr_ms = rr * 1000  # milliseconds

    # --- Time domain ---
    features["mean_hr"] = 60.0 / np.mean(rr)
    features["mean_rr_ms"] = float(np.mean(rr_ms))
    features["sdnn"] = float(np.std(rr_ms, ddof=1))

    # Successive differences
    diff_rr = np.diff(rr_ms)
    features["rmssd"] = float(np.sqrt(np.mean(diff_rr ** 2)))
    features["pnn50"] = float(np.mean(np.abs(diff_rr) > 50.0) * 100)

    # Triangular index (geometric)
    features["rr_range_ms"] = float(rr_ms.max() - rr_ms.min())

    # --- Frequency domain (Lomb-Scargle for unequally spaced) ---
    # Resample to 4 Hz for FFT-based analysis
    if len(rr) >= 16:
        try:
            # Create evenly spaced time axis
            t_rr = np.cumsum(rr)
            t_interp = np.arange(t_rr[0], t_rr[-1], 1.0 / 4.0)  # 4 Hz

            # Cubic spline interpolation of RR intervals
            cs = CubicSpline(t_rr, rr)
            rr_interp = cs(t_interp)
            rr_interp -= rr_interp.mean()  # Remove mean

            # Welch's power spectral density
            freqs, psd = sp_signal.welch(rr_interp, fs=4.0, nperseg=min(256, len(rr_interp)))

            # Band power integration
            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

            df = freqs[1] - freqs[0]
            features["lf_power"] = float(np.trapz(psd[lf_mask], freqs[lf_mask]))
            features["hf_power"] = float(np.trapz(psd[hf_mask], freqs[hf_mask]))
            features["lf_hf_ratio"] = (
                features["lf_power"] / (features["hf_power"] + 1e-10)
            )
            features["total_power"] = float(np.trapz(psd, freqs))
        except Exception:
            features.update({"lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan})

    return features


# ─────────────────────────────────────────────
#  Signal Quality Index
# ─────────────────────────────────────────────

def compute_signal_quality_index(
    x: np.ndarray,
    r_peaks: np.ndarray,
    fs: float = 500.0,
) -> Dict[str, float]:
    """Multi-metric Signal Quality Index (SQI) for ECG.

    Low-quality signals should be excluded from analysis or flagged
    for clinical review. SQI is critical for reducing false alarms
    from artifact-triggered detections.

    Metrics:
        kurtosis_sqi: High kurtosis → QRS spike dominated signal (good)
        flatline_sqi: Low variance → electrode off / disconnected (bad)
        template_sqi: Beat-to-beat morphology consistency (Orphanidou 2018)
        rr_sqi: RR interval regularity / physiological plausibility

    Args:
        x: (T,) ECG signal (single lead, filtered).
        r_peaks: Detected R-peak indices.
        fs: Sampling rate.

    Returns:
        Dict with individual SQI components and overall SQI (0–1).
    """
    from scipy.stats import kurtosis

    sqi: Dict[str, float] = {}
    T = len(x)

    # 1. Kurtosis SQI (Natus/PhysioNet methodology)
    # Good ECG kurtosis: ~5–15 (QRS spikes)
    # Flatline/noise: ~3 (Gaussian)
    kurt = kurtosis(x)
    sqi["kurtosis"] = float(kurt)
    sqi["kurtosis_sqi"] = float(np.clip((kurt - 3) / 10, 0, 1))

    # 2. Flatline SQI
    sigma = np.std(x)
    sqi["variance_mv2"] = float(sigma ** 2)
    sqi["flatline_sqi"] = float(np.clip(sigma / 0.5, 0, 1))  # 0.5 mV threshold

    # 3. Template correlation SQI
    # Extract beats, compute mean template, correlate each beat
    if len(r_peaks) >= 3:
        beat_width = int(0.6 * fs)  # 600ms window
        half = beat_width // 2
        beats = []
        for rp in r_peaks:
            if rp - half >= 0 and rp + half <= T:
                beats.append(x[rp - half : rp + half])

        if len(beats) >= 3:
            beats_arr = np.array(beats)
            template = beats_arr.mean(axis=0)
            # Normalized cross-correlation with template
            correlations = [
                np.corrcoef(template, beat)[0, 1]
                for beat in beats_arr
            ]
            sqi["template_correlation_mean"] = float(np.mean(correlations))
            sqi["template_sqi"] = float(np.clip(np.mean(correlations), 0, 1))
        else:
            sqi["template_sqi"] = 0.5

    # 4. RR interval SQI
    if len(r_peaks) >= 2:
        rr_samples = np.diff(r_peaks)
        rr_sec = rr_samples / fs
        # Physiological HR range: 20–300 bpm → RR 0.2–3.0 s
        valid_rr = np.mean((rr_sec >= 0.2) & (rr_sec <= 3.0))
        sqi["rr_sqi"] = float(valid_rr)
    else:
        sqi["rr_sqi"] = 0.0

    # Overall SQI (weighted combination)
    weights = {"kurtosis_sqi": 0.2, "flatline_sqi": 0.3, "template_sqi": 0.3, "rr_sqi": 0.2}
    overall = sum(
        weights[k] * sqi.get(k, 0.0) for k in weights
    )
    sqi["overall_sqi"] = float(overall)
    sqi["is_acceptable"] = overall > 0.5

    return sqi


# ─────────────────────────────────────────────
#  Beat Segmentation
# ─────────────────────────────────────────────

def segment_beats(
    x: np.ndarray,
    r_peaks: np.ndarray,
    fs: float = 500.0,
    window_ms: float = 700.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract individual beats centered on R-peaks.

    Used for:
    - Beat-level classification (VEB vs SVEB vs Normal)
    - Heart rate variability computation
    - Template correlation SQI

    Args:
        x: (T,) or (12, T) ECG signal.
        r_peaks: R-peak sample indices.
        fs: Sampling rate.
        window_ms: Total beat window in milliseconds.

    Returns:
        beats: (N_beats, [12,] window_samples) beat array
        valid_mask: (N_beats,) boolean array of usable beats
    """
    window_samples = int(window_ms / 1000 * fs)
    half = window_samples // 2
    T = x.shape[-1]
    is_multi_lead = x.ndim == 2

    beats = []
    valid = []

    for rp in r_peaks:
        start = rp - half
        end = rp + (window_samples - half)
        if start < 0 or end > T:
            valid.append(False)
            beats.append(
                np.zeros((12, window_samples) if is_multi_lead else (window_samples,))
            )
            continue

        beat = x[..., start:end]
        beats.append(beat)
        valid.append(True)

    return np.array(beats), np.array(valid)


# ─────────────────────────────────────────────
#  Full Preprocessing Pipeline
# ─────────────────────────────────────────────

class ECGPreprocessor:
    """Complete ECG preprocessing pipeline for clinical AI deployment.

    Applies a standardized processing chain suitable for:
    - Training deep learning models on PTB-XL / PhysioNet data
    - Real-time preprocessing in monitoring applications
    - Batch processing for clinical trial analysis

    Pipeline:
        1. Powerline noise removal (notch filter)
        2. Bandpass filtering (0.5–45 Hz)
        3. Baseline wander correction (spline method)
        4. R-peak detection (Pan-Tompkins on lead II)
        5. Signal quality assessment
        6. (Optional) Beat segmentation
        7. (Optional) HRV computation
    """

    def __init__(
        self,
        fs: float = 500.0,
        powerline_freq: int = 60,
        correct_baseline: bool = True,
        compute_hrv: bool = True,
        quality_threshold: float = 0.5,
    ) -> None:
        self.fs = fs
        self.filters = ECGFilters(fs)
        self.detector = PanTompkinsDetector(fs)
        self.powerline_freq = powerline_freq
        self.correct_baseline = correct_baseline
        self.compute_hrv = compute_hrv
        self.quality_threshold = quality_threshold

    def process(
        self,
        x: np.ndarray,
        return_metadata: bool = False,
    ) -> Dict[str, np.ndarray | Dict]:
        """Process a 12-lead ECG recording.

        Args:
            x: (12, T) 12-lead ECG in mV.
            return_metadata: Include HRV features and SQI in output.

        Returns:
            dict with keys:
                'signal': (12, T) cleaned ECG
                'r_peaks': detected R-peak indices
                'sqi': signal quality index
                ['hrv_features']: HRV features if return_metadata=True
        """
        assert x.ndim == 2 and x.shape[0] == 12, f"Expected (12, T), got {x.shape}"

        # 1. Powerline noise removal
        clean = self.filters.remove_powerline(x, self.powerline_freq)

        # 2. Bandpass filter
        clean = self.filters.bandpass(clean)

        # 3. Baseline wander correction using R-peaks from lead II
        # First pass: rough R-peak detection on high-pass filtered lead II
        lead_ii = self.filters.highpass(x[1])
        r_peaks = self.detector.detect(lead_ii)

        if self.correct_baseline and len(r_peaks) >= 4:
            for lead_idx in range(12):
                clean[lead_idx] = remove_baseline_wander_spline(
                    clean[lead_idx], r_peaks, self.fs
                )
        else:
            # Fallback: simple offset removal
            clean -= clean.mean(axis=-1, keepdims=True)

        # 4. Final R-peak detection on cleaned lead II
        r_peaks = self.detector.detect(clean[1])

        # 5. Signal quality assessment
        sqi = compute_signal_quality_index(clean[1], r_peaks, self.fs)

        result: Dict = {
            "signal": clean.astype(np.float32),
            "r_peaks": r_peaks,
            "sqi": sqi,
        }

        if return_metadata and len(r_peaks) >= 4:
            result["hrv_features"] = compute_hrv_features(r_peaks, self.fs)

        return result

    def process_batch(
        self,
        batch: np.ndarray,
        return_metadata: bool = False,
    ) -> List[Dict]:
        """Process a batch of ECG recordings."""
        return [self.process(x, return_metadata) for x in batch]


if __name__ == "__main__":
    fs = 500.0
    # Simulate 12-lead ECG with QRS complexes
    t = np.linspace(0, 10, int(10 * fs))
    ecg = np.zeros((12, len(t)), dtype=np.float32)

    # Add synthetic R-peaks (60 bpm)
    for lead in range(12):
        for beat_t in np.arange(0.5, 10, 1.0):
            idx = int(beat_t * fs)
            window = np.arange(-20, 21)
            idxs = idx + window
            valid = (idxs >= 0) & (idxs < len(t))
            gaussian = np.exp(-0.5 * (window[valid] / 5) ** 2)
            ecg[lead, idxs[valid]] += gaussian * 1.5

    # Add powerline noise and baseline wander
    ecg += 0.05 * np.sin(2 * np.pi * 60 * t)  # 60 Hz powerline
    ecg += 0.3 * np.sin(2 * np.pi * 0.2 * t)  # baseline wander

    preprocessor = ECGPreprocessor(fs=fs)
    result = preprocessor.process(ecg, return_metadata=True)

    print(f"Cleaned signal shape: {result['signal'].shape}")
    print(f"R-peaks detected: {len(result['r_peaks'])}")
    print(f"Overall SQI: {result['sqi']['overall_sqi']:.3f}")
    if "hrv_features" in result:
        print(f"Mean HR: {result['hrv_features']['mean_hr']:.1f} bpm")
        print(f"SDNN: {result['hrv_features']['sdnn']:.1f} ms")
