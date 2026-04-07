# %% [markdown]
# # CardioAI: ECG Data Exploration Notebook
#
# This notebook provides exploratory data analysis (EDA) of the PTB-XL dataset
# and demonstrates the CardioAI preprocessing and modeling pipeline.
#
# **Sections:**
# 1. Dataset Overview and Statistics
# 2. ECG Signal Visualization
# 3. Preprocessing Pipeline Demo
# 4. Augmentation Visualization
# 5. Class Imbalance Analysis
# 6. Model Architecture Inspection
# 7. Inference Demo and Grad-CAM Visualization
# 8. HRV Feature Analysis
#
# **Run with:** `jupytext --to notebook notebooks/ecg_exploration.py && jupyter notebook`

# %%
# Standard imports
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import signal as sp_signal

# Add project root to path
sys.path.insert(0, str(Path.cwd().parent))

# Plot settings
plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500  # PTB-XL sampling rate (Hz)

print("CardioAI Exploration Notebook")
print("=" * 40)

# %%
# ─────────────────────────────────────────────
# Section 1: Dataset Overview
# ─────────────────────────────────────────────
# %% [markdown]
# ## 1. Dataset Overview
#
# PTB-XL statistics:
# - 21,837 recordings, 18,885 unique patients
# - 500 Hz, 10 second, 12-lead ECG
# - 71 SCP-ECG diagnostic labels

# %%
# Simulate dataset statistics (replace with actual PTB-XL path)
# from src.data.ptbxl_dataset import PTBXLDataset, DIAGNOSTIC_SUPERCLASSES
# dataset = PTBXLDataset("/data/ptb-xl", split="all")

# Synthetic class distribution (mirrors actual PTB-XL)
class_names = ["NORM", "MI", "STTC", "CD", "HYP"]
class_counts = np.array([9528, 5486, 5235, 4907, 2655])
class_prevalence = class_counts / class_counts.sum()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of class distribution
colors = ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0"]
bars = axes[0].bar(class_names, class_counts, color=colors, alpha=0.85, edgecolor="white")
axes[0].set_title("PTB-XL Diagnostic Superclass Distribution", fontweight="bold", fontsize=13)
axes[0].set_ylabel("Number of Recordings")
axes[0].set_xlabel("Diagnostic Class")
for bar, count in zip(bars, class_counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"n={count:,}", ha="center", va="bottom", fontsize=9)

# Pie chart
axes[1].pie(class_counts, labels=class_names, autopct="%1.1f%%",
           colors=colors, startangle=90, pctdistance=0.85)
axes[1].set_title("Class Prevalence", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", bbox_inches="tight")
plt.show()

print("\nClass Distribution:")
for name, count, prev in zip(class_names, class_counts, class_prevalence):
    print(f"  {name:<8}: {count:6,} recordings ({prev*100:.1f}%)")
print(f"\nImbalance ratio (max/min): {class_counts.max()/class_counts.min():.1f}×")
print("→ Class weighting required during training")

# %%
# ─────────────────────────────────────────────
# Section 2: ECG Signal Visualization
# ─────────────────────────────────────────────
# %% [markdown]
# ## 2. ECG Signal Visualization
#
# 12-lead ECG: Standard clinical recording format
# - **Limb leads** (I, II, III, aVR, aVL, aVF): Frontal plane — cardiac axis
# - **Precordial leads** (V1–V6): Horizontal plane — anterior-posterior

# %%
def simulate_normal_ecg(fs=500, duration=10, hr_bpm=70, noise_level=0.02):
    """Simulate a physiologically plausible 12-lead ECG."""
    T = int(duration * fs)
    t = np.arange(T) / fs
    ecg = np.zeros((12, T))
    rr = 60.0 / hr_bpm  # RR interval in seconds

    # Lead-specific amplitudes (simplified)
    p_amp = np.array([0.15, 0.2, 0.05, -0.1, 0.05, 0.15, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15])
    r_amp = np.array([0.8, 1.2, 0.4, -0.6, 0.5, 0.9, -0.3, 0.3, 1.0, 1.5, 1.4, 1.1])
    s_amp = np.array([-0.1, -0.15, -0.05, 0.1, -0.05, -0.1, -0.5, -0.3, -0.1, -0.05, -0.05, -0.05])
    t_amp = np.array([0.2, 0.25, 0.08, -0.15, 0.1, 0.2, -0.1, 0.2, 0.3, 0.4, 0.35, 0.25])

    sigma_p, sigma_qrs, sigma_s, sigma_t = 0.04, 0.012, 0.015, 0.08

    for beat_t in np.arange(0.3, duration, rr):
        for lead in range(12):
            # P wave
            p_t = beat_t - 0.18
            ecg[lead] += p_amp[lead] * np.exp(-0.5 * ((t - p_t) / sigma_p) ** 2)
            # R wave
            ecg[lead] += r_amp[lead] * np.exp(-0.5 * ((t - beat_t) / sigma_qrs) ** 2)
            # S wave
            s_t = beat_t + 0.03
            ecg[lead] += s_amp[lead] * np.exp(-0.5 * ((t - s_t) / sigma_s) ** 2)
            # T wave
            t_t = beat_t + 0.22
            ecg[lead] += t_amp[lead] * np.exp(-0.5 * ((t - t_t) / sigma_t) ** 2)

    # Add realistic noise
    ecg += noise_level * np.random.randn(12, T)
    return ecg, t


# Generate synthetic ECG
ecg, t = simulate_normal_ecg(fs=FS, duration=10, hr_bpm=70)

# 12-lead plot
fig, axes = plt.subplots(12, 1, figsize=(18, 16), sharex=True)
fig.suptitle("Synthetic Normal 12-Lead ECG (70 bpm, 10 seconds)", fontweight="bold", fontsize=14)

lead_colors = ["#1f77b4"] * 6 + ["#d62728"] * 6  # Blue limb, red precordial

for i, (ax, name, color) in enumerate(zip(axes, LEAD_NAMES, lead_colors)):
    ax.plot(t, ecg[i], color=color, linewidth=0.8, alpha=0.9)
    ax.set_ylabel(name, rotation=0, labelpad=25, fontweight="bold")
    ax.set_ylim(-1.5, 2.0)
    # Add gridlines (ECG paper standard: 5mm major, 1mm minor)
    ax.set_yticks([-1.0, 0, 0.5, 1.0, 1.5])

axes[-1].set_xlabel("Time (seconds)", fontsize=12)
for ax in axes[:-1]:
    ax.tick_params(labelbottom=False)
plt.tight_layout()
plt.savefig("outputs/12lead_ecg.png", bbox_inches="tight")
plt.show()

# %%
# ─────────────────────────────────────────────
# Section 3: Preprocessing Pipeline Demo
# ─────────────────────────────────────────────
# %% [markdown]
# ## 3. ECG Preprocessing Pipeline
#
# Standard processing chain:
# 1. Notch filter: Remove 60 Hz powerline noise
# 2. Bandpass: 0.5–45 Hz (AHA clinical standard)
# 3. Baseline wander correction (spline method)
# 4. R-peak detection (Pan-Tompkins 1985)

# %%
from src.data.ecg_preprocessing import ECGFilters, PanTompkinsDetector, ECGPreprocessor

# Add artifacts to demo signal
ecg_noisy = ecg.copy()
# Powerline noise
ecg_noisy += 0.08 * np.sin(2 * np.pi * 60 * t)
# Baseline wander
ecg_noisy += 0.4 * np.sin(2 * np.pi * 0.2 * t)
# EMG burst
burst = np.zeros(len(t))
burst[2500:3500] = 0.2 * np.random.randn(1000)
ecg_noisy += burst

# Process
preprocessor = ECGPreprocessor(fs=FS)
result = preprocessor.process(ecg_noisy, return_metadata=True)
ecg_clean = result["signal"]
r_peaks = result["r_peaks"]
sqi = result["sqi"]

# Visualize Lead II preprocessing stages
fig, axes = plt.subplots(4, 1, figsize=(16, 10))
fig.suptitle("Preprocessing Pipeline: Lead II", fontweight="bold", fontsize=14)

axes[0].plot(t, ecg_noisy[1], color="#E53935", linewidth=0.7)
axes[0].set_title("Raw (with 60Hz noise + baseline wander + EMG burst)")

# After notch filter
filters = ECGFilters(FS)
after_notch = filters.remove_powerline(ecg_noisy[1:2], 60)[0]
axes[1].plot(t, after_notch, color="#FF7043", linewidth=0.7)
axes[1].set_title("After 60 Hz Notch Filter")

# After bandpass
after_bp = filters.bandpass(after_notch.reshape(1, -1))[0]
axes[2].plot(t, after_bp, color="#42A5F5", linewidth=0.7)
axes[2].set_title("After Bandpass (0.5–45 Hz)")

# Final cleaned + R-peaks
axes[3].plot(t, ecg_clean[1], color="#1B5E20", linewidth=0.8)
axes[3].plot(r_peaks / FS, ecg_clean[1, r_peaks], "rv", markersize=8, label=f"R-peaks (n={len(r_peaks)})")
axes[3].set_title("After Baseline Correction + R-Peak Detection (Pan-Tompkins)")
axes[3].legend()

for ax in axes:
    ax.set_ylabel("Amplitude (mV)")
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("outputs/preprocessing_pipeline.png", bbox_inches="tight")
plt.show()

print(f"\nPreprocessing Results:")
print(f"  R-peaks detected: {len(r_peaks)}")
print(f"  Estimated HR: {60.0 * len(r_peaks) / t[-1]:.1f} bpm")
print(f"  Signal Quality:")
for k, v in sqi.items():
    if isinstance(v, (int, float)):
        print(f"    {k}: {v:.3f}")

# %%
# ─────────────────────────────────────────────
# Section 4: Augmentation Visualization
# ─────────────────────────────────────────────
# %% [markdown]
# ## 4. ECG-Specific Data Augmentation
#
# Augmentations preserve clinically relevant ECG features while
# simulating real-world acquisition variability.

# %%
from src.data.augmentation import (
    GaussianNoise, BaselineWander, AmplitudeScaling,
    TimeWarping, LeadDropout, ECGAugmentationPipeline
)

# Use Lead II for demonstration
lead_ii = ecg[1].copy()

# Apply individual augmentations
fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True)
fig.suptitle("ECG Augmentation Suite (Lead II)", fontweight="bold", fontsize=14)

axes[0].plot(t, lead_ii, color="#1976D2", linewidth=0.8)
axes[0].set_title("Original", fontweight="bold")

aug_configs = [
    ("Gaussian Noise (SNR=25dB)", GaussianNoise(snr_db_range=(25, 25), p=1.0), "#E53935"),
    ("Baseline Wander (f=0.3Hz, A=0.3mV)", BaselineWander(
        freq_range=(0.3, 0.3), amplitude_range=(0.3, 0.3), n_components=1, p=1.0), "#FF6F00"),
    ("Amplitude Scaling (×0.75)", AmplitudeScaling(scale_range=(0.75, 0.75), per_lead=False, p=1.0), "#2E7D32"),
    ("Time Warping", TimeWarping(n_knots=4, warp_std=0.08, p=1.0), "#6A1B9A"),
    ("Combined Training Augmentation", None, "#00838F"),
]

np.random.seed(42)
for ax, (title, aug, color) in zip(axes[1:], aug_configs):
    if aug is None:
        pipeline = ECGAugmentationPipeline(mode="train", fs=FS)
        aug_signal = pipeline(lead_ii.reshape(1, -1)).flatten()
    else:
        aug_signal = aug.apply(lead_ii)
    ax.plot(t, aug_signal, color=color, linewidth=0.8)
    ax.set_title(title, fontweight="bold")

for ax in axes:
    ax.set_ylabel("mV")
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("outputs/augmentations.png", bbox_inches="tight")
plt.show()

# %%
# ─────────────────────────────────────────────
# Section 5: Class Imbalance & Weighting
# ─────────────────────────────────────────────
# %% [markdown]
# ## 5. Class Imbalance Analysis
#
# PTB-XL has 3.6× imbalance between NORM and HYP classes.
# Class-weighted loss prevents the model from ignoring rare classes.

# %%
from src.data.ptbxl_dataset import DIAGNOSTIC_SUPERCLASSES

# Simulate class weights
n_samples = 17471  # PTB-XL train split
class_counts_train = np.array([8097, 4663, 4450, 4170, 2257])

# Compute weights (inverse frequency, capped at 10×)
weights = n_samples / (len(class_names) * (class_counts_train + 1e-6))
weights_capped = np.clip(weights, 1.0, 10.0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Class Imbalance and Weighting Strategy", fontweight="bold", fontsize=13)

# Training counts
axes[0].bar(class_names, class_counts_train, color=colors, alpha=0.85, edgecolor="white")
axes[0].set_title("Training Set Class Counts")
axes[0].set_ylabel("Count")

# Class weights
axes[1].bar(class_names, weights_capped, color=colors, alpha=0.85, edgecolor="white")
axes[1].set_title("Inverse-Frequency Class Weights\n(capped at 10×)")
axes[1].set_ylabel("Weight")
axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Unweighted baseline")
axes[1].legend()

# Expected gradient contribution
gradient_raw = class_counts_train / class_counts_train.sum()
gradient_weighted = (class_counts_train * weights_capped) / (class_counts_train * weights_capped).sum()
x = np.arange(len(class_names))
w = 0.35
axes[2].bar(x - w/2, gradient_raw * 100, w, label="Unweighted", color="#90CAF9", edgecolor="white")
axes[2].bar(x + w/2, gradient_weighted * 100, w, label="Weighted", color=colors, edgecolor="white", alpha=0.85)
axes[2].set_xticks(x)
axes[2].set_xticklabels(class_names)
axes[2].set_title("Gradient Contribution per Class (%)")
axes[2].set_ylabel("% of total gradient")
axes[2].legend()

plt.tight_layout()
plt.savefig("outputs/class_imbalance.png", bbox_inches="tight")
plt.show()

print("Class Weights (inverse frequency, capped at 10×):")
for name, w in zip(class_names, weights_capped):
    print(f"  {name}: {w:.2f}×")

# %%
# ─────────────────────────────────────────────
# Section 6: Model Architecture Inspection
# ─────────────────────────────────────────────
# %% [markdown]
# ## 6. Model Architecture
#
# ECGResNet architecture with multi-scale blocks and SE attention.

# %%
import torch
from src.models.ecg_resnet import ECGResNet
from src.models.ecg_transformer import ECGTransformer
from src.models.ecg_lstm import ECGBiLSTM

# Instantiate models
models = {
    "ECGResNet": ECGResNet(num_classes=5),
    "ECGTransformer": ECGTransformer(num_classes=5, n_temporal_layers=3, n_lead_layers=3),
    "ECGBiLSTM": ECGBiLSTM(num_classes=5),
}

print("Model Parameter Counts:")
print("-" * 50)
x_dummy = torch.randn(1, 12, 5000)
for name, model in models.items():
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        out = model(x_dummy)
    print(f"  {name:<20}: {total:>8,} params | Output shape: {out['probs'].shape}")
print("-" * 50)

# Parameter distribution by layer type
import collections
param_by_type = collections.defaultdict(int)
model = models["ECGResNet"]
for name, param in model.named_parameters():
    if "conv" in name:
        param_by_type["Conv1d"] += param.numel()
    elif "bn" in name:
        param_by_type["BatchNorm"] += param.numel()
    elif "fc" in name or "classifier" in name or "linear" in name.lower():
        param_by_type["Linear"] += param.numel()
    else:
        param_by_type["Other"] += param.numel()

fig, ax = plt.subplots(figsize=(8, 5))
types = list(param_by_type.keys())
counts = [param_by_type[t] for t in types]
wedge_colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]
wedges, texts, autotexts = ax.pie(counts, labels=types, autopct="%1.1f%%",
                                   colors=wedge_colors[:len(types)], startangle=90)
ax.set_title("ECGResNet: Parameter Distribution by Layer Type", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/model_params.png", bbox_inches="tight")
plt.show()

# %%
# ─────────────────────────────────────────────
# Section 7: Mock Inference and Grad-CAM
# ─────────────────────────────────────────────
# %% [markdown]
# ## 7. Inference and Saliency Maps
#
# Grad-CAM shows which ECG time segments drive each prediction —
# critical for clinician trust and FDA regulatory review.

# %%
from src.models.ecg_resnet import GradCAMWrapper

# Simulate model prediction with Grad-CAM
model = ECGResNet(num_classes=5)
model.eval()
grad_cam = GradCAMWrapper(model, target_layer_idx=-1)

# Use our synthetic ECG
x_tensor = torch.tensor(ecg[np.newaxis].copy(), dtype=torch.float32)
# Normalize
x_tensor = (x_tensor - x_tensor.mean(dim=-1, keepdim=True)) / (x_tensor.std(dim=-1, keepdim=True) + 1e-8)

probs, cam = grad_cam(x_tensor, class_idx=0)

print("\nModel Predictions (untrained model — uniform probs expected):")
for i, name in enumerate(class_names):
    print(f"  {name}: {probs[0, i].item():.4f}")

# Plot ECG with Grad-CAM saliency
fig, axes = plt.subplots(3, 1, figsize=(16, 10))
fig.suptitle("Grad-CAM: Temporal Saliency Map (Lead II)\n"
             "[Untrained model — illustrative only]", fontweight="bold", fontsize=13)

# Raw ECG
axes[0].plot(t, ecg[1], color="#1976D2", linewidth=0.8)
axes[0].set_title("Input ECG — Lead II")
axes[0].set_ylabel("Amplitude (mV)")

# CAM overlay
cam_np = cam[0].detach().numpy()
axes[1].plot(t, ecg[1], color="#1976D2", linewidth=0.8, alpha=0.5)
sc = axes[1].scatter(t, ecg[1], c=cam_np, cmap="RdYlGn_r", s=2, alpha=0.7)
plt.colorbar(sc, ax=axes[1], label="Saliency")
axes[1].set_title("ECG with Grad-CAM Saliency (High = Most Influential Timestep)")
axes[1].set_ylabel("Amplitude (mV)")

# CAM as bar/fill
axes[2].fill_between(t, cam_np, alpha=0.7, color="#E53935")
axes[2].set_title("Temporal Saliency Map")
axes[2].set_ylabel("Saliency")
axes[2].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("outputs/gradcam.png", bbox_inches="tight")
plt.show()

# %%
# ─────────────────────────────────────────────
# Section 8: HRV Feature Analysis
# ─────────────────────────────────────────────
# %% [markdown]
# ## 8. Heart Rate Variability (HRV) Analysis
#
# HRV features are key for:
# - AF detection (irregular RR intervals)
# - Autonomic function assessment
# - Risk stratification post-MI

# %%
from src.data.ecg_preprocessing import compute_hrv_features

# Simulate different HRV profiles
np.random.seed(123)

def simulate_rr_intervals(mean_rr_s, sdnn_ms, n_beats=100):
    """Simulate RR interval series."""
    rr = mean_rr_s + np.random.randn(n_beats) * sdnn_ms / 1000
    r_peaks = (np.cumsum(rr) * FS).astype(int)
    return r_peaks, rr * 1000  # return in ms

profiles = {
    "Healthy (Normal SR)": simulate_rr_intervals(0.85, 45, 100),    # 70bpm, SDNN=45ms
    "Sinus Bradycardia": simulate_rr_intervals(1.15, 35, 80),       # 52bpm
    "Sinus Tachycardia": simulate_rr_intervals(0.55, 25, 120),      # 109bpm
    "Atrial Fibrillation": simulate_rr_intervals(0.75, 130, 100),   # Highly irregular
}

hrv_data = []
for name, (r_peaks, rr_ms) in profiles.items():
    hrv = compute_hrv_features(r_peaks, FS)
    hrv["profile"] = name
    hrv["rr_ms"] = rr_ms
    hrv_data.append(hrv)

# HRV comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("HRV Profiles: Time and Frequency Domain Features", fontweight="bold", fontsize=14)

profile_colors = ["#1976D2", "#4CAF50", "#FF5722", "#9C27B0"]
metric_pairs = [
    ("SDNN (ms)", [h["sdnn"] for h in hrv_data], axes[0, 0]),
    ("RMSSD (ms)", [h["rmssd"] for h in hrv_data], axes[0, 1]),
    ("pNN50 (%)", [h.get("pnn50", 0) for h in hrv_data], axes[1, 0]),
    ("Mean HR (bpm)", [h["mean_hr"] for h in hrv_data], axes[1, 1]),
]

for label, values, ax in metric_pairs:
    bars = ax.bar(
        [h["profile"].split(" (")[0][:15] for h in hrv_data],
        values,
        color=profile_colors,
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_title(label, fontweight="bold")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
               f"{val:.1f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("outputs/hrv_analysis.png", bbox_inches="tight")
plt.show()

print("\nHRV Summary:")
print(f"{'Profile':<30} {'Mean HR':>9} {'SDNN':>8} {'RMSSD':>8} {'pNN50':>8}")
print("-" * 68)
for h in hrv_data:
    print(f"{h['profile']:<30} {h['mean_hr']:>9.1f} {h['sdnn']:>8.1f} "
          f"{h['rmssd']:>8.1f} {h.get('pnn50', 0):>8.1f}")

# %%
# ─────────────────────────────────────────────
# Section 9: Alert Fatigue Simulation
# ─────────────────────────────────────────────
# %% [markdown]
# ## 9. Alert Fatigue Reduction Analysis
#
# Demonstrating the alert suppression pipeline's effect on false alarm rate.

# %%
from src.evaluation.cardiac_metrics import compute_alert_fatigue_metrics

# Simulate a classifier's predictions at different confidence thresholds
np.random.seed(42)
N = 1000
prevalence = 0.10  # 10% arrhythmia prevalence

y_true = (np.random.rand(N) < prevalence).astype(int)
# Good model: true positives score ~0.85, false positives score ~0.35
y_score = np.where(
    y_true == 1,
    np.clip(np.random.beta(7, 2, N), 0, 1),  # Positive: high scores
    np.clip(np.random.beta(2, 7, N), 0, 1),  # Negative: low scores
)

thresholds = np.linspace(0.1, 0.9, 50)
far_list, sens_list, ppv_list = [], [], []

for thr in thresholds:
    m = compute_alert_fatigue_metrics(y_true, y_score, thr, recording_duration_hours=8.0)
    far_list.append(m["false_alarm_rate_per_hour"])
    sens_list.append(m["sensitivity"])
    ppv_list.append(m["ppv"])

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Alert Fatigue Tradeoffs vs. Classification Threshold",
             fontweight="bold", fontsize=13)

axes[0].plot(thresholds, sens_list, color="#1976D2", linewidth=2)
axes[0].axhline(0.90, color="red", linestyle="--", alpha=0.7, label="90% target sensitivity")
axes[0].axvline(thresholds[np.argmin(np.abs(np.array(sens_list) - 0.90))],
               color="orange", linestyle=":", alpha=0.7)
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Sensitivity (Recall)")
axes[0].set_title("Sensitivity vs. Threshold")
axes[0].legend()

axes[1].plot(thresholds, far_list, color="#E53935", linewidth=2)
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("False Alarms per Hour")
axes[1].set_title("False Alarm Rate vs. Threshold\n(8-hour monitoring session)")

axes[2].plot(thresholds, ppv_list, color="#2E7D32", linewidth=2)
axes[2].set_xlabel("Threshold")
axes[2].set_ylabel("Positive Predictive Value (PPV)")
axes[2].set_title("PPV (Alarm Reliability) vs. Threshold")

plt.tight_layout()
plt.savefig("outputs/alert_fatigue_analysis.png", bbox_inches="tight")
plt.show()

# Find threshold achieving 90% sensitivity
idx_90 = np.argmin(np.abs(np.array(sens_list) - 0.90))
print(f"\nAt 90% sensitivity operating point:")
print(f"  Threshold: {thresholds[idx_90]:.2f}")
print(f"  False alarm rate: {far_list[idx_90]:.2f}/hour")
print(f"  PPV: {ppv_list[idx_90]:.1%}")

idx_95 = np.argmin(np.abs(np.array(sens_list) - 0.95))
print(f"\nAt 95% sensitivity operating point:")
print(f"  Threshold: {thresholds[idx_95]:.2f}")
print(f"  False alarm rate: {far_list[idx_95]:.2f}/hour")
print(f"  PPV: {ppv_list[idx_95]:.1%}")

# %%
# Create outputs directory if needed
import os
os.makedirs("outputs", exist_ok=True)
print("\n✓ Exploration complete. Saved figures to outputs/")
print("Files generated:")
for fname in ["class_distribution.png", "12lead_ecg.png", "preprocessing_pipeline.png",
              "augmentations.png", "class_imbalance.png", "model_params.png",
              "gradcam.png", "hrv_analysis.png", "alert_fatigue_analysis.png"]:
    print(f"  outputs/{fname}")
