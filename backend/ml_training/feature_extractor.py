"""
EchoShield Acoustic Feature Extraction Engine.

Extracts an 89-dimensional discriminative acoustic feature representation for
voice classification across three distinct categories:
  - Class 0: AI_GENERATED (AI cloned voices, synthetic neural TTS, vocoder artifacts)
  - Class 1: ROBOCALL_IVR (Automated announcements, telephony band-limited IVR prompts)
  - Class 2: NATURAL_HUMAN (Organic human speech with natural prosody and dynamics)
"""

import os
import io
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.signal
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    [f"mfcc_{i}_mean" for i in range(20)] +
    [f"mfcc_{i}_std" for i in range(20)] +
    [f"delta_mfcc_{i}_std" for i in range(10)] +
    [f"delta2_mfcc_{i}_std" for i in range(5)] +
    ["spectral_centroid_mean", "spectral_centroid_std"] +
    ["spectral_bandwidth_mean", "spectral_bandwidth_std"] +
    ["spectral_rolloff_85_mean", "spectral_rolloff_85_std"] +
    ["spectral_rolloff_95_mean", "spectral_rolloff_95_std"] +
    ["spectral_flatness_mean", "spectral_flatness_std"] +
    [f"spectral_contrast_b{i}_mean" for i in range(6)] +
    [f"spectral_contrast_b{i}_std" for i in range(6)] +
    ["zcr_mean", "zcr_std"] +
    ["rms_mean", "rms_std", "rms_var"] +
    ["band_ratio_sub300", "band_ratio_telephony_300_3400", "band_ratio_above_3800"] +
    ["voiced_fraction", "f0_mean", "f0_std", "f0_cov", "pitch_jitter", "amplitude_shimmer"]
)

NUM_FEATURES = len(FEATURE_NAMES)
assert NUM_FEATURES == 91, f"Expected 91 features, got {NUM_FEATURES}"


def fast_pitch_and_prosody(y: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Fast autocorrelation-based pitch tracking and prosody extraction.
    Computes voiced fraction, mean F0, F0 std, jitter proxy, and shimmer proxy in <15ms.
    """
    frame_len = int(sr * 0.030)  # 30ms window
    hop_len = int(sr * 0.015)    # 15ms hop
    min_lag = int(sr / 500.0)    # 500 Hz max pitch
    max_lag = int(sr / 60.0)     # 60 Hz min pitch

    n_frames = max(1, (len(y) - frame_len) // hop_len)
    f0_list = []
    energy_list = []

    for i in range(n_frames):
        frame = y[i * hop_len: i * hop_len + frame_len]
        energy = float(np.sqrt(np.mean(frame ** 2)))
        energy_list.append(energy)

        if energy > 0.01:
            frame_win = frame * np.hanning(len(frame))
            corr = np.correlate(frame_win, frame_win, mode='full')
            corr = corr[len(frame_win) - 1:]

            if len(corr) > max_lag:
                corr_slice = corr[min_lag:max_lag]
                peak_idx = int(np.argmax(corr_slice) + min_lag)
                peak_val = corr[peak_idx]
                r0 = corr[0] if corr[0] > 1e-7 else 1e-7
                norm_peak = peak_val / r0

                if norm_peak > 0.35:
                    f0 = sr / float(peak_idx)
                    f0_list.append(f0)
                else:
                    f0_list.append(np.nan)
            else:
                f0_list.append(np.nan)
        else:
            f0_list.append(np.nan)

    f0_arr = np.array(f0_list, dtype=np.float32)
    valid_f0 = f0_arr[~np.isnan(f0_arr)]
    voiced_fraction = float(len(valid_f0) / max(1, len(f0_arr)))

    if len(valid_f0) > 3:
        f0_mean = float(np.mean(valid_f0))
        f0_std = float(np.std(valid_f0))
        f0_cov = float(f0_std / (f0_mean + 1e-6))
        f0_diffs = np.abs(np.diff(valid_f0))
        jitter = float(np.mean(f0_diffs) / (f0_mean + 1e-6))
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_cov = 0.0
        jitter = 0.0

    if len(energy_list) > 3:
        eng_arr = np.array(energy_list, dtype=np.float32)
        eng_mean = float(np.mean(eng_arr)) + 1e-6
        eng_diffs = np.abs(np.diff(eng_arr))
        shimmer = float(np.mean(eng_diffs) / eng_mean)
    else:
        shimmer = 0.0

    return {
        "voiced_fraction": voiced_fraction,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_cov": f0_cov,
        "pitch_jitter": jitter,
        "amplitude_shimmer": shimmer
    }


def extract_features_from_audio(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract the complete 89-dimensional acoustic feature vector from an audio signal.
    Audio signal MUST be mono float32 at 16000 Hz.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    # Basic signal validation
    if y is None or len(y) < int(sr * 0.25):
        return features

    # Ensure peak amplitude normalization
    peak = np.max(np.abs(y))
    if peak > 0.005:
        norm_y = y / peak
    else:
        norm_y = y

    try:
        # 1. MFCCs (20 coefficients)
        mfcc = librosa.feature.mfcc(y=norm_y, sr=sr, n_mfcc=20, n_fft=512, hop_length=256)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # 2. Delta MFCCs (10 coefficients)
        delta_mfcc = librosa.feature.delta(mfcc[:10])
        delta_std = np.std(delta_mfcc, axis=1)

        # 3. Delta-Delta MFCCs (5 coefficients)
        delta2_mfcc = librosa.feature.delta(mfcc[:5], order=2)
        delta2_std = np.std(delta2_mfcc, axis=1)

        # 4. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=norm_y, sr=sr, n_fft=512, hop_length=256)
        cent_mean = float(np.mean(centroid))
        cent_std = float(np.std(centroid))

        # 5. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=norm_y, sr=sr, n_fft=512, hop_length=256)
        bw_mean = float(np.mean(bandwidth))
        bw_std = float(np.std(bandwidth))

        # 6. Spectral Rolloff (85% and 95%)
        rolloff_85 = librosa.feature.spectral_rolloff(y=norm_y, sr=sr, roll_percent=0.85, n_fft=512, hop_length=256)
        ro85_mean = float(np.mean(rolloff_85))
        ro85_std = float(np.std(rolloff_85))

        rolloff_95 = librosa.feature.spectral_rolloff(y=norm_y, sr=sr, roll_percent=0.95, n_fft=512, hop_length=256)
        ro95_mean = float(np.mean(rolloff_95))
        ro95_std = float(np.std(rolloff_95))

        # 7. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=norm_y, n_fft=512, hop_length=256)
        flat_mean = float(np.mean(flatness))
        flat_std = float(np.std(flatness))

        # 8. Spectral Contrast (6 sub-bands, fmin=100)
        contrast = librosa.feature.spectral_contrast(y=norm_y, sr=sr, n_bands=5, fmin=100.0, n_fft=512, hop_length=256)
        cont_mean = np.mean(contrast, axis=1)
        cont_std = np.std(contrast, axis=1)

        # 9. Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(norm_y, frame_length=512, hop_length=256)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # 10. RMS Energy dynamics
        rms = librosa.feature.rms(y=norm_y, frame_length=512, hop_length=256)
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_var = float(np.var(rms))

        # 11. Telephony Band Ratios (via FFT)
        fft_mag = np.abs(np.fft.rfft(norm_y))
        freqs = np.fft.rfftfreq(len(norm_y), 1.0 / sr)
        total_energy = float(np.sum(fft_mag)) + 1e-10

        sub300_idx = np.where(freqs < 300)[0]
        telephony_idx = np.where((freqs >= 300) & (freqs <= 3400))[0]
        above3800_idx = np.where(freqs >= 3800)[0]

        band_sub300 = float(np.sum(fft_mag[sub300_idx]) / total_energy) if len(sub300_idx) > 0 else 0.0
        band_telephony = float(np.sum(fft_mag[telephony_idx]) / total_energy) if len(telephony_idx) > 0 else 0.0
        band_above3800 = float(np.sum(fft_mag[above3800_idx]) / total_energy) if len(above3800_idx) > 0 else 0.0

        # 12. Pitch and Prosody
        prosody = fast_pitch_and_prosody(norm_y, sr=sr)

        # Pack into feature array
        idx = 0
        features[idx:idx+20] = mfcc_mean; idx += 20
        features[idx:idx+20] = mfcc_std; idx += 20
        features[idx:idx+10] = delta_std; idx += 10
        features[idx:idx+5] = delta2_std; idx += 5
        features[idx] = cent_mean; features[idx+1] = cent_std; idx += 2
        features[idx] = bw_mean; features[idx+1] = bw_std; idx += 2
        features[idx] = ro85_mean; features[idx+1] = ro85_std; idx += 2
        features[idx] = ro95_mean; features[idx+1] = ro95_std; idx += 2
        features[idx] = flat_mean; features[idx+1] = flat_std; idx += 2
        features[idx:idx+6] = cont_mean[:6]; idx += 6
        features[idx:idx+6] = cont_std[:6]; idx += 6
        features[idx] = zcr_mean; features[idx+1] = zcr_std; idx += 2
        features[idx] = rms_mean; features[idx+1] = rms_std; features[idx+2] = rms_var; idx += 3
        features[idx] = band_sub300; features[idx+1] = band_telephony; features[idx+2] = band_above3800; idx += 3
        features[idx] = prosody["voiced_fraction"]
        features[idx+1] = prosody["f0_mean"]
        features[idx+2] = prosody["f0_std"]
        features[idx+3] = prosody["f0_cov"]
        features[idx+4] = prosody["pitch_jitter"]
        features[idx+5] = prosody["amplitude_shimmer"]
        idx += 6

        assert idx == NUM_FEATURES

    except Exception as e:
        logger.warning(f"Error during feature extraction: {e}")

    # Sanitize NaNs and Infs
    features = np.nan_to_num(features, nan=0.0, posinf=1e4, neginf=-1e4)
    return features


def load_audio_normalized(
    file_path: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    target_sr: int = 16000
) -> Tuple[Optional[np.ndarray], int]:
    """
    Safely load audio from file or bytes across multiple formats:
    WAV, MP3, OGG, FLAC, M4A, WebM.
    Converts stereo to mono and resamples to target_sr (16000 Hz).
    """
    raw_bytes = audio_bytes
    if not raw_bytes and file_path and os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            logger.warning(f"Failed to read audio file {file_path}: {e}")

    if not raw_bytes or len(raw_bytes) < 44:
        return None, target_sr

    # 1. Try soundfile (handles WAV, MP3, OGG, FLAC directly)
    try:
        data, sr = sf.read(io.BytesIO(raw_bytes), dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return data, target_sr
    except Exception:
        pass

    # 2. Try file path with soundfile
    if file_path and os.path.exists(file_path):
        try:
            data, sr = sf.read(file_path, dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            return data, target_sr
        except Exception:
            pass

    # 3. Try librosa.load (handles WebM, M4A via ffmpeg/soundfile)
    try:
        y, sr = librosa.load(file_path or io.BytesIO(raw_bytes), sr=target_sr, mono=True)
        return y.astype(np.float32), target_sr
    except Exception:
        pass

    # 4. Fallback: Parse linear 16-bit PCM if RIFF header or raw stream
    try:
        offset = 44 if raw_bytes[:4] == b"RIFF" else 0
        pcm = raw_bytes[offset:]
        if len(pcm) % 2 != 0:
            pcm = pcm[:-1]
        if len(pcm) >= 160:
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            return arr, target_sr
    except Exception as e:
        logger.warning(f"Linear PCM fallback failed: {e}")

    return None, target_sr
