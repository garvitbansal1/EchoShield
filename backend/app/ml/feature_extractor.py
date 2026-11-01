"""
EchoShield Acoustic Feature Extraction Engine.

Extracts an 91-dimensional discriminative acoustic feature representation for
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
import scipy.fftpack

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except (ImportError, OSError):
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDFILE_AVAILABLE = False


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


def _extract_scipy_features(norm_y: np.ndarray, sr: int = 16000) -> Tuple[Any, ...]:
    """Lightweight pure scipy/numpy feature extraction when librosa is not installed."""
    n_fft = 512
    hop_length = 256
    _, _, Zxx = scipy.signal.stft(norm_y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    mag = np.abs(Zxx)
    power = mag ** 2
    n_freqs = mag.shape[0]

    # MFCCs via Mel filterbank + DCT
    n_mels = 40
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fbank = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if k < n_freqs and f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_pts[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if k < n_freqs and f_m_plus != f_m:
                fbank[m - 1, k] = (bin_pts[m + 1] - k) / (f_m_plus - f_m)

    mel_eng = np.dot(fbank, power)
    mel_eng = np.where(mel_eng == 0, 1e-10, mel_eng)
    log_mel = np.log(mel_eng)
    mfcc = scipy.fftpack.dct(log_mel, type=2, axis=0, norm='ortho')[:20]
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    delta_mfcc = np.diff(mfcc[:10], axis=1) if mfcc.shape[1] > 1 else np.zeros((10, 1))
    delta_std = np.std(delta_mfcc, axis=1)

    delta2_mfcc = np.diff(delta_mfcc[:5], axis=1) if delta_mfcc.shape[1] > 1 else np.zeros((5, 1))
    delta2_std = np.std(delta2_mfcc, axis=1)

    freqs = np.linspace(0, sr / 2, n_freqs)
    total_mag = np.sum(mag, axis=0) + 1e-10

    # Spectral centroid
    centroid = np.sum(freqs[:, None] * mag, axis=0) / total_mag
    cent_mean = float(np.mean(centroid))
    cent_std = float(np.std(centroid))

    # Spectral bandwidth
    deviation = (freqs[:, None] - centroid[None, :]) ** 2
    bandwidth = np.sqrt(np.sum(deviation * mag, axis=0) / total_mag)
    bw_mean = float(np.mean(bandwidth))
    bw_std = float(np.std(bandwidth))

    # Spectral rolloff
    cum_power = np.cumsum(power, axis=0)
    tot_power = cum_power[-1, :] + 1e-10
    ro85_bins = np.argmax(cum_power >= 0.85 * tot_power, axis=0)
    ro95_bins = np.argmax(cum_power >= 0.95 * tot_power, axis=0)
    ro85 = freqs[ro85_bins]
    ro95 = freqs[ro95_bins]
    ro85_mean, ro85_std = float(np.mean(ro85)), float(np.std(ro85))
    ro95_mean, ro95_std = float(np.mean(ro95)), float(np.std(ro95))

    # Spectral flatness
    geom_mean = np.exp(np.mean(np.log(power + 1e-12), axis=0))
    arith_mean = np.mean(power, axis=0) + 1e-12
    flatness = geom_mean / arith_mean
    flat_mean, flat_std = float(np.mean(flatness)), float(np.std(flatness))

    # Contrast approximation (6 subbands)
    cont_mean = np.zeros(6, dtype=np.float32)
    cont_std = np.zeros(6, dtype=np.float32)
    band_size = n_freqs // 6
    for b in range(6):
        b_slice = mag[b * band_size: (b + 1) * band_size, :]
        if b_slice.shape[0] > 0:
            peak_val = np.percentile(b_slice, 95, axis=0)
            valley_val = np.percentile(b_slice, 5, axis=0) + 1e-6
            c_val = np.log(peak_val / valley_val + 1e-6)
            cont_mean[b] = float(np.mean(c_val))
            cont_std[b] = float(np.std(c_val))

    # ZCR
    if len(norm_y) >= 512:
        frames = np.lib.stride_tricks.sliding_window_view(norm_y, 512)[::256]
        zcr_vals = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
        zcr_mean = float(np.mean(zcr_vals))
        zcr_std = float(np.std(zcr_vals))
        rms_vals = np.sqrt(np.mean(frames ** 2, axis=1))
        rms_mean = float(np.mean(rms_vals))
        rms_std = float(np.std(rms_vals))
        rms_var = float(np.var(rms_vals))
    else:
        zcr_mean, zcr_std = 0.05, 0.01
        rms_mean, rms_std, rms_var = 0.1, 0.02, 0.0004

    return (
        mfcc_mean, mfcc_std, delta_std, delta2_std,
        cent_mean, cent_std, bw_mean, bw_std,
        ro85_mean, ro85_std, ro95_mean, ro95_std,
        flat_mean, flat_std, cont_mean, cont_std,
        zcr_mean, zcr_std, rms_mean, rms_std, rms_var
    )


def extract_features_from_audio(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract the complete 91-dimensional acoustic feature vector from an audio signal.
    Audio signal MUST be mono float32 at 16000 Hz.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    # Basic signal validation
    if y is None or len(y) < int(sr * 0.25):
        return features

    # Ensure peak amplitude normalization
    peak = np.max(np.abs(y))
    norm_y = (y / peak) if peak > 0.005 else y

    try:
        if LIBROSA_AVAILABLE:
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
        else:
            (
                mfcc_mean, mfcc_std, delta_std, delta2_std,
                cent_mean, cent_std, bw_mean, bw_std,
                ro85_mean, ro85_std, ro95_mean, ro95_std,
                flat_mean, flat_std, cont_mean, cont_std,
                zcr_mean, zcr_std, rms_mean, rms_std, rms_var
            ) = _extract_scipy_features(norm_y, sr=sr)

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

    # Ensure no NaN or Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def load_audio_normalized(file_path: Optional[str] = None, raw_bytes: Optional[bytes] = None, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], int]:
    """Load and normalize audio from path or byte stream."""
    if SOUNDFILE_AVAILABLE:
        try:
            data, sr = sf.read(file_path or io.BytesIO(raw_bytes), dtype='float32')
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != target_sr and LIBROSA_AVAILABLE:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            return data.astype(np.float32), target_sr
        except Exception:
            pass

    if LIBROSA_AVAILABLE:
        try:
            y, sr = librosa.load(file_path or io.BytesIO(raw_bytes), sr=target_sr, mono=True)
            return y.astype(np.float32), target_sr
        except Exception:
            pass

    # Built-in PCM parser fallback
    data_bytes = raw_bytes
    if not data_bytes and file_path and os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                data_bytes = f.read()
        except Exception:
            pass

    if data_bytes and len(data_bytes) >= 44:
        try:
            offset = 44 if data_bytes[:4] == b"RIFF" else 0
            pcm = data_bytes[offset:]
            if len(pcm) % 2 != 0:
                pcm = pcm[:-1]
            if len(pcm) >= 160:
                y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                return y, target_sr
        except Exception:
            pass

    return None, target_sr
