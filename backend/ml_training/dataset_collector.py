"""
EchoShield ML Training: Dataset Collector and Validator.

Downloads and organizes legally usable, publicly accessible voice datasets across 3 classes:
  1. AI-Generated / Cloned Voice: ASVspoof 2015 & 2017 evaluation spoofed partitions (neural TTS, vocoders, voice cloning).
  2. Automated Robocall / IVR: FreeSWITCH Callie standard automated telephony IVR prompts.
  3. Natural Human Voice: CMU Arctic natural human speech corpus (male & female speakers) + ASVspoof authentic speech.
"""

import os
import io
import sys
import ssl
import tarfile
import logging
import urllib.request
import pandas as pd
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
AI_DIR = os.path.join(DATA_RAW_DIR, "ai_generated")
IVR_DIR = os.path.join(DATA_RAW_DIR, "robocall_ivr")
HUMAN_DIR = os.path.join(DATA_RAW_DIR, "natural_human")

for d in [AI_DIR, IVR_DIR, HUMAN_DIR]:
    os.makedirs(d, exist_ok=True)

ssl_ctx = ssl.create_default_context()


def collect_ai_voices(target_count=200):
    """Collect genuine AI-generated and cloned voice samples from ASVspoof 2015 and 2017 benchmarks."""
    logger.info("--- Collecting Class 1: AI-Generated / Cloned Voice Dataset ---")
    current_files = [f for f in os.listdir(AI_DIR) if f.endswith(".wav")]
    if len(current_files) >= target_count:
        logger.info(f"AI voice dataset already populated with {len(current_files)} samples.")
        return

    saved = 0
    hf_sources = [
        "https://huggingface.co/datasets/DynamicSuperb/SpoofDetection_ASVspoof2015/resolve/main/data/test-00000-of-00001-4a252b1e92dc86af.parquet",
        "https://huggingface.co/datasets/DynamicSuperb/SpoofDetection_ASVspoof2017/resolve/main/data/test-00000-of-00001-e7ed034c56ededdd.parquet"
    ]

    for url in hf_sources:
        if saved >= target_count:
            break
        try:
            logger.info(f"Downloading benchmark partition from {url.split('/')[-1]}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60, context=ssl_ctx) as resp:
                df = pd.read_parquet(io.BytesIO(resp.read()))

            spoofed = df[df['label'] == 'spoofed']
            logger.info(f"Found {len(spoofed)} spoofed/synthetic samples in partition.")

            for _, row in spoofed.iterrows():
                if saved >= target_count:
                    break
                fname = f"ai_{row['file']}"
                out_path = os.path.join(AI_DIR, fname)
                audio_entry = row['audio']
                if isinstance(audio_entry, dict) and 'bytes' in audio_entry and audio_entry['bytes']:
                    with open(out_path, "wb") as f:
                        f.write(audio_entry['bytes'])
                    saved += 1
        except Exception as e:
            logger.error(f"Error fetching from {url}: {e}")

    logger.info(f"Successfully collected {saved} AI-generated voice samples in {AI_DIR}.")


def collect_ivr_robocalls(target_count=200):
    """Collect authentic automated IVR and robocall telephony announcements from FreeSWITCH Callie."""
    logger.info("--- Collecting Class 2: Automated Robocall / IVR Dataset ---")
    current_files = [f for f in os.listdir(IVR_DIR) if f.endswith(".wav")]
    if len(current_files) >= target_count:
        logger.info(f"IVR robocall dataset already populated with {len(current_files)} samples.")
        return

    ivr_tar_url = "https://files.freeswitch.org/releases/sounds/freeswitch-sounds-en-us-callie-8000-1.0.51.tar.gz"
    logger.info("Downloading FreeSWITCH Callie automated IVR soundpack (18.6 MB)...")
    req = urllib.request.Request(ivr_tar_url, headers={'User-Agent': 'Mozilla/5.0'})
    saved = 0

    try:
        with urllib.request.urlopen(req, timeout=90, context=ssl_ctx) as resp:
            tar_bytes = resp.read()
        logger.info("Soundpack downloaded. Extracting telephony prompt samples...")

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if saved >= target_count:
                    break
                if member.isfile() and member.name.endswith(".wav"):
                    # Pick diverse IVR prompts from ivr, voicemail, conference, and digits
                    f = tar.extractfile(member)
                    if f:
                        wav_data = f.read()
                        if len(wav_data) > 4000:  # Skip tiny silence stubs (<0.25s)
                            out_fname = f"ivr_{saved:04d}_{os.path.basename(member.name)}"
                            out_path = os.path.join(IVR_DIR, out_fname)
                            with open(out_path, "wb") as out_f:
                                out_f.write(wav_data)
                            saved += 1

        logger.info(f"Successfully collected {saved} IVR / automated robocall samples in {IVR_DIR}.")
    except Exception as e:
        logger.error(f"Failed to download/extract IVR soundpack: {e}")


def collect_natural_human(target_count=200):
    """Collect natural human speech samples from CMU Arctic corpus and authentic ASVspoof speech."""
    logger.info("--- Collecting Class 3: Natural Human Voice Dataset ---")
    current_files = [f for f in os.listdir(HUMAN_DIR) if f.endswith(".wav")]
    if len(current_files) >= target_count:
        logger.info(f"Natural human dataset already populated with {len(current_files)} samples.")
        return

    saved = 0

    # 1. Download authentic samples from ASVspoof 2017
    try:
        url17 = "https://huggingface.co/datasets/DynamicSuperb/SpoofDetection_ASVspoof2017/resolve/main/data/test-00000-of-00001-e7ed034c56ededdd.parquet"
        req = urllib.request.Request(url17, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60, context=ssl_ctx) as resp:
            df17 = pd.read_parquet(io.BytesIO(resp.read()))
        auth = df17[df17['label'] == 'authentic']
        for _, row in auth.iterrows():
            fname = f"human_asv_{row['file']}"
            out_path = os.path.join(HUMAN_DIR, fname)
            audio_entry = row['audio']
            if isinstance(audio_entry, dict) and 'bytes' in audio_entry and audio_entry['bytes']:
                with open(out_path, "wb") as f:
                    f.write(audio_entry['bytes'])
                saved += 1
        logger.info(f"Added {saved} authentic speech samples from ASVspoof.")
    except Exception as e:
        logger.warning(f"Could not load ASVspoof authentic: {e}")

    # 2. Download natural speech from CMU Arctic (female: slt, male: bdl)
    speakers = [("slt", "cmu_us_slt_arctic"), ("bdl", "cmu_us_bdl_arctic")]
    needed_per_speaker = (target_count - saved + 1) // 2

    for spk_code, spk_dir in speakers:
        logger.info(f"Fetching natural human speech for speaker '{spk_code}' from CMU Arctic...")
        spk_saved = 0
        for i in range(1, needed_per_speaker + 50):
            if saved >= target_count or spk_saved >= needed_per_speaker:
                break
            idx_str = f"a{i:04d}"
            url = f"http://www.festvox.org/cmu_arctic/cmu_arctic/{spk_dir}/wav/arctic_{idx_str}.wav"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as resp:
                    wav_data = resp.read()
                if len(wav_data) > 8000:
                    out_fname = f"human_cmu_{spk_code}_{idx_str}.wav"
                    out_path = os.path.join(HUMAN_DIR, out_fname)
                    with open(out_path, "wb") as out_f:
                        out_f.write(wav_data)
                    saved += 1
                    spk_saved += 1
            except Exception:
                continue

    logger.info(f"Successfully collected {saved} natural human voice samples in {HUMAN_DIR}.")


def validate_collected_datasets():
    """Verify that all downloaded files can be opened and contain valid speech."""
    logger.info("--- Validating Collected Audio Samples ---")
    counts = {}
    for label, dir_path in [("AI_GENERATED", AI_DIR), ("ROBOCALL_IVR", IVR_DIR), ("NATURAL_HUMAN", HUMAN_DIR)]:
        valid = 0
        corrupt = 0
        files = [f for f in os.listdir(dir_path) if f.endswith(".wav")]
        for f in files:
            p = os.path.join(dir_path, f)
            try:
                data, sr = sf.read(p)
                duration = len(data) / float(sr)
                if duration >= 0.25 and np.max(np.abs(data)) > 0.005:
                    valid += 1
                else:
                    os.remove(p)
                    corrupt += 1
            except Exception:
                try:
                    os.remove(p)
                except OSError:
                    pass
                corrupt += 1
        counts[label] = valid
        logger.info(f"Class '{label}': {valid} valid audio samples (removed {corrupt} corrupt/short).")

    return counts


if __name__ == "__main__":
    collect_ai_voices(target_count=200)
    collect_ivr_robocalls(target_count=200)
    collect_natural_human(target_count=200)
    summary = validate_collected_datasets()
    print("\nDataset Collection Summary:", summary)
