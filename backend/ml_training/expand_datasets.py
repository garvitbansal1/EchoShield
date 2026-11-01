import os
import io
import sys
import tarfile
import urllib.request
import ssl
import soundfile as sf
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
IVR_DIR = os.path.join(DATA_RAW, "robocall_ivr")
HUMAN_DIR = os.path.join(DATA_RAW, "natural_human")
ssl_ctx = ssl.create_default_context()


def expand_ivr_to_300():
    current = len([f for f in os.listdir(IVR_DIR) if f.endswith(".wav")])
    if current >= 300:
        print(f"IVR already has {current} samples.")
        return
    print(f"Expanding IVR dataset from {current} to 300...")
    tar_url = "https://files.freeswitch.org/releases/sounds/freeswitch-sounds-en-us-callie-8000-1.0.51.tar.gz"
    req = urllib.request.Request(tar_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=90, context=ssl_ctx) as resp:
        tar_bytes = resp.read()

    saved = current
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if saved >= 300:
                break
            if member.isfile() and member.name.endswith(".wav"):
                f = tar.extractfile(member)
                if f:
                    wav_data = f.read()
                    if len(wav_data) > 3000:
                        out_fname = f"ivr_exp_{saved:04d}_{os.path.basename(member.name)}"
                        out_path = os.path.join(IVR_DIR, out_fname)
                        if not os.path.exists(out_path):
                            with open(out_path, "wb") as out_f:
                                out_f.write(wav_data)
                            saved += 1
    print(f"IVR collection complete. Total: {saved} samples.")


def expand_human_to_300():
    current = len([f for f in os.listdir(HUMAN_DIR) if f.endswith(".wav")])
    if current >= 300:
        print(f"Natural human already has {current} samples.")
        return
    print(f"Expanding Natural Human dataset from {current} to 300...")
    needed = 300 - current
    speakers = [("slt", "cmu_us_slt_arctic"), ("bdl", "cmu_us_bdl_arctic")]
    per_spk = (needed + 1) // 2
    saved = current

    for spk_code, spk_dir in speakers:
        spk_saved = 0
        for i in range(120, 250):
            if saved >= 300 or spk_saved >= per_spk:
                break
            idx_str = f"a{i:04d}"
            url = f"http://www.festvox.org/cmu_arctic/cmu_arctic/{spk_dir}/wav/arctic_{idx_str}.wav"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10, context=ssl_ctx) as resp:
                    wav_data = resp.read()
                if len(wav_data) > 8000:
                    out_fname = f"human_exp_{spk_code}_{idx_str}.wav"
                    out_path = os.path.join(HUMAN_DIR, out_fname)
                    if not os.path.exists(out_path):
                        with open(out_path, "wb") as out_f:
                            out_f.write(wav_data)
                        saved += 1
                        spk_saved += 1
            except Exception:
                continue

    print(f"Natural Human collection complete. Total: {saved} samples.")


if __name__ == "__main__":
    expand_ivr_to_300()
    expand_human_to_300()

