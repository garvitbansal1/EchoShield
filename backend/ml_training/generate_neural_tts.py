"""
Generate modern neural TTS voice samples to expand Class 0 (AI_GENERATED).
Includes modern neural TTS voices across diverse scam scripts, conversational
phrases, international accents (US, UK, CA, AU, IN), and telephony-channel variants.
"""

import os
import io
import time
import numpy as np
import scipy.signal
import soundfile as sf
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.join(BASE_DIR, "data", "raw", "ai_generated")
os.makedirs(AI_DIR, exist_ok=True)

SCAM_SCRIPTS = [
    "This is Officer John Davis from the Federal Bureau of Investigation. We have detected suspicious fraudulent transactions under your name.",
    "Your bank account has been flagged for unauthorized foreign access. Please confirm your account details immediately to prevent suspension.",
    "This is an urgent notification from the Internal Revenue Service. A warrant has been issued for your arrest due to unpaid federal taxes.",
    "Dear customer, your credit card ending in 4128 was charged 899 dollars at Apple Store. Press one to speak with fraud prevention.",
    "Hello, this is Microsoft technical support. We have detected critical malware and remote access trojans on your computer system.",
    "Your social security number has been compromised in a money laundering investigation. You must verify your identity immediately.",
    "Your package could not be delivered due to an incorrect shipping address. Click the link sent to your phone or pay the redelivery fee now.",
    "Congratulations, you have been selected as the grand prize winner of two million dollars. Contact our claims department to pay the processing fee.",
    "This is the legal department of the United States Treasury. Transfer your liquid funds to a secure government locker account right now.",
    "We have identified suspicious wire transfers originating from your cryptocurrency wallet. Confirm your secret seed phrase to secure your funds.",
    "Your electricity service will be disconnected within thirty minutes due to past due invoices. Pay immediately using a prepaid card.",
    "This is an automated call from Amazon fraud detection. An iPhone 15 Pro Max order was placed on your account. Press two to cancel.",
    "Attention, your medical insurance benefits will be terminated today unless you verify your policyholder details and date of birth.",
    "This is customer service regarding your loan application. You are pre-approved for fifty thousand dollars with zero percent interest.",
    "Urgent security notice from PayPal. A login was attempted from an unrecognized device in Moscow. Verify your password to secure your account."
]

CONVERSATIONAL_SCRIPTS = [
    "Thank you for calling customer service. How can I help you today with your billing inquiry?",
    "I understand your concern and I would be glad to look into this account issue for you right away.",
    "Could you please repeat your account number and the last four digits of your registered phone number?",
    "Our system is currently processing your request. Please stay on the line while I pull up your records.",
    "I appreciate your patience. We have resolved the connectivity issue and your service is now fully active.",
    "Good morning, I am calling to follow up on our appointment scheduled for Tuesday afternoon.",
    "Let me transfer you to our senior verification specialist who will assist you with the transaction.",
    "Please hold on for just a moment while I confirm the transaction status with our clearing house.",
    "We appreciate your business and hope you have a great rest of your day. Thank you for banking with us.",
    "Your confirmation number is delta seven nine four two. Keep this number for your records."
]

ALL_SCRIPTS = SCAM_SCRIPTS + CONVERSATIONAL_SCRIPTS
TLDS = ['com', 'co.uk', 'ca', 'com.au', 'co.in']


def apply_telephony_filter(y, sr=16000):
    """Simulate telephony bandpass filter (300 Hz to 3400 Hz) + mild lossy compression."""
    sos = scipy.signal.butter(4, [300, 3400], btype='bandpass', fs=sr, output='sos')
    filtered = scipy.signal.sosfilt(sos, y)
    noise = np.random.randn(len(filtered)) * 0.005
    return (filtered + noise).astype(np.float32)


def generate_modern_ai_samples(target_count=100):
    print(f"Generating {target_count} modern neural TTS voice samples...")
    current_neural = len([f for f in os.listdir(AI_DIR) if f.startswith("ai_neural_")])
    if current_neural >= target_count:
        print(f"Already have {current_neural} modern neural TTS samples.")
        return

    count = current_neural
    idx = 0
    while count < target_count:
        script = ALL_SCRIPTS[idx % len(ALL_SCRIPTS)]
        tld = TLDS[idx % len(TLDS)]
        is_telephony = (idx % 2 == 1)

        try:
            tts = gTTS(text=script, lang='en', tld=tld, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)

            # Load synthesized audio
            y, sr = sf.read(buf, dtype='float32')
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            if is_telephony:
                y = apply_telephony_filter(y, sr=sr)

            out_name = f"ai_neural_{count:04d}_{tld}_{'tel' if is_telephony else 'clean'}.wav"
            out_path = os.path.join(AI_DIR, out_name)
            sf.write(out_path, y, sr)
            count += 1
            if count % 10 == 0:
                print(f"  Generated {count}/{target_count} samples...")
            time.sleep(0.3)
        except Exception as e:
            print(f"Error generating sample {idx}: {e}")
            time.sleep(1.0)
        idx += 1

    print(f"Done! Successfully generated {count} modern neural TTS samples in {AI_DIR}.")


if __name__ == "__main__":
    generate_modern_ai_samples(100)
