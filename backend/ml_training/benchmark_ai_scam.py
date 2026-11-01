from gtts import gTTS
import io
import os
import sys

sys.path.append(os.path.abspath("backend"))
from app.ml.voice_classifier_service import classify_voice_audio

scam_text = "This is Officer John Davis from the Federal Bureau of Investigation. We have detected suspicious fraudulent transactions under your name. To avoid immediate suspension of your account and legal prosecution, confirm your identity and wire verification funds now."

print("Synthesizing AI scam voice call using neural TTS...")
tts = gTTS(text=scam_text, lang='en', tld='com')
buf = io.BytesIO()
tts.write_to_fp(buf)
ai_bytes = buf.getvalue()

print(f"Generated {len(ai_bytes)} bytes of AI speech audio.")

res = classify_voice_audio(audio_bytes=ai_bytes)
print("\n=== Classification Result ===")
print("Voice Type:", res["voice_type"])
print("Is AI Generated:", res["is_ai_generated"])
print("AI %:", res["ai_voice_percentage"])
print("Robocall %:", res["pre_recorded_percentage"])
print("Human %:", res["human_authenticity_percentage"])
print("Verdict:", res["voice_verdict"])
print("Confidence:", res["confidence"])
print("Markers:", res["acoustic_markers"])
print("Metrics:", res.get("metrics"))
if "segment_predictions" in res:
    print("Segment Predictions:")
    for sp in res["segment_predictions"][:3]:
        print(" ", sp)
