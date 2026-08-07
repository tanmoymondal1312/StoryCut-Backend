from faster_whisper import WhisperModel, format_timestamp
from google import genai

import json
import sys
from config import GEMINI_API_KEY

# --------------------------
# Whisper
# --------------------------

try:
    model = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8"
    )

    segments, info = model.transcribe(
        "video.mp4",
        beam_size=5,
        task="transcribe",
        language="bn"
    )

    print("Detected Language:", info.language)

    srt = []

    for i, seg in enumerate(segments, 1):

        start = format_timestamp(
            seg.start,
            always_include_hours=True,
            decimal_marker=","
        )

        end = format_timestamp(
            seg.end,
            always_include_hours=True,
            decimal_marker=","
        )

        srt.append(
            f"{i}\n{start} --> {end}\n{seg.text.strip()}\n"
        )

    with open("video.srt", "w", encoding="utf-8") as f:
        f.write("\n".join(srt))

    print("SRT Generated")

except Exception as e:
    print(f"Whisper Error: {e}")
    sys.exit(1)

# --------------------------
# Read Files
# --------------------------

try:
    with open("video.srt", "r", encoding="utf-8") as f:
        subtitle = f.read()

    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

except FileNotFoundError as e:
    print(f"File Error: {e}")
    sys.exit(1)

# --------------------------
# Gemini
# --------------------------

try:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""
{prompt}

Subtitle:

{subtitle}
"""
    )

    text = response.text

    text = text.replace("```json", "")
    text = text.replace("```", "")

    data = json.loads(text)

    with open("highlight.json", "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("highlight.json Generated")

except json.JSONDecodeError as e:
    print(f"JSON Parse Error: {e}")
    print(f"Gemini Response:\n{text}")
    sys.exit(1)

except Exception as e:
    print(f"Gemini Error: {e}")
    sys.exit(1)
