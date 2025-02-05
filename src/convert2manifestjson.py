import pandas as pd
import json
import librosa

# Load dataset CSV
df = pd.read_csv("../input/cleaned_csv_updated.csv")

# Generate manifest file
manifest_path = "../input/train_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:  # Ensure UTF-8 encoding
    for _, row in df.iterrows():
        entry = {
            "audio_filepath": row["file_name"],  # Path to audio file
            "text": row["Name"],  # Transcription
            "duration": librosa.get_duration(path=row["file_name"])  # Get duration
        }
        json.dump(entry, f, ensure_ascii=False)  # Ensure readable text
        f.write("\n")  # New line for JSONL format

print("✅ Manifest file saved successfully:", manifest_path)
