import json
import librosa

def calculate_lengths(manifest_file):
    # Open and load the manifest JSON file
    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)

    updated_manifest = []

    for entry in manifest_data:
        audio_filepath = entry["audio_filepath"]
        duration = entry["duration"]
        sample_rate = entry["sample_rate"]
        transcript = entry["text"]

        # Calculate audio_length (number of samples)
        audio_length = int(duration * sample_rate)  

        # Calculate transcript_length (number of characters in transcript including special tokens)
        transcript_length = len(transcript)

        # Set signal_len as the same as audio_length
        signal_len = audio_length

        # Add the calculated values to the entry
        entry["audio_length"] = audio_length
        entry["transcript_length"] = transcript_length
        entry["signal_len"] = signal_len

        updated_manifest.append(entry)

    # Write the updated manifest with the new parameters
    with open('updated_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(updated_manifest, f, ensure_ascii=False, indent=4)

# Path to the original manifest file
manifest_file = "../input/train_manifest.json"

# Run the function to calculate lengths and update the manifest
calculate_lengths(manifest_file)

print("Manifest has been updated with audio_length, transcript_length, and signal_len.")
