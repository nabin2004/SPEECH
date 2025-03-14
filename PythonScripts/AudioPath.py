import os
import pandas as pd
from natsort import natsorted

def get_sorted_wav_filenames(directory):
    """
    Extracts and sorts names of .wav files from a given directory.

    Args:
        directory (str): Path to the directory containing .wav files.

    Returns:
        list: Sorted list of .wav file names.
    """
    return [f for f in sorted(os.listdir(directory)) if f.endswith(".wav")]

def update_csv_with_paths(input_csv, output_csv, wav_directory):
    """
    Update CSV file with new WAV file names and relative paths while preserving order.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
        wav_directory (str): Directory containing .wav files.
    """
    df = pd.read_csv(input_csv)
    
    # Get sorted list of .wav files
    wav_files = natsorted(get_sorted_wav_filenames(wav_directory))
    
    # Ensure number of audio files matches number of rows
    if len(df) != len(wav_files):
        raise ValueError(f"Mismatch: CSV has {len(df)} rows, but {len(wav_files)} WAV files found!")

    # Assign sorted .wav file names to the 'file_name' column
    df["file_name"] = [os.path.join(wav_directory, wav_files[i]) for i in range(len(df))]

    # Save updated CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file updated successfully: {output_csv}")

# **Run the function**
input_csv = "../input/cleaned_unique_csv.csv"
output_csv = "../input/cleaned_csv_updated.csv"
wav_directory = "../input/wav_format/"  

# Update CSV with new file paths (Preserves Order)
update_csv_with_paths(input_csv, output_csv, wav_directory)