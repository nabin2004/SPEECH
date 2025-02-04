import librosa
import soundfile as sf
import os

def convert_audios_in_folder(input_dir, output_dir, target_sr=16000):
    """
    Convert all audio files in input directory to WAV format in output directory.
    
    Args:
        input_dir (str): Path to input directory containing audio files
        output_dir (str): Path to output directory for WAV files
        target_sr (int): Target sample rate (default: 16000 Hz)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files in input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        
        if not os.path.isfile(input_path):
            continue
        
        try:
            audio, sr = librosa.load(input_path, sr=target_sr)
            
            output_filename = os.path.splitext(filename)[0] + '.wav'
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as WAV file with PCM_16 encoding
            sf.write(output_path, audio, target_sr, subtype='PCM_16')
            print(f"Converted: {filename} -> {output_filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

convert_audios_in_folder("../input/voice", "../input/wav_format")