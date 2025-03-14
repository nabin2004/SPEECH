import os
import csv
from google.cloud import texttospeech

def text_to_speech_google(text, output_file):
    # Set the environment variable for Google Cloud credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../key.json"
    
    # Initialize the Google Cloud Text-to-Speech client
    client = texttospeech.TextToSpeechClient()

    # Set up the synthesis input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Select the voice parameters (e.g., language, gender)
    voice = texttospeech.VoiceSelectionParams(
        language_code="hi-IN",  # Language code 
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL  # Gender selection (NEUTRAL)
    )

    # Set up the audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3  # Output in MP3 format
    )

    # Request speech synthesis
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Write the audio content to an MP3 file
    with open(output_file, "wb") as out:
        out.write(response.audio_content)
        print(f"Generated speech for text: '{text}' and saved to {output_file}")

# Main function to process each line of the CSV and generate speech files
def generate_text_to_speech_dataset(input_csv, output_dir):
    with open(input_csv, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Skip header if there's one
        next(csvreader, None)

        # Process each row
        row_index = 1  # Starting row index (adjust as needed based on data)
        for row in csvreader:
            # Assuming each row in the CSV has a single column with text data
            text = row[0]  # Adjust the column index if necessary
            
            # Clean the text to generate a valid filename
            audio_filename = f"{row_index}_{text[:30].strip().replace(' ', '_').replace('/', '_')}.mp3"
            audio_file_path = os.path.join(output_dir, audio_filename)
            
            # Generate speech from the text
            print(f"Generating speech for: {text}")
            text_to_speech_google(text, audio_file_path)
            
            row_index += 1  # Increment the row index for the next iteration

# Example usage:
input_csv = '../input/cleaned_unique_csv.csv'  # Path to the CSV file
output_audio_directory = '../input/voice/'  # Output directory for audio files

# Ensure the output directory exists
os.makedirs(output_audio_directory, exist_ok=True)

generate_text_to_speech_dataset(input_csv, output_audio_directory)
