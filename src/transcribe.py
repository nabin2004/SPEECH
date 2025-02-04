import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
 
def transcribe_audio(model_path: str, audio_file: str, lang_id: str = None):
    """
    Transcribes an audio file using a pretrained NeMo ASR model.
    
    Args:
        model_path (str): Path to the pretrained model.
        audio_file (str): Path to the audio file to transcribe.
        lang_id (str, optional): Language ID for multilingual models.

    Returns:
        str: The transcribed text.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load the model
        model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
        model.eval()  # Set to inference mode
        model = model.to(device)  # Move model to appropriate device
        
        # Perform transcription
        transcription = model.transcribe([audio_file], batch_size=1, logprobs=False, language_id=lang_id)[0]
        
        return transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

if __name__ == "__main__":
    model_path = "your_model_path.nemo" 
    audio_file = "sample_audio_infer_ready.wav"
    lang_id = "en"  

    # Transcribe audio
    transcription = transcribe_audio(model_path, audio_file, lang_id)
    
    if transcription:
        print("Transcription:", transcription)
