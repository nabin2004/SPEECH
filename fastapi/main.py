from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import nemo.collections.asr as nemo_asr
import torchaudio
import io
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the fine-tuned ASR model
model_path = "../models/ne.nemo"
model = nemo_asr.models.ASRModel.restore_from(model_path)

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set to evaluation mode

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """Endpoint to transcribe an uploaded audio file."""
    try:
        # Read the uploaded file
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        # Load the audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_buffer)
        waveform = waveform.to(device)

        # Transcribe the audio using RNNT model
        transcription = model.transcribe([waveform])
        print("Real-time Transcription:", transcription[0])

        return {"filename": file.filename, "transcription": transcription[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    """Basic endpoint to check if the API is running."""
    return {"message": "ASR Model API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
