import torch
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel  
from nemo.collections.asr.parts.utils import ctc_utils
from omegaconf import OmegaConf
import soundfile as sf
import librosa
import json
import pandas as pd

# Load the CTC-based model
model_path = "../models/ne_ctc_model.nemo"
asr_model = EncDecCTCModel.restore_from(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print("Model loaded successfully!")

# Load dataset (CSV)
dataset_path = "../input/cleaned_csv.csv"
df = pd.read_csv(dataset_path)

# Preprocessing the audio files
def preprocess(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return torch.tensor(audio, dtype=torch.float32)

# Create training data
train_data = []
for _, row in df.iterrows():
    file_path = row['file_name']
    transcript = row['text']
    input_values = preprocess(file_path)
    train_data.append({"audio": input_values, "text": transcript})

# Save the manifest file
manifest_path = "../input/train_manifest.json"
with open(manifest_path, "w") as f:
    for sample in train_data:
        json.dump({"audio_filepath": sample["audio"], "text": sample["text"], "duration": len(sample["audio"]) / 16000}, f)
        f.write("\n")

# Set up training config
cfg = asr_model.cfg
cfg.train_ds.manifest_filepath = manifest_path
cfg.train_ds.batch_size = 8
cfg.optim.lr = 1e-4

trainer = nemo.utils.exp_manager.ExpManager(asr_model, OmegaConf.create({"exp_dir": "./results"}))

# Set up the training and validation data
asr_model.setup_training_data(train_data_config=cfg.train_ds)
asr_model.setup_validation_data(val_data_config=cfg.train_ds)

# Train the model
asr_model.train()

# Save the fine-tuned CTC model
asr_model.save_to("../models/nepali-ctc-model.nemo")
print("Fine-tuned CTC model saved successfully!")

# For CTC decoding, we can use a greedy decoder
def transcribe_with_ctc(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(asr_model.device)
    logits = asr_model.forward(audio_tensor)  

    transcription = ctc_utils.ctc_greedy_decoder(logits)
    return transcription[0] 


sample_audio = "path/to/sample.wav"
transcription = transcribe_with_ctc(sample_audio)
print("Transcription:", transcription)
