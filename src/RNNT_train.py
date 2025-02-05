import torch
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules import rnnt 
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from omegaconf import OmegaConf
import librosa
import json
import pandas as pd

model_path = "../models/ne.nemo"
asr_model = EncDecRNNTBPEModel.restore_from(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print("Model loaded successfully!")

dataset_path = "../input/cleaned_csv_updated.csv"
df = pd.read_csv(dataset_path)

def preprocess(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    return torch.tensor(audio, dtype=torch.float32)

train_data = []
for _, row in df.iterrows():
    file_path = row['file_name']
    transcript = row['Name']
    input_values = preprocess(file_path)
    train_data.append({"audio": input_values, "text": transcript})

manifest_path = "../input/train_manifest.json"
# with open(manifest_path, "w") as f:
#     for sample in train_data:
#         json.dump({"audio_filepath": sample["audio"], "text": sample["text"], "duration": len(sample["audio"]) / 16000}, f)
#         f.write("\n")

cfg = asr_model.cfg
cfg.train_ds.manifest_filepath = manifest_path
cfg.train_ds.batch_size = 8
cfg.optim.lr = 1e-4

trainer = nemo.utils.exp_manager.ExpManager(asr_model, OmegaConf.create({"exp_dir": "./results"}))

asr_model.setup_training_data(train_data_config=cfg.train_ds)
asr_model.setup_validation_data(val_data_config=cfg.train_ds)
asr_model.train()

asr_model.save_to("../models/nepali-stt-model-rnnt.nemo")
print("Fine-tuned RNNT model saved successfully!")

beam_search = rnnt(asr_model.decoder, asr_model.joint, beam_size=5)
def transcribe_with_beam_search(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(asr_model.device)
    hypotheses = beam_search(audio_tensor)
    best_hyp = max(hypotheses, key=lambda x: x.score)
    return best_hyp.text

sample_audio = "path/to/sample.wav"
transcription = transcribe_with_beam_search(sample_audio)
print("Transcription:", transcription)
