import os
import json
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from natsort import natsorted  
import torchaudio.transforms as T  

class CustomAudioDataset(Dataset):
    def __init__(self, manifest_file, audio_dir, transform=None, target_transform=None, max_length=None):
        with open(manifest_file, 'r') as f:
            self.audio_labels = json.load(f)
        
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
        self.max_length = max_length

        self.audio_labels = natsorted(self.audio_labels, key=lambda x: x['audio_filepath'])

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        audio_info = self.audio_labels[idx]
        audio_path = audio_info['audio_filepath']
        
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.max_length and waveform.size(1) < self.max_length:
            padding_length = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding_length))

        label = os.path.basename(audio_path)

        transcription = audio_info['text']

        return waveform, sample_rate, label, transcription

model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.freeze()

model.cur_decoder = "rnnt"

manifest_file = "../input/train_manifest.json"
audio_dir = "../input/wav_format/"

train_dataset = CustomAudioDataset(manifest_file=manifest_file, audio_dir=audio_dir, max_length=160000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

trainer_cfg = OmegaConf.create({
    "max_steps": 10000,
    "lr": 0.0001,
    "batch_size": 32,
    "num_epochs": 10,
    "save_interval": 1000,
})

optimizer_params = {
    "params": model.parameters(),
    "lr": trainer_cfg.lr,
}
optimizer = torch.optim.AdamW(**optimizer_params)

loss_fn = model.loss

mel_transform = T.MelSpectrogram()

model.train()
for epoch in range(trainer_cfg.num_epochs):
    for batch_idx, (audio_batch, sample_rate_batch, label_batch, transcription_batch) in enumerate(train_loader):
        audio_batch = audio_batch.squeeze(1)  
        audio_batch = audio_batch.to(device)
        
        spectrogram_batch = [mel_transform(waveform.to(device)) for waveform in audio_batch]

        spectrogram_batch = torch.stack(spectrogram_batch, dim=0)

        audio_lengths = torch.tensor([waveform.size(-1) for waveform in audio_batch]).to(device)
        transcription_lengths = torch.tensor([len(t) for t in transcription_batch]).to(device)

        optimizer.zero_grad()
        
        loss = model(input_signal=audio_batch, 
                     input_signal_length=audio_lengths, 
                     processed_signal=spectrogram_batch, 
                     processed_signal_length=transcription_lengths)

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item()}")

    model.save_to(f"fine_tuned_model_epoch_{epoch}.nemo")

model.save_to("fine_tuned_model.nemo")
