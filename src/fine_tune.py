import os
import json
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from natsort import natsorted  # Import natsorted
import torchaudio.transforms as T  # Import torchaudio transforms for feature extraction

# Define the Custom Dataset Class
class CustomAudioDataset(Dataset):
    def __init__(self, manifest_file, audio_dir, transform=None, target_transform=None, max_length=None):
        # Load the manifest JSON file
        with open(manifest_file, 'r') as f:
            self.audio_labels = json.load(f)
        
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_transform = target_transform
        self.max_length = max_length

        # Sort the audio filenames using natsorted (based on 'audio_filepath')
        self.audio_labels = natsorted(self.audio_labels, key=lambda x: x['audio_filepath'])

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        # Retrieve the audio file path and transcription from the manifest
        audio_info = self.audio_labels[idx]
        audio_path = audio_info['audio_filepath']
        
        # Load the audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Apply padding if necessary
        if self.max_length and waveform.size(1) < self.max_length:
            padding_length = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding_length))

        label = os.path.basename(audio_path)  # Extract label from the filename

        # Extract the transcription from the 'text' field in the manifest
        transcription = audio_info['text']

        return waveform, sample_rate, label, transcription


# Load the pre-trained ASR model
model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large")

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move the model to the device
model.freeze()  # Freeze the model for inference mode initially (before fine-tuning)

# Optionally, unfreeze the model if you want to fine-tune
# model.unfreeze()

# Set decoder type to RNNT (already set by default in the pre-trained model)
model.cur_decoder = "rnnt"

# Path to the manifest JSON and audio directory
manifest_file = "../input/train_manifest.json"
audio_dir = "../input/wav_format/"

# Create the dataset and dataloaders
train_dataset = CustomAudioDataset(manifest_file=manifest_file, audio_dir=audio_dir, max_length=160000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Configure training parameters
trainer_cfg = OmegaConf.create({
    "max_steps": 10000,  # Number of training steps
    "lr": 0.0001,  # Learning rate
    "batch_size": 32,  # Batch size for training
    "num_epochs": 10,  # Number of epochs for training
    "save_interval": 1000,  # Model save interval
})

# Define the optimizer and loss function
optimizer_params = {
    "params": model.parameters(),
    "lr": trainer_cfg.lr,
}
optimizer = torch.optim.AdamW(**optimizer_params)

# Loss function for RNNT (already defined in the model)
loss_fn = model.loss

# Define MelSpectrogram transformation
mel_transform = T.MelSpectrogram()

# Fine-tune the model
model.train()
for epoch in range(trainer_cfg.num_epochs):
    for batch_idx, (audio_batch, sample_rate_batch, label_batch, transcription_batch) in enumerate(train_loader):
        # Transfer data to the same device as the model
        audio_batch = audio_batch.squeeze(1)  
        audio_batch = audio_batch.to(device)
        
        # Apply the MelSpectrogram transformation to each audio in the batch
        spectrogram_batch = [mel_transform(waveform.to(device)) for waveform in audio_batch]

        # Convert the spectrogram list to a tensor
        spectrogram_batch = torch.stack(spectrogram_batch, dim=0)

        # Calculate the length of each audio sample and transcription length
        # Ensure audio_batch is a 3D tensor. If not, adjust the logic accordingly.
        audio_lengths = torch.tensor([waveform.size(-1) for waveform in audio_batch]).to(device)
        transcription_lengths = torch.tensor([len(t) for t in transcription_batch]).to(device)

        # Zero gradients, forward pass, and compute loss
        optimizer.zero_grad()
        
        # Pass the processed spectrogram to the model
        loss = model(input_signal=audio_batch, 
                     input_signal_length=audio_lengths, 
                     processed_signal=spectrogram_batch, 
                     processed_signal_length=transcription_lengths)

        loss.backward()
        optimizer.step()

        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item()}")

    # Save the model after each epoch (optional)
    model.save_to(f"fine_tuned_model_epoch_{epoch}.nemo")

# Save the fine-tuned model after training is complete
model.save_to("fine_tuned_model.nemo")