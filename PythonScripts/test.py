import nemo.core
import torch
import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

# Load Pretrained Model
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained("ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = asr_model.to(device)

# Unfreezing the model for training
asr_model.unfreeze()

# Define dataset paths
train_manifest = "../input/train_manifest.json"
valid_manifest = "../input/validation_data/valid_manifest.json"


asr_model.setup_training_data(
    train_data_config={
        "manifest_filepath": train_manifest,
        "sample_rate": 16000,
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "signal": "audio_filepath",
        "signal_len": "audio_length",
        "transcript": "text",
        "transcript_len": "transcript_length",
        "language_ids": "ne",
    }
)

asr_model.setup_validation_data(
    val_data_config={
        "manifest_filepath": valid_manifest,
        "sample_rate": 16000,
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 2,
        "signal": "audio_filepath",
        "signal_len": "audio_length",
        "transcript": "text",
        "transcript_len": "transcript_length",
        "language_ids": "ne",
    }
)


trainer = nemo.core.pytorch_lightning.Trainer(
    max_epochs=10, # Adjust epochs based on dataset size
    accelerator="gpu" if torch.cuda.is_available() else "cpu"
)

trainer.fit(asr_model)


asr_model.save_to("fine_tuned_model.nemo")