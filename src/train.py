from transformers import Wav2Vec2Processor, AutoModelForSeq2SeqLM # for RNNT 
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import librosa

# Load IndicConformer RNNT Model & Processor
model_name = "../models/ne.nemo"  
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load Dataset
# dataset = load_dataset("csv", data_files={"train": "data/train/train.csv" , "validation": "data/valid/valid.csv"})  
dataset = load_dataset("csv", data_files={"train": "../input/cleaned_csv.csv"})  

# Preprocessing Function
def preprocess(batch):
    audio_path = batch["file_name"]
    
    # Load audio file
    audio, _ = librosa.load(audio_path, sr=16000)
    
    # Convert to model input format
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    
    # Tokenize transcription text
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch

# Apply Preprocessing
dataset = dataset.map(preprocess)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=1e-4,
    num_train_epochs=10,
    save_total_limit=2,  # Keep only last 2 checkpoints
)

# Trainer for RNNT Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

# Start Training
trainer.train()

# Save Fine-tuned Model
model.save_pretrained("../models/nepali-stt-model-rnnt")
processor.save_pretrained("../models/nepali-stt-model-rnnt")
