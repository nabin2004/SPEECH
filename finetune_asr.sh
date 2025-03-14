#!/bin/bash

# Run Speech to Text fine-tuning using NeMo
python3 examples/asr/speech_to_text_finetune.py \
model.tokenizer.update_tokenizer=False \
trainer.devices=1 \
trainer.accelerator="cpu" \
trainer.max_epochs=10 
# ++model.train_ds.use_semi_sorted_batching=true
# ++model.train_ds.randomization_factor=0.1
# trainer.sync_batchnorm= False \
# +init_from_nemo_model="models/ne.nemo"
