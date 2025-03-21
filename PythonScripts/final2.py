import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

# asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large")

# asr_model.setup_training_data(train_data_config={"manifest_filepath": "../input/train_manifest.json", "batch_size": 16, "shuffle": True, 'sample_rate':16000})
# asr_model.setup_validation_data(val_data_config={"manifest_filepath": "../input/validation_data/valid_manifest.json", "batch_size": 16, 'sample_rate':16000})

# trainer1 = pl.Trainer(devices=1, max_epochs=10)

# trainer1.fit(asr_model)

# asr_model.save_to("fine_tuned_indicconformer.nemo")

model_path = "../models/ne.nemo"
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
model.eval()

lang_id = "ne"
model.cur_decoder = "rnnt"
ctc_text = model.transcribe(['test.wav'], batch_size=1,logprobs=False, language_id=lang_id)[0]
print(ctc_text)