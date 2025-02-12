import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import pytorch_lightning as pl

params = OmegaConf.load("./contextnet_rnnt.yaml")

print(OmegaConf.to_yaml(params))

params.model.tokenizer.dir = "../tokenizer/"
params.model.tokenizer.type = "bpe"

trainer1 = pl.Trainer(devices=1, max_epochs=10)

asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=params.model, trainer=trainer1)

trainer1.fit(asr_model)

asr_model.save_to("asr_model.nemo")

print(params.model.optim)