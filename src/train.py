import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr

model_path = "ne.nemo"
lang_id = "ne"

file_path = "/content/drive/MyDrive/ne_np_female/ne_np_female"
wavs_files = f"{file_path}/wavs/*.wav"
labels = f"{file_path}/line_index.tsv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
model.eval() 
model = model.to(device)

# download and load audio
MEDIA_URL = "/content/drive/MyDrive/ne_np_female/ne_np_female/wavs/nep_0258_0119737288.wav"

#Download
# !ffmpeg -i "$MEDIA_URL" -ac 1 -ar 16000 sample_audio_infer_ready.wav -y

model.cur_decoder = "ctc"
ctc_text = model.transcribe(['sample_audio_infer_ready.wav'], batch_size=1,logprobs=False, language_id=lang_id)[0]
print(ctc_text)