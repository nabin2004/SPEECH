# import torch
# import nemo.collections.asr as nemo_asr
# import contextlib
# import gc

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# # Clear up memory
# torch.cuda.empty_cache()
# gc.collect()
# model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("ai4bharat/indicconformer_stt_ne_hybrid_rnnt_large")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # device = 'cpu'  # You can transcribe even longer samples on the CPU, though it will take much longer !
# model = model.to(device)
# model.lang_id = 1

# concat_audio_path = "./test.wav"

# # Helper for torch amp autocast
# if torch.cuda.is_available():
#     autocast = torch.cuda.amp.autocast
# else:
#     @contextlib.contextmanager
#     def autocast():
#         print("AMP was not available, using FP32!")
#         yield
        
# with autocast():
#     print(model.transcribe([concat_audio_path], batch_size=1)[0])

# # Clear up memory
# torch.cuda.empty_cache()
# gc.collect()