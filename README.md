# Speech-to-Text Fine-Tuning

## Dataset Format
The dataset is structured in JSON format, where each entry represents an audio sample with metadata. Below is an example:

```json
{"audio_filepath": "./input/wav_format/106_मेडिकल_चोक.wav", "text": " मेडिकल चोक ", "duration": 1.296,  "language_ids": "1", "sample_ids": "0001", "lang": "ne"}
{"audio_filepath": "./input/wav_format/107_एकता_चोक.wav", "text": "  एकता चोक ", "duration": 1.152,  "language_ids": "1", "sample_ids": "0002", "lang": "ne"}
{"audio_filepath": "./input/wav_format/108_चित्रपुर.wav", "text": " चित्रपुर ", "duration": 0.936,  "language_ids": "1", "sample_ids": "0003", "lang": "ne"}
{"audio_filepath": "./input/wav_format/109_Kumaripati.wav", "text": " Kumaripati ", "duration": 1.08,  "language_ids": "1", "sample_ids": "0004", "lang": "ne"}
{"audio_filepath": "./input/wav_format/110_Pulchowk.wav", "text": " Pulchowk ", "duration": 0.96,  "language_ids": "1", "sample_ids": "0005", "lang": "ne"}
```

### Fields Explanation
- `audio_filepath`: Path to the audio file.
- `text`: Transcription of the audio.
- `duration`: Length of the audio in seconds.
- `language_ids`: Language identifier.
- `sample_ids`: Unique identifier for the sample.
- `lang`: Language code (e.g., "ne" for Nepali).

## Configuration File
The fine-tuning process is managed using a configuration file:

```
examples/asr/conf/asr_finetune/speech_to_text_finetune.yaml
```

This file contains hyperparameters and training settings for fine-tuning the ASR model.

## Running the Fine-Tuning Script
To start the fine-tuning process, execute the following bash script:

```bash
./finetune_asr.sh
```

## Requirements
The required dependencies are listed in the `requirements/` directory. To install them, you can use the provided script:

```bash
bash reinstall.sh
```

