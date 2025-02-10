import streamlit as st
import requests
import numpy as np
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import soundfile as sf

# WebRTC Configuration
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def save_audio_file(audio_frames):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        audio_data = np.concatenate([frame.to_ndarray()[:, 0] for frame in audio_frames])
        sf.write(temp_audio.name, audio_data, 44100, format='WAV')
        return temp_audio.name

# Streamlit UI
def main():
    st.title("Automatic Speech Recognition (ASR) App")
    st.write("Record or upload an audio file and get its transcription.")
    
    # WebRTC-based voice recording
    webrtc_ctx = webrtc_streamer(
        key="speech-recorder",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": False, "audio": True},
    )
    
    if st.button("Transcribe Recorded Audio") and webrtc_ctx.audio_receiver:
        audio_frames = webrtc_ctx.audio_receiver.get_frames()
        if audio_frames:
            temp_audio_path = save_audio_file(audio_frames)
            
            with open(temp_audio_path, "rb") as audio_file:
                files = {"file": audio_file.read()}
                response = requests.post("http://127.0.0.1:8000/transcribe/", files=files)
                
                if response.status_code == 200:
                    transcription = response.json().get("transcription", "No transcription available")
                    st.success("Transcription:")
                    st.write(transcription)
                    print("Transcription:", transcription)
                else:
                    st.error("Error in transcription: " + response.json().get("error", "Unknown error"))
    
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "flac"])  
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("Transcribe Upload"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8000/transcribe/", files=files)
            
            if response.status_code == 200:
                transcription = response.json().get("transcription", "No transcription available")
                st.success("Transcription:")
                st.write(transcription)
                print("Transcription:", transcription)
            else:
                st.error("Error in transcription: " + response.json().get("error", "Unknown error"))

if __name__ == "__main__":
    main()
