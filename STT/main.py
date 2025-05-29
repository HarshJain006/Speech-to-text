import streamlit as st
import numpy as np
import time
import torch
from transformers import pipeline
import librosa
import soundfile as sf
import sounddevice as sd
import io
import uuid
import re
import os
import tempfile

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #F5F6F5;
            color: #333;
        }
        .stApp {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(90deg, #4169E1, #1E3A8A);
            color: #FFFFFF;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 32px;
        }
        .stTabs [role="tab"] {
            background-color: #FFFFFF;
            color: #4169E1;
            border: 2px solid #4169E1;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #4169E1;
            color: #FFFFFF;
        }
        .stTabs [role="tab"]:hover {
            background-color: #C0C0C0;
            color: #FFFFFF;
        }
        .stButton > button {
            background-color: #C0C0C0;
            color: #333;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            background-color: #A9A9A9;
        }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            border: 1px solid #4169E1;
            border-radius: 8px;
            padding: 10px;
            background-color: #FFFFFF;
            color: #333;
        }
        .stFileUploader > div > div {
            border: 1px solid #4169E1;
            border-radius: 8px;
            background-color: #FFFFFF;
        }
        .status-box {
            background-color: #F0F0F0;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            border: 1px solid #4169E1;
        }
        .info-text {
            color: #4169E1;
            font-size: 14px;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'english_hindi_transcriber' not in st.session_state:
    st.session_state.english_hindi_transcriber = None
    st.session_state.english_hindi_loaded = False
    st.session_state.hindi_only_transcriber = None
    st.session_state.hindi_only_loaded = False
    st.session_state.transcription_en_hi = ""
    st.session_state.duration_en_hi = "0.0 seconds"
    st.session_state.proc_time_en_hi = "0.0 seconds"
    st.session_state.transcription_hi = ""
    st.session_state.duration_hi = "0.0 seconds"
    st.session_state.proc_time_hi = "0.0 seconds"
    st.session_state.status_en_hi = "Speech2Text is OFF"
    st.session_state.status_hi = "Speech2Text is OFF"
    st.session_state.working_devices = []
    st.session_state.recorded_audio = None

# Microphone device functions
def list_input_devices():
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append((i, dev['name']))
        return input_devices
    except Exception as e:
        return []

def test_device(device_index, duration=1, fs=44100):
    try:
        sd.default.device = (device_index, None)
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        if np.any(np.abs(recording) > 0.001):
            return True
        return False
    except Exception:
        return False

# Model loading/unloading functions
def load_english_hindi(model_size="tiny"):
    if not st.session_state.english_hindi_loaded:
        try:
            st.session_state.english_hindi_transcriber = pipeline(
                "automatic-speech-recognition",
                model=f"openai/whisper-{model_size}",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            st.session_state.english_hindi_loaded = True
            return f"Speech2Text (English & Hindi) {model_size} loaded successfully"
        except Exception as e:
            return f"Error loading Speech2Text (English & Hindi): {str(e)}"
    return "Speech2Text (English & Hindi) already loaded"

def load_hindi_only():
    if not st.session_state.hindi_only_loaded:
        try:
            st.session_state.hindi_only_transcriber = pipeline(
                "automatic-speech-recognition",
                model="AI4Bharat/indicwav2vec-hindi",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            st.session_state.hindi_only_loaded = True
            return "Speech2Text (Hindi Only) loaded successfully"
        except Exception as e:
            return f"Error loading Speech2Text (Hindi Only): {str(e)}"
    return "Speech2Text (Hindi Only) already loaded"

def unload_english_hindi():
    st.session_state.english_hindi_transcriber = None
    st.session_state.english_hindi_loaded = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Speech2Text (English & Hindi) unloaded"

def unload_hindi_only():
    st.session_state.hindi_only_transcriber = None
    st.session_state.hindi_only_loaded = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Speech2Text (Hindi Only) unloaded"

# Audio processing function
def process_audio(audio, sr=16000):
    if audio is None:
        return None, "0.0 seconds", "No audio detected"
    try:
        if isinstance(audio, tuple):  # From mic recording (sounddevice)
            input_sr, y = audio
            if len(y) == 0:
                return None, "0.0 seconds", "Empty audio data from microphone"
            y = y.astype(np.float32)
            if len(y.shape) == 2:
                y = np.mean(y, axis=1)
            # Resample if necessary
            if input_sr != sr:
                y = librosa.resample(y, orig_sr=input_sr, target_sr=sr)
        else:  # From file upload
            # Validate file extension
            if hasattr(audio, 'name'):
                ext = os.path.splitext(audio.name)[1].lower()
                if ext not in ['.wav', '.mp3']:
                    return None, "0.0 seconds", f"Unsupported file format: {ext}. Please upload WAV or MP3."
            # Save uploaded file temporarily
            temp_file = f"temp_audio_{uuid.uuid4()}{ext if hasattr(audio, 'name') else '.wav'}"
            with open(temp_file, "wb") as f:
                f.write(audio.read())
            try:
                y, input_sr = sf.read(temp_file)
                if len(y) == 0:
                    return None, "0.0 seconds", "No audio data found in file"
                y = y.astype(np.float32)
                if len(y.shape) == 2:
                    y = np.mean(y, axis=1)
                if input_sr != sr:
                    y = librosa.resample(y, orig_sr=input_sr, target_sr=sr)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        audio_duration = len(y) / sr
        audio_duration_str = f"{audio_duration:.2f} seconds"
        if np.max(np.abs(y)) > 0:
            y /= np.max(np.abs(y))  # Normalize
        return (sr, y), audio_duration_str, None
    except Exception as e:
        return None, "0.0 seconds", f"Error processing audio: {str(e)}"

# Transcription functions
def transcribe_english_hindi(audio, language):
    if not st.session_state.english_hindi_loaded:
        return (
            "Please turn on Speech2Text (English & Hindi) first",
            "0.0 seconds",
            "No processing performed"
        )
    processed_audio, audio_duration_str, error = process_audio(audio)
    if error:
        return error, audio_duration_str, "No processing performed"
    sr, y = processed_audio
    try:
        start_time = time.time()
        with st.spinner("Transcribing..."):
            lang_code = "en" if language == "english" else "hi"
            result = st.session_state.english_hindi_transcriber(
                {"sampling_rate": sr, "raw": y},
                generate_kwargs={"language": lang_code, "task": "transcribe"},
                return_timestamps=True
            )
            if "chunks" in result and result["chunks"]:
                transcription = "\n".join([chunk["text"].strip() for chunk in result["chunks"] if chunk["text"].strip()])
            else:
                transcription = result.get("text", "").strip()
            processing_time = time.time() - start_time
            processing_time_str = f"{processing_time:.2f} seconds"
        return transcription, audio_duration_str, processing_time_str
    except Exception as e:
        return (
            f"Error in Speech2Text transcription: {str(e)}",
            audio_duration_str,
            "Error during processing"
        )

def transcribe_hindi_only(audio):
    if not st.session_state.hindi_only_loaded:
        return (
            "Please turn on Speech2Text (Hindi Only) first",
            "0.0 seconds",
            "No processing performed"
        )
    processed_audio, audio_duration_str, error = process_audio(audio)
    if error:
        return error, audio_duration_str, "No processing performed"
    sr, y = processed_audio
    try:
        start_time = time.time()
        with st.spinner("Transcribing..."):
            result = st.session_state.hindi_only_transcriber(
                {"sampling_rate": sr, "raw": y}
            )
            raw_text = result.get("text", "").strip()
            segments = re.split(r'[।,.!?]\s*', raw_text)
            transcription = "\n".join([segment.strip() for segment in segments if segment.strip()])
            processing_time = time.time() - start_time
            processing_time_str = f"{processing_time:.2f} seconds"
        return transcription, audio_duration_str, processing_time_str
    except Exception as e:
        return (
            f"Error in Speech2Text transcription: {str(e)}",
            audio_duration_str,
            "Error during processing"
        )

# Main app
def main():
    st.markdown('<div class="header"><h1>Speech Recognition System</h1></div>', unsafe_allow_html=True)
    st.markdown("### Choose between Speech2Text models", unsafe_allow_html=True)

    # Scan microphone devices
    if not st.session_state.working_devices:
        devices = list_input_devices()
        for idx, name in devices:
            if test_device(idx):
                st.session_state.working_devices.append((idx, name))

    # Tabs
    tab1, tab2 = st.tabs(["Speech2Text (English & Hindi)", "Speech2Text (Hindi Only)"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            all_models = ["tiny", "base", "small", "medium", "large-v2"]
            selectable_models = [m for m in all_models if m != "large-v2"]
            model_size = st.selectbox(
                "Model Size",
                selectable_models,
                index=0,
                format_func=lambda x: f"{x} (disabled)" if x == "large-v2" else x,
                help="Tiny or small recommended for Indian accents. Large-v2 is disabled due to resource constraints.",
                key="model_size_en_hi"
            )
            power_button_en_hi = st.button(
                "Turn ON Speech2Text" if not st.session_state.english_hindi_loaded else "Turn OFF Speech2Text",
                key="power_button_en_hi"
            )
            if power_button_en_hi:
                status = load_english_hindi(model_size) if not st.session_state.english_hindi_loaded else unload_english_hindi()
                st.session_state.status_en_hi = f"Speech2Text is {'ON' if st.session_state.english_hindi_loaded else 'OFF'}: {status}"
            st.markdown(f'<div class="status-box">{st.session_state.status_en_hi}</div>', unsafe_allow_html=True)
        with col2:
            language_selection = st.radio(
                "Select Language",
                ["english", "hindi"],
                index=0,
                help="Select language for transcription",
                key="language_selection_en_hi"
            )

        # Microphone recording
        st.markdown("### Record Audio", unsafe_allow_html=True)
        if not st.session_state.working_devices:
            st.error("No working microphones found! Please upload an audio file instead.")
        else:
            device_names = [name for idx, name in st.session_state.working_devices]
            chosen_name = st.selectbox("Choose microphone input device", device_names, key="mic_select_en_hi")
            chosen_idx = None
            for idx, name in st.session_state.working_devices:
                if name == chosen_name:
                    chosen_idx = idx
                    break
            duration = st.number_input(
                "Recording duration (seconds)",
                min_value=1,
                max_value=600,
                value=5,
                step=1,
                key="duration_en_hi"
            )
            if st.button("Record Audio", key=f"record_en_hi_{uuid.uuid4()}"):
                try:
                    sd.default.device = (chosen_idx, None)
                    st.info(f"Recording for {duration} seconds from '{chosen_name}'...")
                    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
                    sd.wait()
                    st.success("Recording finished!")
                    rms = np.sqrt(np.mean(recording**2))
                    st.write(f"Recorded audio RMS amplitude: {rms:.6f}")
                    if rms < 1e-4:
                        st.warning("Warning: Recorded audio seems silent!")
                    st.session_state.recorded_audio = (44100, recording)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        sf.write(tmpfile.name, recording.astype(np.float32), 44100)
                        st.audio(tmpfile.name)
                        os.remove(tmpfile.name)
                except Exception as e:
                    st.error(f"Error recording audio: {str(e)}")

        uploaded_file_en_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_en_hi")
        
        if st.button("Transcribe with Speech2Text", key="transcribe_en_hi"):
            audio = st.session_state.recorded_audio if st.session_state.recorded_audio else uploaded_file_en_hi
            if audio:
                transcription, duration, proc_time = transcribe_english_hindi(audio, language_selection)
                st.session_state.transcription_en_hi = transcription
                st.session_state.duration_en_hi = duration
                st.session_state.proc_time_en_hi = proc_time
            else:
                st.error("No audio input provided. Please record or upload an audio file.")
        
        st.text_area("Transcription", st.session_state.transcription_en_hi, height=150, disabled=True, key="transcription_en_hi")
        col3, col4 = st.columns(2)
        with col3:
            st.text_input("Audio Duration", st.session_state.duration_en_hi, disabled=True, key="duration_en_hi")
        with col4:
            st.text_input("Processing Time", st.session_state.proc_time_en_hi, disabled=True, key="proc_time_en_hi")

    with tab2:
        st.markdown("### This Speech2Text model is specialized for Hindi language only", unsafe_allow_html=True)
        power_button_hi = st.button(
            "Turn ON Speech2Text" if not st.session_state.hindi_only_loaded else "Turn OFF Speech2Text",
            key="power_button_hi"
        )
        if power_button_hi:
            status = load_hindi_only() if not st.session_state.hindi_only_loaded else unload_hindi_only()
            st.session_state.status_hi = f"Speech2Text is {'ON' if st.session_state.hindi_only_loaded else 'OFF'}: {status}"
        st.markdown(f'<div class="status-box">{st.session_state.status_hi}</div>', unsafe_allow_html=True)

        # Microphone recording
        st.markdown("### Record Audio", unsafe_allow_html=True)
        if not st.session_state.working_devices:
            st.error("No working microphones found! Please upload an audio file instead.")
        else:
            device_names = [name for idx, name in st.session_state.working_devices]
            chosen_name = st.selectbox("Choose microphone input device", device_names, key="mic_select_hi")
            chosen_idx = None
            for idx, name in st.session_state.working_devices:
                if name == chosen_name:
                    chosen_idx = idx
                    break
            duration = st.number_input(
                "Recording duration (seconds)",
                min_value=1,
                max_value=600,
                value=5,
                step=1,
                key="duration_hi"
            )
            if st.button("Record Audio", key=f"record_hi_{uuid.uuid4()}"):
                try:
                    sd.default.device = (chosen_idx, None)
                    st.info(f"Recording for {duration} seconds from '{chosen_name}'...")
                    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
                    sd.wait()
                    st.success("Recording finished!")
                    rms = np.sqrt(np.mean(recording**2))
                    st.write(f"Recorded audio RMS amplitude: {rms:.6f}")
                    if rms < 1e-4:
                        st.warning("Warning: Recorded audio seems silent!")
                    st.session_state.recorded_audio = (44100, recording)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        sf.write(tmpfile.name, recording.astype(np.float32), 44100)
                        st.audio(tmpfile.name)
                        os.remove(tmpfile.name)
                except Exception as e:
                    st.error(f"Error recording audio: {str(e)}")

        uploaded_file_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_hi")
        
        if st.button("Transcribe with Speech2Text", key="transcribe_hi"):
            audio = st.session_state.recorded_audio if st.session_state.recorded_audio else uploaded_file_hi
            if audio:
                transcription, duration, proc_time = transcribe_hindi_only(audio)
                st.session_state.transcription_hi = transcription
                st.session_state.duration_hi = duration
                st.session_state.proc_time_hi = proc_time
            else:
                st.error("No audio input provided. Please record or upload an audio file.")
        
        st.text_area("Transcription", st.session_state.transcription_hi, height=150, disabled=True, key="transcription_hi")
        col5, col6 = st.columns(2)
        with col5:
            st.text_input("Audio Duration", st.session_state.duration_hi, disabled=True, key="duration_hi")
        with col6:
            st.text_input("Processing Time", st.session_state.proc_time_ms, placeholder="proc_time_hi", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
