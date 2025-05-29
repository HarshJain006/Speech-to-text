import streamlit as st
import numpy as np
import time
import torch
from transformers import pipeline
import librosa
import soundfile as sf
import sounddevice as sd
import tempfile
import uuid
import re
import os
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
if not hasattr(st.session_state, 'initialized'):
    st.session_state.initialized = True
    st.session_state.english_hindi_transcriber = None
    st.session_state.english_hindi_loaded = False
    st.session_state.hindi_only_transcriber = None
    st.session_state.hindi_only_loaded = False
    st.session_state.transcription_en_hi = ""
    st.session_state.transcription_duration_en_hi = "0.0 seconds"
    st.session_state.transcription_proc_time_en_hi = "0.0 seconds"
    st.session_state.transcription_hi = ""
    st.session_state.transcription_duration_hi = "0.0 seconds"
    st.session_state.transcription_proc_time_hi = "0.0 seconds"
    st.session_state.status_en_hi = "Speech2Text is OFF"
    st.session_state.status_hi = "Speech2Text is OFF"
    st.session_state.working_devices = []
    st.session_state.recorded_audio = None
    st.session_state.device_error = None
    st.session_state.recorded_audio_path = None

# Device enumeration and testing functions
def list_input_devices():
    try:
        logger.debug("Enumerating audio devices...")
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append((i, dev['name']))
        logger.debug(f"Found {len(input_devices)} input devices: {input_devices}")
        return input_devices
    except Exception as e:
        logger.error(f"Error enumerating devices: {str(e)}")
        return []

def test_device(device_index, duration=1, fs=44100):
    try:
        logger.debug(f"Testing device index {device_index}")
        sd.default.device = (device_index, None)
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        if np.any(np.abs(recording) > 0.001):
            logger.debug(f"Device {device_index} is functional")
            return True
        else:
            logger.debug(f"Device {device_index} recorded silence")
            return False
    except Exception as e:
        logger.error(f"Error testing device {device_index}: {str(e)}")
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
            with st.spinner("Loading Hindi-only Speech2Text model..."):
                # Try loading AI4Bharat model first
                try:
                    st.session_state.hindi_only_transcriber = pipeline(
                        "automatic-speech-recognition",
                        model="AI4Bharat/indicwav2vec-hindi",
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    logger.info("Loaded AI4Bharat/indicwav2vec-hindi successfully")
                except Exception as e:
                    logger.warning(f"Failed to load AI4Bharat/indicwav2vec-hindi: {str(e)}. Falling back to facebook/mms-1b-all.")
                    # Fallback to facebook/mms-1b-all with Hindi language
                    st.session_state.hindi_only_transcriber = pipeline(
                        "automatic-speech-recognition",
                        model="facebook/mms-1b-all",
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    logger.info("Loaded facebook/mms-1b-all as fallback")
                st.session_state.hindi_only_loaded = True
                return "Speech2Text (Hindi Only) loaded successfully"
        except Exception as e:
            logger.error(f"Error loading Hindi-only model: {str(e)}")
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
        if isinstance(audio, tuple):  # From sounddevice
            input_sr, y = audio
            if len(y) == 0:
                return None, "0.0 seconds", "Empty audio data from microphone"
            y = y.astype(np.float32)
            if len(y.shape) == 2:
                y = np.mean(y, axis=1)
            if input_sr != sr:
                y = librosa.resample(y, orig_sr=input_sr, target_sr=sr)
        else:  # From file upload
            if hasattr(audio, 'name'):
                ext = os.path.splitext(audio.name)[1].lower()
                if ext not in ['.wav', '.mp3']:
                    return None, "0.0 seconds", f"Unsupported file format: {ext}. Please upload WAV or MP3."
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
        logger.error(f"Error processing audio: {str(e)}")
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
        logger.error(f"Error in English/Hindi transcription: {str(e)}")
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
            # Check if using MMS model
            if st.session_state.hindi_only_transcriber.model.name_or_path == "facebook/mms-1b-all":
                result = st.session_state.hindi_only_transcriber(
                    {"sampling_rate": sr, "raw": y},
                    generate_kwargs={"language": "hin", "task": "transcribe"}
                )
            else:
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
        logger.error(f"Error in Hindi-only transcription: {str(e)}")
        return (
            f"Error in Speech2Text transcription: {str(e)}",
            audio_duration_str,
            "Error during processing"
        )

# Main app
def main():
    st.markdown('<div class="header"><h1>Speech Recognition System</h1></div>', unsafe_allow_html=True)
    st.markdown("### Choose between Speech2Text models", unsafe_allow_html=True)

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
        if st.button("Scan Devices and Record", key="scan_record_en_hi"):
            try:
                logger.debug("Scanning devices for English/Hindi tab")
                st.session_state.working_devices = []
                devices = list_input_devices()
                if not devices:
                    st.session_state.device_error = "No audio input devices detected. Please check your microphone."
                else:
                    for idx, name in devices:
                        if test_device(idx):
                            st.session_state.working_devices.append((idx, name))
                    if not st.session_state.working_devices:
                        st.session_state.device_error = "No working microphones found! Please ensure your microphone is connected and functional."
                    else:
                        st.session_state.device_error = None
                        st.success(f"Found {len(st.session_state.working_devices)} working microphone(s)")
            except Exception as e:
                logger.error(f"Error during device scan: {str(e)}")
                st.session_state.device_error = f"Error scanning devices: {str(e)}"

        if st.session_state.device_error:
            st.error(f"Error: {st.session_state.device_error}")
        elif not st.session_state.working_devices:
            st.info("Click 'Scan Devices and Record' to scan for available microphones.")
        else:
            device_names = [name for idx, name in st.session_state.working_devices]
            chosen_name = st.selectbox("Choose microphone input device", device_names, key="mic_select_en_hi")
            chosen_idx = next(idx for idx, name in st.session_state.working_devices if name == chosen_name)
            
            duration = st.number_input(
                "Enter recording duration (seconds)",
                min_value=1,
                max_value=600,
                value=5,
                step=1,
                key="record_duration_en_hi"
            )
            
            if st.button("Record Audio", key="record_en_hi"):
                try:
                    sd.default.device = (chosen_idx, None)
                    st.info(f"Recording for {duration} seconds from '{chosen_name}'...")
                    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
                    sd.wait()
                    st.success("Recording finished!")
                    
                    st.session_state.recorded_audio = (44100, recording)
                    rms = np.sqrt(np.mean(recording**2))
                    st.write(f"Recorded audio RMS amplitude: {rms:.6f}")
                    if rms < 1e-4:
                        st.warning("Warning: Recorded audio seems silent!")
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        sf.write(tmpfile.name, recording.astype(np.float32), 44100)
                        st.audio(tmpfile.name)
                        st.session_state.recorded_audio_path = tmpfile.name
                except Exception as e:
                    logger.error(f"Error recording audio: {str(e)}")
                    st.error(f"Error recording audio: {str(e)}")
                finally:
                    if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                        try:
                            os.remove(st.session_state.recorded_audio_path)
                            st.session_state.recorded_audio_path = None
                        except Exception as e:
                            logger.error(f"Error cleaning up temporary file: {str(e)}")

        uploaded_file_en_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_en_hi")
        
        if st.button("Transcribe with Speech2Text", key="transcribe_en_hi"):
            audio = st.session_state.recorded_audio if st.session_state.recorded_audio else uploaded_file_en_hi
            if audio:
                transcription, duration, proc_time = transcribe_english_hindi(audio, language_selection)
                st.session_state.transcription_en_hi = transcription
                st.session_state.transcription_duration_en_hi = duration
                st.session_state.transcription_proc_time_en_hi = proc_time
                # Clean up recorded audio after transcription
                st.session_state.recorded_audio = None
                if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                    try:
                        os.remove(st.session_state.recorded_audio_path)
                        st.session_state.recorded_audio_path = None
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary file after transcription: {str(e)}")
            else:
                st.error("No audio input provided. Please record or upload an audio file.")
        
        st.text_area("Transcription", st.session_state.transcription_en_hi, height=150, disabled=True, key="transcription_en_hi")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Audio Duration", st.session_state.transcription_duration_en_hi)
        with col4:
            st.metric("Processing Time", st.session_state.transcription_proc_time_en_hi)

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
        if st.button("Scan Devices and Record", key="scan_record_hi"):
            try:
                logger.debug("Scanning devices for Hindi-only tab")
                st.session_state.working_devices = []
                devices = list_input_devices()
                if not devices:
                    st.session_state.device_error = "No audio input devices detected. Please check your microphone."
                else:
                    for idx, name in devices:
                        if test_device(idx):
                            st.session_state.working_devices.append((idx, name))
                    if not st.session_state.working_devices:
                        st.session_state.device_error = "No working microphones found! Please ensure your microphone is connected and functional."
                    else:
                        st.session_state.device_error = None
                        st.success(f"Found {len(st.session_state.working_devices)} working microphone(s)")
            except Exception as e:
                logger.error(f"Error during device scan: {str(e)}")
                st.session_state.device_error = f"Error scanning devices: {str(e)}"

        if st.session_state.device_error:
            st.error(f"Error: {st.session_state.device_error}")
        elif not st.session_state.working_devices:
            st.info("Click 'Scan Devices and Record' to scan for available microphones.")
        else:
            device_names = [name for idx, name in st.session_state.working_devices]
            chosen_name = st.selectbox("Choose microphone input device", device_names, key="mic_select_hi")
            chosen_idx = next(idx for idx, name in st.session_state.working_devices if name == chosen_name)
            
            duration = st.number_input(
                "Enter recording duration (seconds)",
                min_value=1,
                max_value=600,
                value=5,
                step=1,
                key="record_duration_hi"
            )
            
            if st.button("Record Audio", key="record_hi"):
                try:
                    sd.default.device = (chosen_idx, None)
                    st.info(f"Recording for {duration} seconds from '{chosen_name}'...")
                    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
                    sd.wait()
                    st.success("Recording finished!")
                    
                    st.session_state.recorded_audio = (44100, recording)
                    rms = np.sqrt(np.mean(recording**2))
                    st.write(f"Recorded audio RMS amplitude: {rms:.6f}")
                    if rms < 1e-4:
                        st.warning("Warning: Recorded audio seems silent!")
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        sf.write(tmpfile.name, recording.astype(np.float32), 44100)
                        st.audio(tmpfile.name)
                        st.session_state.recorded_audio_path = tmpfile.name
                except Exception as e:
                    logger.error(f"Error recording audio: {str(e)}")
                    st.error(f"Error recording audio: {str(e)}")
                finally:
                    if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                        try:
                            os.remove(st.session_state.recorded_audio_path)
                            st.session_state.recorded_audio_path = None
                        except Exception as e:
                            logger.error(f"Error cleaning up temporary file: {str(e)}")

        uploaded_file_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_hi")
        
        if st.button("Transcribe with Speech2Text", key="transcribe_hi"):
            audio = st.session_state.recorded_audio if st.session_state.recorded_audio else uploaded_file_hi
            if audio:
                transcription, duration, proc_time = transcribe_hindi_only(audio)
                st.session_state.transcription_hi = transcription
                st.session_state.transcription_duration_hi = duration
                st.session_state.transcription_proc_time_hi = proc_time
                # Clean up recorded audio after transcription
                st.session_state.recorded_audio = None
                if st.session_state.recorded_audio_path and os.path.exists(st.session_state.recorded_audio_path):
                    try:
                        os.remove(st.session_state.recorded_audio_path)
                        st.session_state.recorded_audio_path = None
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary file after transcription: {str(e)}")
            else:
                st.error("No audio input provided. Please record or upload an audio file.")
        
        st.text_area("Transcription", st.session_state.transcription_hi, height=150, disabled=True, key="transcription_hi")
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Audio Duration", st.session_state.transcription_duration_hi)
        with col6:
            st.metric("Processing Time", st.session_state.transcription_proc_time_hi)

if __name__ == "__main__":
    main()
