import streamlit as st
import numpy as np
import time
import torch
from transformers import pipeline
import librosa
import soundfile as sf
import tempfile
import uuid
import re
import os
import logging
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import av

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
        if isinstance(audio, bytes):  # From webrtc_streamer
            # Write bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(audio)
                tmpfile_path = tmpfile.name
            try:
                y, input_sr = sf.read(tmpfile_path)
                if len(y) == 0:
                    return None, "No audio data from microphone", "0.0 seconds"
                y = y.astype(np.float32)
                if len(y.shape) == 2:
                    y = np.mean(y, axis=1)
                if input_sr != sr:
                    y = librosa.resample(y, orig_sr=input_sr, target_sr=sr)
            finally:
                if os.path.exists(tmpfile_path):
                    os.remove(tmpfile_path)
        else:  # From file upload
            if hasattr(audio, 'name'):
                ext = os.path.splitext(audio.name)[1].lower()
                if ext not in ['.wav', '.mp3']:
                    return None, "0.0 seconds", f"Unsupported file format:: {ext}"
                temp_file = f"temp_audio_{uuid.uuid4()}.{ext}"
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
            "Please turn on Speech2Text first",
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
                generate_kwargs={"language": "lang_code", "task": "transcribe"},
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
            "Please turn on Speech2Text first",
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

# Function to handle WebRTC audio
def handle_webrtc_audio(key):
    audio_queue = queue.Queue()
    st.session_state[f"recording_status_{key}"] = "Not recording"

    def process_audio_frame(audio_frame: av.AudioFrame) -> av.AudioFrame:
        try:
            audio_data = audio_frame.to_ndarray().flatten()
            audio_queue.put(audio_data)
            logger.debug(f"Received audio frame: {len(audio_data)} samples")
        except Exception as e:
            logger.error(f"Error processing audio frame: {str(e)}")
        return audio_frame

    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDONLY,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
        audio_receiver_size=2048,  # Increased buffer size
        async_processing=True,
        audio_frame_callback=process_audio_frame
    )

    # Update recording status
    if ctx.state.playing:
        st.session_state[f"recording_status_{key}"] = "Recording..."
        st.info(f"Recording audio for {key}. Click 'Stop' to finish.")
    else:
        if st.session_state[f"recording_status_{key}"] == "Recording...":
            st.session_state[f"recording_status_{key}"] = "Stopped"
            st.success(f"Recording stopped for {key}.")

    # Display recording status
    st.write(f"Status: {st.session_state[f'recording_status_{key}']}")

    if ctx.state.playing:
        try:
            audio_frames = []
            timeout = 15  # Increased timeout to 15 seconds
            start_time = time.time()
            while time.time() - start_time < timeout and ctx.state.playing:
                try:
                    audio_data = audio_queue.get(timeout=1)
                    if audio_data is not None:
                        audio_frames.append(audio_data)
                        logger.debug(f"Collected audio frame, total frames: {len(audio_frames)}")
                except queue.Empty:
                    logger.debug("Audio queue empty, waiting...")
                    continue
            if audio_frames:
                logger.info(f"Collected {len(audio_frames)} audio frames")
                audio_data = np.concatenate(audio_frames, axis=0)
                # Convert to bytes for processing
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    sf.write(tmpfile.name, audio_data, 16000)
                    with open(tmpfile.name, "rb") as f:
                        audio_bytes = f.read()
                    os.remove(tmpfile.name)
                logger.info("Audio captured successfully")
                return audio_bytes
            else:
                logger.warning("No audio frames captured")
                st.warning("No audio data captured. Ensure your microphone is enabled and try again.")
                return None
        except Exception as e:
            logger.error(f"Error in WebRTC audio processing: {str(e)}")
            st.error(f"Error capturing audio: {str(e)}")
            return None
    return None

# Main app
def main():
    # Initialize session state
    default_state = {
        'initialized': True,
        'english_hindi_transcriber': None,
        'english_hindi_loaded': False,
        'hindi_only_transcriber': None,
        'hindi_only_loaded': False,
        'transcription_en_hi': "",
        'transcription_duration_en_hi': "0.0 seconds",
        'transcription_proc_time_en_hi': "0.0 seconds",
        'transcription_hi': "",
        'transcription_duration_hi': "0.0 seconds",
        'transcription_proc_time_hi': "0.0 seconds",
        'status_en_hi': "Speech2Text is OFF",
        'status_hi': "Speech2Text is OFF",
        'recorded_audio': None,
        'recorded_audio_path': None,
        'recording_status_audio_en_hi': "Not recording",
        'recording_status_audio_hi': "Not recording"
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

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

        # Microphone recording with webrtc_streamer
        st.markdown("### Record Audio", unsafe_allow_html=True)
        st.info("Click 'Start' to begin recording and 'Stop' to finish. Ensure your microphone is enabled.")
        recorded_audio_en_hi = handle_webrtc_audio("audio_en_hi")
        if recorded_audio_en_hi:
            st.session_state.recorded_audio = recorded_audio_en_hi
            st.audio(st.session_state.recorded_audio, format="audio/wav")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(st.session_state.recorded_audio)
                y, sr = sf.read(tmpfile.name)
                rms = np.sqrt(np.mean(y**2))
                st.write(f"Recorded audio RMS amplitude: {rms:.6f}")
                if rms < 1e-4:
                    st.warning("Warning: Recorded audio seems silent!")
                st.session_state.recorded_audio_path = tmpfile.name
            if os.path.exists(st.session_state.recorded_audio_path):
                try:
                    os.remove(st.session_state.recorded_audio_path)
                    st.session_state.recorded_audio_path = None
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

        uploaded_file_en_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_en_hi")
        
        if st.button("Transcribe with Speech2Text", key="transcribe_en_hi"):
            audio = st.session_state.recorded_audio if st.session_state.recorded_audio else uploaded_file_en_hi
            if audio:
                logger.info("Starting transcription for English/Hindi")
                transcription, duration, proc_time = transcribe_english_hindi(audio, language_selection)
                st.session_state.transcription_en_hi = transcription
                st.session_state.transcription_duration_en_hi = duration
                st.session_state.transcription_proc_time_en_hi = proc_time
                st.session_state.recorded_audio = None  # Clear after transcription
                logger.info("Transcription completed")
            else:
                logger.warning("No audio input provided for English/Hindi")
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

        # Microphone recording with webrtc_streamer
        st.markdown("### Record Audio", unsafe_allow_html=True)
        st.info("Click 'Start' to begin recording and 'Stop' to finish. Ensure your microphone is enabled.")
        recorded_audio_hi = handle_webrtc_audio("audio_hi")
        if recorded_audio_hi:
            st.session_state.recorded_audio = recorded_audio_hi
            st.audio(st.session_state.recorded_audio, format="audio/wav")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(st.session_state.recorded_audio)
                y, sr = sf.read(tmpfile.name)
                rms = np.sqrt(np.mean(y**2))
                st.write(f"Recorded audio RMS amplitude: {rms:.6f}")
                if rms < 1e-4:
                    st.warning("Warning: Recorded audio seems silent!")
                st.session_state.recorded_audio_path = tmpfile.name
            if os.path.exists(st.session_state.recorded_audio_path):
                try:
                    os.remove(st.session_state.recorded_audio_path)
                    st.session_state.recorded_audio_path = None
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

        uploaded_file_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_hi")
        
        if st.button("Transcribe with Speech2Text", key="transcribe_hi"):
            audio = st.session_state.recorded_audio if st.session_state.recorded_audio else uploaded_file_hi
            if audio:
                logger.info("Starting transcription for Hindi-only")
                transcription, duration, proc_time = transcribe_hindi_only(audio)
                st.session_state.transcription_hi = transcription
                st.session_state.transcription_duration_hi = duration
                st.session_state.transcription_proc_time_hi = proc_time
                st.session_state.recorded_audio = None  # Clear after transcription
                logger.info("Transcription completed")
            else:
                logger.warning("No audio input provided for Hindi-only")
                st.error("No audio input provided. Please record or upload an audio file.")
        
        st.text_area("Transcription", st.session_state.transcription_hi, height=150, disabled=True, key="transcription_hi")
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Audio Duration", st.session_state.transcription_duration_hi)
        with col6:
            st.metric("Processing Time", st.session_state.transcription_proc_time_hi)

if __name__ == "__main__":
    main()
