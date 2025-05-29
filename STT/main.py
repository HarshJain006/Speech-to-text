import streamlit as st
import numpy as np
import time
import torch
import librosa
import soundfile as sf
import io
import uuid
import re
import os
import warnings
from transformers import pipeline

# Try importing sounddevice
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    st.session_state.debug_info = "sounddevice module not found. Microphone recording disabled."

# Suppress benign PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

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
        .warning-box {
            background-color: #FFF3CD;
            color: #856404;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #FFEEBA;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Audio device management
def list_input_devices():
    if not SOUNDDEVICE_AVAILABLE:
        return []
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append((i, dev['name']))
        return input_devices
    except Exception as e:
        st.session_state.debug_info = f"Error listing devices: {str(e)}"
        return []

def test_device(device_index, duration=1, fs=44100):
    if not SOUNDDEVICE_AVAILABLE:
        return False
    try:
        sd.default.device = (device_index, None)
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        return np.any(np.abs(recording) > 0.001)
    except Exception:
        return False

# Initialize session state
def initialize_session_state():
    defaults = {
        'english_hindi_transcriber': None,
        'device_en_hi': None,
        'device_hi': None,
        'english_hindi_loaded': False,
        'hindi_only_transcriber': None,
        'hindi_only_loaded': False,
        'transcription_en_hi': "",
        'duration_en_hi': "0.0 seconds",
        'proc_time_en_hi': "0.0 seconds",
        'transcription_hi': "",
        'duration_hi': "0.0 seconds",
        'proc_time_hi': "0.0 seconds",
        'status_en_hi': "Speech2Text is OFF",
        'status_hi': "Speech2Text is OFF",
        'debug_info': "",
        'retry_count_en': 0,
        'retry_count_hi': 0,
        'audio_input_en_hi': None,
        'audio_input_hi': None,
        'working_devices': [],
        'recording_en_hi': None,
        'recording_hi': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Model loading/unloading
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
            return f"Error loading Speech2Text: {str(e)}"
    return "Speech2Text already loaded"

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
            return f"Error loading Speech2Text: {str(e)}"
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

# Audio processing
def process_audio(audio, sr=16000):
    try:
        start_time = time.time()
        if isinstance(audio, np.ndarray):  # From sounddevice
            y = audio.flatten().astype(np.float32)
            input_sr = 44100
            debug_info = f"Microphone input: shape={y.shape}, dtype={y.dtype}, sample_rate={input_sr}, max_amplitude={np.max(np.abs(y)) if len(y) > 0 else 0}"
            st.session_state.debug_info = debug_info
            max_abs = np.max(np.abs(y))
            if max_abs > 0:
                y *= (0.25 / max_abs)  # Reduced amplification
            if np.max(np.abs(y)) < 1e-6:
                return None, "0.0 seconds", "Microphone audio too quiet"
            if input_sr != sr:
                y = librosa.resample(y, orig_sr=input_sr, target_sr=sr)
        elif isinstance(audio, bytes):  # For compatibility
            with io.BytesIO(audio) as wav_io:
                y, input_sr = sf.read(wav_io)
            if y is None or len(y) == 0:
                return None, "0.0 seconds", "Empty audio data"
            if not isinstance(y, np.ndarray):
                return None, "0.0 seconds", f"Invalid data type: {type(y)}"
            if input_sr <= 0:
                return None, "0.0 seconds", f"Invalid sample rate: {input_sr}"
            debug_info = f"Microphone input: shape={y.shape}, dtype={y.dtype}, sample_rate={input_sr}, max_amplitude={np.max(np.abs(y)) if len(y) > 0 else 0}"
            st.session_state.debug_info = debug_info
            y = y.astype(np.float32)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            max_abs = np.max(np.abs(y))
            if max_abs > 0:
                y *= (0.25 / max_abs)
            if np.max(np.abs(y)) < 1e-6:
                return None, "0.0 seconds", "Microphone audio too quiet"
            if input_sr != sr:
                y = librosa.resample(y, orig_sr=input_sr, target_sr=sr)
        else:  # File upload
            if hasattr(audio, 'name'):
                ext = os.path.splitext(audio.name)[1].lower()
                if ext not in ['.wav', '.mp3']:
                    return None, "0.0 seconds", f"Unsupported file format: {ext}"
            temp_file = f"temp_audio_{uuid.uuid4()}{ext if 'ext' in locals() else '.wav'}"
            with open(temp_file, "wb") as f:
                f.write(audio.read())
            try:
                y, input_sr = sf.read(temp_file)
                if len(y) == 0:
                    return None, "0.0 seconds", "No audio data found"
                y = y.astype(np.float32)
                if len(y.shape) > 1:
                    y = np.mean(y, axis=1)
                if input_sr != sr:
                    y = librosa.resample(y, orig_sr=input_sr, target_sr=sr)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        audio_duration = len(y) / sr
        processing_time = time.time() - start_time
        debug_info = f"{st.session_state.debug_info}, audio processing time: {processing_time:.2f} seconds"
        st.session_state.debug_info = debug_info
        audio_duration_str = f"{audio_duration:.2f} seconds"
        if np.max(np.abs(y)) > 0:
            y /= np.max(np.abs(y))
        return (sr, y), audio_duration_str, None
    except Exception as e:
        return None, "0.0 seconds", f"Error processing audio: {str(e)}"

# Transcription functions
def transcribe_english_hindi(audio, language, max_retries=2):
    if not st.session_state.english_hindi_loaded:
        return "Please turn on Speech2Text first", "0.0 seconds", "0.0 seconds"
    for attempt in range(max_retries):
        processed_audio, audio_duration_str, error = process_audio(audio)
        if error:
            st.session_state.retry_count_en += 1
            if st.session_state.retry_count_en >= max_retries:
                return f"{error} (Retry {st.session_state.retry_count_en}/{max_retries})", audio_duration_str, "0.0 seconds"
            continue
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
                transcription = "\n".join([chunk["text"].strip() for chunk in result.get("chunks", []) if chunk["text"].strip()]) or result.get("text", "").strip()
                if not transcription:
                    transcription = "Warning: No transcription generated"
                processing_time = time.time() - start_time
                processing_time_str = f"{processing_time:.2f} seconds"
                st.session_state.retry_count_en = 0
                return transcription, audio_duration_str, processing_time_str
        except Exception as e:
            st.session_state.retry_count_en += 1
            if st.session_state.retry_count_en >= max_retries:
                return f"Error in transcription: {str(e)}", audio_duration_str, "0.0 seconds"
    return "Transcription failed", audio_duration_str, "0.0 seconds"

def transcribe_hindi_only(audio, max_retries=2):
    if not st.session_state.hindi_only_loaded:
        return "Please turn on Speech2Text first", "0.0 seconds", "0.0 seconds"
    for attempt in range(max_retries):
        processed_audio, audio_duration_str, error = process_audio(audio)
        if error:
            st.session_state.retry_count_hi += 1
            if st.session_state.retry_count_hi >= max_retries:
                return f"{error} (Retry {st.session_state.retry_count_hi}/{max_retries})", audio_duration_str, "0.0 seconds"
            continue
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
                if not transcription:
                    transcription = "Warning: No transcription generated"
                processing_time = time.time() - start_time
                processing_time_str = f"{processing_time:.2f} seconds"
                st.session_state.retry_count_hi = 0
                return transcription, audio_duration_str, processing_time_str
        except Exception as e:
            st.session_state.retry_count_hi += 1
            if st.session_state.retry_count_hi >= max_retries:
                return f"Error in transcription: {str(e)}", audio_duration_str, "0.0 seconds"
    return "Transcription failed", audio_duration_str, "0.0 seconds"

# Main app
def main():
    initialize_session_state()
    st.markdown('<div class="header"><h1>Speech Recognition System</h1></div>', unsafe_allow_html=True)
    st.markdown("Select a Speech-to-Text model and record/upload audio.", unsafe_allow_html=True)
    st.markdown("**Note**: Ensure microphone is connected and working. Use Chrome/Firefox for best compatibility.", unsafe_allow_html=True)

    # Scan working devices
    if not st.session_state.working_devices and SOUNDDEVICE_AVAILABLE:
        devices = list_input_devices()
        for idx, name in devices:
            if test_device(idx):
                st.session_state.working_devices.append((idx, name))
        if not st.session_state.working_devices:
            st.markdown('<div class="warning-box">No working microphones found! Check connections and permissions.</div>', unsafe_allow_html=True)

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
                help="Tiny or small recommended for Indian accents",
                key="model_size_en_hi"
            )
            power_button_en_hi = st.button(
                "Turn ON Speech2Text" if not st.session_state.english_hindi_loaded else "Turn OFF Speech2Text",
                key="power_button_en_hi"
            )
            if power_button_en_hi:
                status = load_english_hindi(model_size) if not st.session_state.english_hindi_loaded else unload_english_hindi()
                st.session_state.status_en_hi = f"Status: Speech2Text is {'ON' if st.session_state.english_hindi_loaded else 'OFF'}: {status}"
            st.markdown(f'<div class="status-box">{st.session_state.status_en_hi}</div>', unsafe_allow_html=True)
        with col2:
            language_selection = st.radio(
                "Select Language",
                ["english", "hindi"],
                index=0,
                help="Select language for transcription",
                key="language_selection_en_hi"
            )

        st.markdown("**Record Audio**")
        if not SOUNDDEVICE_AVAILABLE:
            st.markdown('<div class="warning-box">Microphone recording disabled due to missing sounddevice module. Please use file upload.</div>', unsafe_allow_html=True)
        elif st.session_state.working_devices:
            device_names = [name for idx, name in st.session_state.working_devices]
            chosen_name_en_hi = st.selectbox("Choose microphone input device", device_names, key="mic_en_hi")
            chosen_idx_en_hi = None
            for idx, name in st.session_state.working_devices:
                if name == chosen_name_en_hi:
                    chosen_idx_en_hi = idx
                    break
            duration_en_hi = st.number_input("Recording duration (seconds)", min_value=1, max_value=600, value=5, step=1, key="duration_en_hi")
            if st.button("Record Audio", key="record_en_hi"):
                try:
                    sd.default.device = (chosen_idx_en_hi, None)
                    st.info(f"Recording for {duration_en_hi} seconds from '{chosen_name_en_hi}'...")
                    recording = sd.rec(int(duration_en_hi * 44100), samplerate=44100, channels=1)
                    sd.wait()
                    st.session_state.recording_en_hi = recording
                    st.success("Recording finished!")
                    rms = np.sqrt(np.mean(recording**2))
                    st.markdown(f'<div class="status-box">RMS amplitude: {rms:.6f}</div>', unsafe_allow_html=True)
                    if rms < 1e-4:
                        st.markdown('<div class="warning-box">Warning: Recorded audio seems silent!</div>', unsafe_allow_html=True)
                    with io.BytesIO() as wav_io:
                        sf.write(wav_io, recording.astype(np.float32), 44100, format='WAV')
                        wav_io.seek(0)
                        st.audio(wav_io, format='audio/wav')
                except Exception as e:
                    st.markdown(f'<div class="warning-box">Error recording: {str(e)}</div>', unsafe_allow_html=True)
                    st.session_state.debug_info = f"Recording error: {str(e)}"
        else:
            st.markdown('<div class="warning-box">No microphones available for recording.</div>', unsafe_allow_html=True)

        if st.button("Retry Recording", key="retry_en_hi"):
            st.session_state.audio_input_en_hi = None
            st.session_state.recording_en_hi = None
            st.session_state.retry_count_en = 0
            st.session_state.transcription_en_hi = ""

        uploaded_file_en_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_en_hi")

        if st.button("Transcribe with Speech2Text", key="transcribe_en_hi"):
            audio = st.session_state.recording_en_hi or uploaded_file_en_hi
            if audio is not None:
                st.session_state.audio_input_en_hi = audio
                transcription, duration, proc_time = transcribe_english_hindi(audio, language_selection)
                st.session_state.transcription_en_hi = transcription
                st.session_state.duration_en_hi = duration
                st.session_state.proc_time_en_hi = proc_time
                if "too quiet" in transcription.lower():
                    st.markdown('<div class="warning-box">Warning: Audio is too quiet. Increase microphone gain or speak louder.</div>', unsafe_allow_html=True)
            else:
                st.session_state.transcription_en_hi = "No audio input provided. Please record or upload an audio file."
                st.markdown('<div class="warning-box">No audio input provided.</div>', unsafe_allow_html=True)

        st.text_area("Transcription", st.session_state.transcription_en_hi, height=150, disabled=True, key="transcription_en_hi")
        col3, col4 = st.columns(2)
        with col3:
            st.text_input("Audio Duration", st.session_state.duration_en_hi, disabled=True, key="duration_en_hi")
        with col4:
            st.text_input("Processing Time", st.session_state.proc_time_en_hi, disabled=True, key="proc_time_en_hi")
        if st.session_state.debug_info:
            st.markdown(f'<div class="warning-box">Debug: {st.session_state.debug_info}</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### Specialized for Hindi language only", unsafe_allow_html=True)
        power_button_hi = st.button(
            "Turn ON Speech2Text" if not st.session_state.hindi_only_loaded else "Turn OFF Speech2Text",
            key="power_button_hi"
        )
        if power_button_hi:
            status = load_hindi_only() if not st.session_state.hindi_only_loaded else unload_hindi_only()
            st.session_state.status_hi = f"Status: Speech2Text is {'ON' if st.session_state.hindi_only_loaded else 'OFF'}: {status}"
        st.markdown(f'<div class="status-box">{st.session_state.status_hi}</div>', unsafe_allow_html=True)

        st.markdown("**Record Audio**")
        if not SOUNDDEVICE_AVAILABLE:
            st.markdown('<div class="warning-box">Microphone recording disabled due to missing sounddevice module. Please use file upload.</div>', unsafe_allow_html=True)
        elif st.session_state.working_devices:
            device_names = [name for idx, name in st.session_state.working_devices]
            chosen_name_hi = st.selectbox("Choose microphone input device", device_names, key="mic_hi")
            chosen_idx_hi = None
            for idx, name in st.session_state.working_devices:
                if name == chosen_name_hi:
                    chosen_idx_hi = idx
                    break
            duration_hi = st.number_input("Recording duration (seconds)", min_value=1, max_value=600, value=5, step=1, key="duration_hi")
            if st.button("Record Audio", key="record_hi"):
                try:
                    sd.default.device = (chosen_idx_hi, None)
                    st.info(f"Recording for {duration_hi} seconds from '{chosen_name_hi}'...")
                    recording = sd.rec(int(duration_hi * 44100), samplerate=44100, channels=1)
                    sd.wait()
                    st.session_state.recording_hi = recording
                    st.success("Recording finished!")
                    rms = np.sqrt(np.mean(recording**2))
                    st.markdown(f'<div class="status-box">RMS amplitude: {rms:.6f}</div>', unsafe_allow_html=True)
                    if rms < 1e-4:
                        st.markdown('<div class="warning-box">Warning: Recorded audio seems silent!</div>', unsafe_allow_html=True)
                    with io.BytesIO() as wav_io:
                        sf.write(wav_io, recording.astype(np.float32), 44100, format='WAV')
                        wav_io.seek(0)
                        st.audio(wav_io, format='audio/wav')
                except Exception as e:
                    st.markdown(f'<div class="warning-box">Error recording: {str(e)}</div>', unsafe_allow_html=True)
                    st.session_state.debug_info = f"Recording error: {str(e)}"
        else:
            st.markdown('<div class="warning-box">No microphones available for recording.</div>', unsafe_allow_html=True)

        if st.button("Retry Recording", key="retry_hi"):
            st.session_state.audio_input_hi = None
            st.session_state.recording_hi = None
            st.session_state.retry_count_hi = 0
            st.session_state.transcription_hi = ""

        uploaded_file_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_hi")

        if st.button("Transcribe with Speech2Text", key="transcribe_hi"):
            audio = st.session_state.recording_hi or uploaded_file_hi
            if audio is not None:
                st.session_state.audio_input_hi = audio
                transcription, duration, proc_time = transcribe_hindi_only(audio)
                st.session_state.transcription_hi = transcription
                st.session_state.duration_hi = duration
                st.session_state.proc_time_hi = proc_time
                if "too quiet" in transcription.lower():
                    st.markdown('<div class="warning-box">Warning: Audio is too quiet. Increase microphone gain or speak louder.</div>', unsafe_allow_html=True)
            else:
                st.session_state.transcription_hi = "No audio input provided. Please record or upload an audio file."
                st.markdown('<div class="warning-box">No audio input provided.</div>', unsafe_allow_html=True)

        st.text_area("Transcription", st.session_state.transcription_hi, height=150, disabled=True, key="transcription_hi")
        col5, col6 = st.columns(2)
        with col5:
            st.text_input("Audio Duration", st.session_state.duration_hi, disabled=True, key="duration_hi")
        with col6:
            st.text_input("Processing Time", st.session_state.proc_time_hi, disabled=True, key="proc_time_hi")
        if st.session_state.debug_info:
            st.markdown(f'<div class="warning-box">Debug: {st.session_state.debug_info}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
