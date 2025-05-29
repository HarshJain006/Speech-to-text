import streamlit as st
from audio_recorder_streamlit import audio_recorder
import streamlit.components.v1 as components
import numpy as np
import time
import torch
from transformers import pipeline
import librosa
import soundfile as sf
import io
import uuid
import re
import os

# Custom CSS for styling and microphone feedback
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
        .audio-recorder {
            background-color: #FFFFFF;
            border: 2px solid #4169E1;
            border-radius: 8px;
            padding: 10px;
        }
        .audio-recorder.recording {
            background-color: #FF0000;
            border-color: #C0C0C0;
        }
    </style>
""", unsafe_allow_html=True)

# Custom component for audio device selection (debugging only)
def audio_device_selector(key):
    component_html = """
    <select id="audioDevice_{0}" style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #4169E1;">
        <option value="">Select a microphone (default used)</option>
    </select>
    <script>
        async function populateDevices() {{
            try {{
                // Request microphone permission
                await navigator.mediaDevices.getUserMedia({{ audio: true }});
                const devices = await navigator.mediaDevices.enumerateDevices();
                const select = document.getElementById("audioDevice_{0}");
                devices.forEach(device => {{
                    if (device.kind === "audioinput") {{
                        const option = document.createElement("option");
                        option.value = device.deviceId;
                        option.text = device.label || "Microphone " + (select.options.length + 1);
                        select.appendChild(option);
                    }}
                }});
                // Send selected deviceId to Streamlit
                select.addEventListener("change", () => {{
                    window.parent.postMessage({{
                        type: "streamlit:setComponentValue",
                        value: select.value
                    }}, "*");
                }});
            }} catch (err) {{
                console.error("Error accessing devices:", err);
                const select = document.getElementById("audioDevice_{0}");
                select.innerHTML = '<option value="">Microphone access denied</option>';
            }}
        }}
        populateDevices();
    </script>
    """.replace("{0}", key)
    components.html(component_html, height=50)
    return st.session_state.get(f"device_{key}", None)

# Initialize session state
if 'english_hindi_transcriber' not in st.session_state:
    st.session_state.english_hindi_transcriber = None
    st.session_state.device_en_hi = None
    st.session_state.device_hi = None
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
    st.session_state.debug_info = ""
    st.session_state.retry_count_en_hi = 0
    st.session_state.retry_count_hi = 0

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
        start_time = time.time()
        if isinstance(audio, bytes):  # From audio_recorder (microphone)
            with io.BytesIO(audio) as wav_io:
                y, input_sr = sf.read(wav_io)
            if y is None or len(y) == 0:
                return None, "0.0 seconds", "Empty audio data from microphone"
            if not isinstance(y, np.ndarray):
                return None, "0.0 seconds", f"Invalid microphone data type: {type(y)}"
            if input_sr <= 0:
                return None, "0.0 seconds", f"Invalid sample rate: {input_sr}"
            debug_info = f"Microphone input: shape={y.shape}, dtype={y.dtype}, sample_rate={input_sr}, max_amplitude={np.max(np.abs(y)) if len(y) > 0 else 0}"
            st.session_state.debug_info = debug_info
            y = y.astype(np.float32)
            if len(y.shape) == 2:
                y = np.mean(y, axis=1)
            max_abs = np.max(np.abs(y))
            if max_abs > 0:
                y *= (0.5 / max_abs)  # Amplify to 0.5 peak
            if np.max(np.abs(y)) < 1e-7:
                return None, "0.0 seconds", "Microphone audio is too quiet (max amplitude below threshold)"
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
        processing_time = time.time() - start_time
        debug_info = f"{st.session_state.debug_info}, audio processing time: {processing_time:.2f} seconds"
        st.session_state.debug_info = debug_info
        audio_duration_str = f"{audio_duration:.2f} seconds"
        if np.max(y) > 0:
            y /= np.max(np.abs(y))  # Normalize
        return (sr, y), audio_duration_str, None
    except Exception as e:
        return None, "0.0 seconds", f"Error processing audio: {str(e)}"

# Transcription functions
def transcribe_english_hindi(audio, language, max_retries=3):
    if not st.session_state.english_hindi_loaded:
        return (
            "Please turn on Speech2Text (English & Hindi) first",
            "0.0 seconds",
            "0.0 seconds"
        )
    for attempt in range(max_retries):
        processed_audio, audio_duration_str, error = process_audio(audio)
        if error:
            st.session_state.retry_count_en_hi += 1
            if st.session_state.retry_count_en_hi >= max_retries:
                return f"{error} (Retry {st.session_state.retry_count_en_hi}/{max_retries})", audio_duration_str, "0.0 seconds"
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
                if "chunks" in result and result["chunks"]:
                    transcription = "\n".join([chunk["text"].strip() for chunk in result["chunks"] if chunk["text"].strip()])
                else:
                    transcription = result.get("text", "").strip()
                if not transcription:
                    transcription = "Warning: No transcription generated. Audio may be too short or unclear."
                processing_time = time.time() - start_time
                processing_time_str = f"{processing_time:.2f} seconds"
                st.session_state.retry_count_en_hi = 0
                return transcription, audio_duration_str, processing_time_str
        except Exception as e:
            st.session_state.retry_count_en_hi += 1
            if st.session_state.retry_count_en_hi >= max_retries:
                return (
                    f"Error in transcription: {str(e)} (Retry {max_retries}/{max_retries})",
                    audio_duration_str,
                    "0.0 seconds"
                )
    return None, audio_duration_str, "0.0 seconds"

def transcribe_hindi_only(audio, max_retries=3):
    if not st.session_state.hindi_only_loaded:
        return (
            "Please turn on Speech2Text (Hindi Only) first",
            "0.0 seconds",
            "0.0 seconds"
        )
    for attempt in range(max_retries):
        processed_audio, audio_duration_str, error = process_audio(audio)
        if error:
            st.session_state.retry_count_hi += 1
            if st.session_state.retry_count_hi >= max_retries:
                return f"{error} (Retry {max_retries}/{max_retries})", audio_duration_str, "0.0 seconds"
            continue
        sr, y = processed_audio
        try:
            start_time = time.time()
            with st.spinner("Transcribing..."):
                result = st.session_state.hindi_only_transcriber(
                    {"sampling_rate": sr, "raw": y}
                )
                raw_text = result.get("text", "").strip()
                segments = re.split(r'[।,.?!]\s*', raw_text)
                transcription = "\n".join([segment.strip() for segment in segments if segment.strip()])
                if not transcription:
                    transcription = "Warning: No transcription generated. Audio may be too short or unclear."
                processing_time = time.time() - start_time
                processing_time_str = f"{processing_time:.2f} seconds"
                st.session_state.retry_count_hi = 0
                return transcription, audio_duration_str, processing_time_str
        except Exception as e:
            st.session_state.retry_count_hi += 1
            if st.session_state.retry_count_hi >= max_retries:
                return (
                    f"Error in transcription: {str(e)} (Retry {max_retries}/{max_retries})",
                    audio_duration_str,
                    "0.0 seconds"
                )
    return None, audio_duration_str, "0.0 seconds"

# Main app
def main():
    st.markdown('<div class="header"><h1>Speech Recognition System</h1></div>', unsafe_allow_html=True)
    st.markdown("Select a Speech-to-Text model and record/upload audio.", unsafe_allow_html=True)
    st.markdown("**Note**: Ensure microphone permissions are enabled in your browser (Chrome/Firefox recommended).", unsafe_allow_html=True)

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
                help="Tiny or small recommended for Indian accents. Large-v2 is disabled.",
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

        st.markdown("**Select Microphone (for debugging; default mic used)**")
        device_id_en_hi = audio_device_selector("en_hi")
        if device_id_en_hi:
            st.session_state.device_en_hi = device_id_en_hi
            st.markdown(f'<div class="status-box">Selected device ID: {device_id_en_hi}</div>', unsafe_allow_html=True)

        st.markdown("**Record Audio**")
        audio_input_en_hi = audio_recorder(
            key="audio_en_hi",
            energy_threshold=100,
            pause_threshold=0.5
        )
        if st.button("Retry Recording", key="retry_en_hi"):
            st.session_state.retry_count_en_hi = 0
            st.experimental_rerun()
        
        uploaded_file_en_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_en_hi")
        
        if st.button("Transcribe with Speech2Text", key="transcribe_en_hi"):
            audio = audio_input_en_hi or uploaded_file_en_hi
            if audio:
                transcription, duration, proc_time = transcribe_english_hindi(audio, language_selection)
                st.session_state.transcription_en_hi = transcription
                st.session_state.duration_en_hi = duration
                st.session_state.proc_time_en_hi = proc_time
                if "too quiet" in transcription.lower():
                    st.markdown('<div class="warning-box">Warning: Audio is too quiet. Increase microphone gain or speak louder.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">No audio input provided. Please record or upload an audio file.</div>', unsafe_allow_html=True)
        
        st.text_area("Transcription", st.session_state.transcription_en_hi, height=150, disabled=True, key="transcription_en_hi")
        col3, col4 = st.columns(2)
        with col3:
            st.text_input("Audio Duration", st.session_state.duration_en_hi, disabled=True, key="duration_en_hi")
        with col4:
            st.text_input("Processing Time", st.session_state.proc_time_en_hi, disabled=True, key="proc_time_en_hi")
        if st.session_state.debug_info and audio_input_en_hi:
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

        st.markdown("**Select Microphone (for debugging; default used)**")
        device_id_hi = audio_device_selector("hi")
        if device_id_hi:
            st.session_state.device_hi = device_id_hi
            st.markdown(f'<div class="status-box">Selected device ID: {device_id_hi}</div>', unsafe_allow_html=True)

        st.markdown("Record Audio")
        audio_input_hi = audio_recorder(
            key="audio_hi",
            energy_threshold=100,
            pause_threshold=0.5
        )
        if st.button("Retry Recording", key="retry_hi"):
            st.session_state.retry_count_hi = 0
            st.experimental_rerun()
        
        uploaded_file_hi = st.file_uploader("Or upload an audio file", type=["wav", "mp3"], key="upload_hi")
        
        if st.button("Transcribe with Speech2Text", key="transcribe_hi"):
            audio = audio_input_hi or uploaded_file_hi
            if audio:
                transcription, duration, proc_time = transcribe_hindi_only(audio)
                st.session_state.transcription_hi = transcription
                st.session_state.duration_hi = duration
                st.session_state.proc_time_hi = proc_time
                if "too quiet" in transcription.lower():
                    st.markdown('<div class="warning-box">Warning: Audio is too quiet. Increase microphone gain or speak louder.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">No audio input provided. Please record or upload an audio file.</div>', unsafe_allow_html=True)
        
        st.text_area("Transcription", st.session_state.transcription_hi, height=150, disabled=True, key="transcription_hi")
        col5, col6 = st.columns(2)
        with col5:
            st.text_input("Audio Duration", st.session_state.duration_hi, disabled=True, key="duration_hi")
        with col6:
            st.text_input("Processing Time", st.session_state.proc_time_hi, disabled=True, key="proc_time_hi")
        if st.session_state.debug_info and audio_input_hi:
            st.markdown(f'<div class="warning-box">Debug: {st.session_state.debug_info}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
