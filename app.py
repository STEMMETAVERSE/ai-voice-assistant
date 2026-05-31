import streamlit as st
import tempfile
import base64
import os

from transformers import pipeline
from huggingface_hub import InferenceClient

# ==================================================
# CONFIG
# ==================================================

HF_TOKEN = st.secrets["HF_TOKEN"]

client = InferenceClient(token=HF_TOKEN)

# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base"
    )

asr = load_asr()

# ==================================================
# AUDIO PLAYER
# ==================================================

def autoplay_audio(audio_bytes):

    b64 = base64.b64encode(audio_bytes).decode()

    md = f"""
    <audio controls autoplay>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """

    st.markdown(md, unsafe_allow_html=True)

# ==================================================
# UI
# ==================================================

st.title("🎤 Voice AI Assistant")

st.info(
    """
    Voice → Whisper → Llama → Response

    Speak into your microphone and receive an AI answer.
    """
)

audio = st.audio_input("Record your voice")

# ==================================================
# PROCESS AUDIO
# ==================================================

if audio is not None:

    st.success("Audio received.")

    # Save uploaded audio to temp file
    audio_bytes = audio.read()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(audio_bytes)
        temp_audio_path = tmp.name

    try:

        # ==========================================
        # SPEECH TO TEXT
        # ==========================================

        with st.spinner("Transcribing speech..."):

            result = asr(temp_audio_path)

            user_text = result["text"]

        st.subheader("📝 You Said")

        st.write(user_text)

        # ==========================================
        # LLM RESPONSE
        # ==========================================

        with st.spinner("Thinking..."):

            response = client.chat.completions.create(
                model="meta-llama/Llama-3.2-1B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful voice assistant. "
                            "Keep responses concise and clear."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ],
                max_tokens=200,
                temperature=0.7,
                top_p=0.9
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

        st.subheader("🤖 AI Response")

        st.write(answer)

        # ==========================================
        # OPTIONAL TEXT TO SPEECH
        # ==========================================

        st.info(
            "Text-to-speech disabled for maximum Streamlit compatibility."
        )

        # If you later add a TTS model,
        # play audio using:
        #
        # autoplay_audio(tts_audio)

    except Exception as e:

        st.error(f"Error: {e}")

    finally:

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
