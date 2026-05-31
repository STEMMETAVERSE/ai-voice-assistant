import streamlit as st
from huggingface_hub import InferenceClient
import base64

# =========================
# CONFIG
# =========================

HF_TOKEN = st.secrets["HF_TOKEN"]
client = InferenceClient(token=HF_TOKEN)

st.title("🎤 Voice AI Assistant (Full Pipeline)")

st.info(
    """
    Speak → AI understands → AI responds with voice + text
    """
)

# =========================
# AUDIO INPUT
# =========================

audio = st.audio_input("🎙️ Record your voice")

# =========================
# TEXT-TO-SPEECH FUNCTION
# =========================

def autoplay_audio(audio_bytes):

    b64 = base64.b64encode(audio_bytes).decode()

    audio_html = f"""
    <audio autoplay controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)

# =========================
# PIPELINE
# =========================

if audio:

    st.success("Audio received ✔")

    # Save file
    audio_bytes = audio.read()

    with open("input.wav", "wb") as f:
        f.write(audio_bytes)

    # =========================
    # 1. SPEECH TO TEXT
    # =========================

    st.spinner("Transcribing speech...")

    with open("input.wav", "rb") as f:

        transcript = client.automatic_speech_recognition(
            file=f,
            model="openai/whisper-small"
        )

    user_text = transcript["text"]

    st.subheader("📝 You said:")
    st.write(user_text)

    # =========================
    # 2. LLM RESPONSE
    # =========================

    st.spinner("Thinking...")

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Keep answers short and clear."
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        max_tokens=200,
        temperature=0.6
    )

    ai_text = response.choices[0].message.content

    st.subheader("🤖 AI Response")
    st.write(ai_text)

    # =========================
    # 3. TEXT TO SPEECH
    # =========================

    st.spinner("Generating voice response...")

    try:

        tts_audio = client.text_to_speech(
            text=ai_text,
            model="espnet/kan-bayashi_ljspeech_vits"
        )

        st.subheader("🔊 AI Voice Output")

        autoplay_audio(tts_audio)

    except Exception as e:

        st.warning(
            "Text-to-speech not available on this model endpoint."
        )
        st.caption(str(e))
