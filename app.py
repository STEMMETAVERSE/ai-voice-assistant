import streamlit as st
from huggingface_hub import InferenceClient

client = InferenceClient()

audio = st.audio_input("Record")

if audio:

    audio_bytes = audio.read()

    result = client.automatic_speech_recognition(
        audio_bytes,
        model="openai/whisper-small"
    )

    st.write(type(result))
    st.write(result)
