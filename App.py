import streamlit as st
from tensorflow.keras.models import load_model
import os
import time
from joblib import load
import numpy as np
import librosa

# Load the pre-trained model and scaler
scaler = load('D:/Project/FinalY/Webapp/scaler.joblib')
model = load('emotion_model.joblib')

# Page Configuration
st.set_page_config(page_title="Stress Detection", page_icon="🎧", layout="wide")

CAT = ["unstressed", "neutral", "stressed"]

# Function to save uploaded audio file
def save_audio(file):
    if file.size > 4000000:
        return 1  # File size too large
    folder = "audio"
    if not os.path.exists(folder):
        os.makedirs(folder)

    filepath = os.path.join(folder, file.name)
    with open(filepath, "wb") as f:
        f.write(file.getbuffer())
    return filepath

# Feature extraction
def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

# Function to display color-coded results
def display_state(predicted_state):
    if "Unstressed" in predicted_state:
        st.success(f"### 🎉 {predicted_state}")
        st.markdown("😊 You seem relaxed. Keep up the good work!")
    elif "Stressed" in predicted_state:
        st.error(f"### ⚠️ {predicted_state}")
        st.warning("⚠️ **Take care! Here are some tips to reduce stress:**")
        st.markdown("""
        - 🧘 **Practice mindfulness** or meditation.\n
        - 🎶 **Listen to calming music**.\n
        - 🌳 **Take a short walk** and get fresh air.\n
        - 😴 **Ensure you're getting enough sleep**.\n
        - 💧 **Stay hydrated** and take regular breaks.\n
        """)
    elif "Neutral" in predicted_state:
        st.info(f"### 😐 {predicted_state}")
        st.markdown("You're doing fine. Stay balanced and keep an eye on your well-being!")

# Main app
def main():
    st.title("🎧 Audio-Based Stress Detection Web App")
    st.markdown("### Upload your audio file to analyze stress levels!")

    # Initialize session state variables
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    if "previous_states" not in st.session_state:
        st.session_state.previous_states = []  # To store previous predictions

    # File uploader with dynamic key
    audio_file = st.file_uploader("Upload your audio (WAV format only)", type=["wav"], 
                                  key=f"audio_uploader_{st.session_state.file_uploader_key}")

    if audio_file:
        st.audio(audio_file, format="audio/wav", start_time=0)

        # Save audio and validate file size
        st.markdown("### Processing File...")
        filepath = save_audio(audio_file)
        if filepath == 1:
            st.warning("⚠️ File size too large. Please upload a smaller file.")
            return

        # Add spinner while extracting features and predicting
        with st.spinner("🔍 Analyzing audio..."):
            time.sleep(1.5)  # Simulating a short delay for better UX
            data, sample_rate = librosa.load(filepath, duration=2.5, offset=0.6)
            features = extract_features(data, sample_rate)
            features = scaler.transform([features])
            time.sleep(1)  # Another delay for progress effect

        # Progress bar to simulate real-time analysis
        st.markdown("### Analyzing...")
        progress_bar = st.progress(0)
        for percent in range(100):
            time.sleep(0.02)
            progress_bar.progress(percent + 1)

        # Make prediction
        prediction = model.predict(features)
        state_mapping = {
            "positive": "Unstressed 😊",
            "negative": "Stressed ⚠️",
            "neutral": "Neutral 😐"
        }
        predicted_state = state_mapping.get(prediction[0], "Unknown")

        # Store the current prediction and file name in previous states
        if not st.session_state.previous_states or st.session_state.previous_states[-1][0] != audio_file.name:
            st.session_state.previous_states.append((audio_file.name, predicted_state))


        # Display state with color coding
        # Add a header before displaying the predicted state
        st.markdown("---")  # Add a horizontal line for separation
        st.subheader("🎯 Predicted State")
        display_state(predicted_state)


        # "Try Another File" button
        if st.button("🔄 Try Another File"):
            st.session_state.file_uploader_key += 1  # Increment the key
            st.rerun()

    # Show previous states
    if st.button("📋 Show Previous States"):
        st.markdown("## Previous Predictions:")
        if st.session_state.previous_states:
            for idx, (filename, state) in enumerate(st.session_state.previous_states, 1):
                st.write(f"{idx}. **File:** {filename} → **Prediction:** {state}")
        else:
            st.info("No previous predictions available.")

if __name__ == '__main__':
    main()

