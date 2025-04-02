import streamlit as st
import requests

st.title("🚑 Ambulance Detection System")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    st.video("temp_video.mp4")

    if st.button("Detect Ambulance"):
        with st.spinner("Processing..."):
            response = requests.post("http://127.0.0.1:5000/detect", files={"video": open("temp_video.mp4", "rb")})

        if response.status_code == 200:
            data = response.json()
            st.write("Ambulance Detections:", data["ambulance_detections"])
            if data["siren_detected"]:
                st.success("🚨 Siren Detected!")
            else:
                st.warning("🚑 Off-duty Ambulance (No Siren)")
