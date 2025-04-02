import cv2
import torch
import numpy as np
import os
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from keras.models import load_model
import moviepy.editor as mp

app = Flask(__name__)
CORS(app)

# Load Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("models/best.pt")  # Ensure best.pt is in the directory
audio_model = load_model("models/ambulance_siren_model.h5")  # Siren model

SAMPLE_RATE = 22050  # Audio sampling rate

def detect_ambulance(video_path):
    cap = cv2.VideoCapture(video_path)
    results_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        # YOLO Object Detection
        results = yolo_model.predict(frame, conf=0.65, device=device)
        
        detections = []
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                confidence = result.boxes.conf[0].item() * 100
                if confidence > 65:
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": confidence
                    })

        if detections:
            results_list.append({"frame": cap.get(cv2.CAP_PROP_POS_FRAMES), "detections": detections})

    cap.release()
    return results_list

def detect_siren(video_path):
    try:
        video = mp.VideoFileClip(video_path)
        temp_audio_path = "temp_audio.wav"
        video.audio.write_audiofile(temp_audio_path, fps=SAMPLE_RATE)
        
        y, sr = librosa.load(temp_audio_path, sr=SAMPLE_RATE)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = np.expand_dims(log_mel_spec, axis=-1)
        
        prediction = audio_model.predict(np.expand_dims(log_mel_spec, axis=0))
        return prediction[0][0] > 0.5

    except Exception as e:
        return False

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['video']
    if file:
        video_path = os.path.join("uploads", file.filename)
        file.save(video_path)

        # Detect Ambulance
        detections = detect_ambulance(video_path)

        # Detect Siren
        siren_detected = detect_siren(video_path)

        return jsonify({
            "ambulance_detections": detections,
            "siren_detected": siren_detected
        })

    return jsonify({"error": "No video uploaded"}), 400

if __name__ == '__main__':
    app.run(debug=True)
