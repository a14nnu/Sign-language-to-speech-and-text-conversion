import cv2
import mediapipe as mp
import numpy as np
import pygame
import os
import math
from gtts import gTTS
from flask import Flask, render_template, Response
import webbrowser
from threading import Timer

app = Flask(__name__)

# --- 1. SETUP AUDIO ---
def play_voice(text):
    if not os.path.exists("sounds"):
        os.makedirs("sounds")
    file_path = f"sounds/{text.replace(' ', '_')}.mp3"
    if not os.path.exists(file_path):
        tts = gTTS(text=text, lang='en')
        tts.save(file_path)
    
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    if not pygame.mixer.music.get_busy(): 
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# --- 2. MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def gen_frames():
    cap = cv2.VideoCapture(0)
    last_label, label_counter = "", 0
    STABILITY_THRESHOLD = 20 

    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        detected_label = "SEARCHING..."

        if results.multi_hand_landmarks:
            all_hands = results.multi_hand_landmarks
            
            # Refined finger detection for both hands
            def get_fingers(lm, hand_type):
                # Thumb logic changes based on Left vs Right hand
                if hand_type == "Right":
                    thumb = 1 if lm[4].x < lm[3].x else 0
                else: # Left hand
                    thumb = 1 if lm[4].x > lm[3].x else 0
                
                fingers = [thumb]
                for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                    fingers.append(1 if lm[tip].y < lm[pip].y else 0)
                return fingers

            if len(all_hands) == 1:
                lm = all_hands[0].landmark
                hand_info = results.multi_handedness[0].classification[0].label
                f = get_fingers(lm, hand_info)
                
                # --- 15 GESTURES LOGIC ---
                # Distinguish HELLO (High) vs PLEASE (Lower/Chest level)
                if f == [1, 1, 1, 1, 1]:
                    if lm[9].y < 0.4: # Hand is in upper part of frame
                        detected_label = "HELLO"
                    else: # Hand is lower down (near chest)
                        detected_label = "PLEASE"
                
                elif f == [0, 1, 1, 1, 1]: detected_label = "THANK YOU"
                elif f == [0, 1, 1, 0, 0]: detected_label = "NO"
                elif f == [1, 1, 0, 0, 1]: detected_label = "I LOVE YOU"
                elif f == [1, 0, 0, 0, 1]: detected_label = "PLAY"
                elif f == [0, 1, 1, 1, 0]: detected_label = "WIN"
                elif f == [0, 0, 0, 0, 1]: detected_label = "SORRY"
                elif sum(f) == 0: detected_label = "YES"
                elif f == [1, 0, 0, 0, 0]: detected_label = "GOOD"
                elif get_distance(lm[4], lm[8]) < 0.05 and f[2:] == [1, 1, 1]: detected_label = "OK"
                elif f == [0, 1, 0, 0, 0]: detected_label = "WATER"

            elif len(all_hands) == 2:
                h1, h2 = all_hands[0].landmark, all_hands[1].landmark
                if get_distance(h1[8], h2[0]) < 0.08 or get_distance(h2[8], h1[0]) < 0.08: detected_label = "TIME"
                elif get_distance(h1[8], h2[8]) < 0.05: detected_label = "HOUSE"
                else:
                    h1_y, h2_y = h1[9].y, h2[9].y
                    top, bot = (h1, h2) if h1_y < h2_y else (h2, h1)
                    if sum(get_fingers(top, "Any")) == 0 and sum(get_fingers(bot, "Any")) >= 4: detected_label = "HELP"

            for hand_lms in all_hands:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

        # --- STABILITY LOGIC & VISUAL BAR ---
        if detected_label == last_label and detected_label != "SEARCHING...":
            label_counter += 1
            # Draw Stability Bar (Green progress bar)
            bar_width = int((label_counter / STABILITY_THRESHOLD) * 200)
            cv2.rectangle(img, (50, 70), (250, 90), (200, 200, 200), -1) # Background
            cv2.rectangle(img, (50, 70), (50 + bar_width, 90), (0, 255, 0), -1) # Progress
            
            if label_counter >= STABILITY_THRESHOLD:
                play_voice(detected_label)
                label_counter = 0
        else:
            last_label, label_counter = detected_label, 0

        cv2.putText(img, detected_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(2, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)