import cv2
import mediapipe as mp
import csv
import os

# CSV file to save alphabet data
data_file = "alphabet_data.csv"
if not os.path.exists(data_file):
    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

# Open camera
cam = cv2.VideoCapture(0)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

print("Camera opened. Click the camera window to focus, then show your hand and press A–Z keys to save data.")

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect landmarks
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list += [lm.x, lm.y, lm.z]

            # Check key press
            key = cv2.waitKey(1) & 0xFF
            if 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
                label = chr(key).upper()
                with open(data_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + landmark_list)
                print(f"Saved data for {label}")

    # Show camera feed
    cv2.imshow("Alphabet Data Collection", frame)
