import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

# Load trained model
model = joblib.load("sign_language_model.pkl")

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Prediction stability variables
prediction_queue = deque(maxlen=8)
stable_prediction = ""

# Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            features = []

            # Extract x and y only (42 features)
            for lm in hand_landmarks.landmark:
                features.append(lm.x)
                features.append(lm.y)

            # Convert to numpy array
            features = np.array(features).reshape(1, -1)

            # Prediction
            prediction = model.predict(features)[0]
            prediction_queue.append(prediction)

            # Check stability
            if prediction_queue.count(prediction) == len(prediction_queue):
                stable_prediction = prediction

    # Display stable prediction
    cv2.putText(
        frame,
        f"Prediction: {stable_prediction}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Sign Language Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
