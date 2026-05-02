import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# Load trained model
model = joblib.load("two_hand_words_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=15)

print("Starting two-hand word detection...")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    features = []

    if result.multi_hand_landmarks:
        # Sort hands left → right
        hand_landmarks = sorted(
            result.multi_hand_landmarks,
            key=lambda h: h.landmark[0].x
        )

        for hand in hand_landmarks:
            for lm in hand.landmark:
                features.extend([lm.x, lm.y])

        # If only one hand detected → pad zeros
        while len(features) < 84:
            features.extend([0.0, 0.0])

        if len(features) == 84:
            X = np.array(features).reshape(1, -1)
            prediction = model.predict(X)[0]

            prediction_buffer.append(prediction)
            stable_prediction = max(set(prediction_buffer), key=prediction_buffer.count)

            cv2.putText(
                frame,
                f"Prediction: {stable_prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Two-Hand Word Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
