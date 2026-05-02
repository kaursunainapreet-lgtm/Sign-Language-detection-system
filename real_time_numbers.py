import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque

model = joblib.load("number_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
history = deque(maxlen=7)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            features = []
            for lm in hand.landmark:
                features.extend([lm.x, lm.y])

            pred = model.predict([features])[0]
            history.append(pred)

            stable = max(set(history), key=history.count)
            cv2.putText(frame, f"Number: {stable}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Number Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
