import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# ===== DIRECT MODEL PATHS (NO CONFUSION) =====
BASE_PATH = r"C:\Users\Sunainapreet Kaur\OneDrive\Desktop\SignLanguage"

alphabet_model = joblib.load(os.path.join(BASE_PATH, "alphabet_model.pkl"))
number_model   = joblib.load(os.path.join(BASE_PATH, "number_model.pkl"))
word_model     = joblib.load(os.path.join(BASE_PATH, "word_model.pkl"))
twohand_model  = joblib.load(os.path.join(BASE_PATH, "two_hand_word.pkl"))

# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

mode = None
prediction_text = "Press A / N / W / T"

print("A: Alphabets | N: Numbers | W: Words | T: Two-hand Words | Q: Quit")

# ===== FEATURE EXTRACTION =====
def extract_features(results, two_hand=False):
    features = []

    if not results.multi_hand_landmarks:
        return None

    hands_used = results.multi_hand_landmarks[:2] if two_hand else results.multi_hand_landmarks[:1]

    for hand in hands_used:
        for lm in hand.landmark:
            features.append(lm.x)
            features.append(lm.y)

    if two_hand and len(hands_used) < 2:
        return None  # require both hands

    return np.array(features).reshape(1, -1)

# ===== MAIN LOOP =====
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if mode == "A":
            feats = extract_features(results, two_hand=False)
            if feats is not None:
                prediction_text = alphabet_model.predict(feats)[0]

        elif mode == "N":
            feats = extract_features(results, two_hand=False)
            if feats is not None:
                prediction_text = number_model.predict(feats)[0]

        elif mode == "W":
            feats = extract_features(results, two_hand=False)
            if feats is not None:
                prediction_text = word_model.predict(feats)[0]

        elif mode == "T":
            feats = extract_features(results, two_hand=True)
            if feats is not None:
                prediction_text = twohand_model.predict(feats)[0]

    cv2.putText(frame, f"Mode: {mode}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(frame, f"Prediction: {prediction_text}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Sign Language System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        mode = "A"
        prediction_text = "Alphabet Mode"

    elif key == ord('n'):
        mode = "N"
        prediction_text = "Number Mode"

    elif key == ord('w'):
        mode = "W"
        prediction_text = "Word Mode"

    elif key == ord('t'):
        mode = "T"
        prediction_text = "Two-Hand Word Mode"

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
