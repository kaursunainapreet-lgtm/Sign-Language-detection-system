import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

file_name = "number_data_fixed.csv"
file_exists = os.path.isfile(file_name)

cap = cv2.VideoCapture(0)

print("Press keys 0–9 to save that number")
print("Press Q to quit")

with open(file_name, "a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}"]
        header.append("label")
        writer.writerow(header)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y])

                key = cv2.waitKey(1) & 0xFF

                if key in range(ord('0'), ord('9') + 1):
                    label = chr(key)
                    writer.writerow(landmarks + [label])
                    print(f"Saved number {label}")

        cv2.imshow("Collect Numbers", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
