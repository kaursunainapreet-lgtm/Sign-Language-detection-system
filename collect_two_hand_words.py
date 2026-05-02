import cv2
import csv
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

LABEL = input("Enter word label (e.g. WATER): ").upper()
SAMPLES = 120
count = 0

with open("two_hand_words_raw.csv", "a", newline="") as f:
    writer = csv.writer(f)

    while count < SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            row = []

            # Sort hands left → right
            hands_sorted = sorted(
                results.multi_hand_landmarks,
                key=lambda h: sum([lm.x for lm in h.landmark])
            )

            for hand in hands_sorted:
                for lm in hand.landmark:
                    row.extend([lm.x, lm.y])

            row.append(LABEL)
            writer.writerow(row)
            count += 1

            cv2.putText(frame, f"Saved: {count}/{SAMPLES}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.imshow("Collect TWO HAND Words", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
