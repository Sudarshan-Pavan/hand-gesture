import cv2
import mediapipe as mp
import math
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def get_palm_center(landmarks):
    indices = [0, 1, 5, 9, 13, 17]
    x = sum(landmarks[i].x for i in indices) / len(indices)
    y = sum(landmarks[i].y for i in indices) / len(indices)
    z = sum(landmarks[i].z for i in indices) / len(indices)
    return x, y, z

def euclidean_distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2 +
        (p1.z - p2.z)**2
    )

# Finger state function with handedness awareness
def get_finger_states(hand_landmarks, hand_label):
    finger_states = []

    # Compute palm center
    indices = [0, 1, 5, 9, 13, 17]
    center = hand_landmarks.landmark
    cx = sum(center[i].x for i in indices) / len(indices)
    cy = sum(center[i].y for i in indices) / len(indices)
    cz = sum(center[i].z for i in indices) / len(indices)

    # Create a landmark-like object for palm center
    class Point:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
    palm_center = Point(cx, cy, cz)

    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ref = hand_landmarks.landmark[2]
    dist_tip = euclidean_distance(thumb_tip, palm_center)
    dist_ref = euclidean_distance(thumb_ref, palm_center)

    thumb_open = dist_tip > dist_ref

    # Correction: if tip and base are on opposite sides of palm center in X-axis, consider thumb closed
    if (thumb_tip.x < cx and thumb_ref.x > cx) or (thumb_tip.x > cx and thumb_ref.x < cx):
        thumb_open = False

    finger_states.append(1 if thumb_open else 0)

    # Index, Middle, Ring, Pinky
    finger_tips = [8, 12, 16, 20]
    finger_refs = [6, 10, 14, 18]
    for tip_idx, ref_idx in zip(finger_tips, finger_refs):
        tip = hand_landmarks.landmark[tip_idx]
        ref = hand_landmarks.landmark[ref_idx]
        dist_tip = euclidean_distance(tip, palm_center)
        dist_ref = euclidean_distance(ref, palm_center)
        finger_states.append(1 if dist_tip > dist_ref else 0)

    return finger_states

def match_gesture(finger_states):
    gesture_dict = {
        (0, 0, 0, 0, 0): "FIST",
        (1, 1, 1, 1, 1): "PALM",
        (0, 1, 1, 0, 0): "PEACE",
        (1, 0, 0, 0, 0): "THUMBS UP",
        (0, 1, 0, 0, 0): "POINT",  
        (1, 1, 0, 0, 0): "POINT",
        (0, 0, 1, 0, 0): "SHUT DOWN",
        (1, 0, 1, 0, 0): "****",
        (0, 1, 0, 0, 1): "ROCK",
        (1, 1, 0, 0, 1): "ROCK",
        (0, 1, 1, 1, 0): "THREE",
        (0, 0, 1, 1, 1): "THREE",
        (0, 1, 1, 1, 1): "FOUR",
        (0, 1, 1, 0, 1): "BABA",
        (1, 0, 0, 0, 1): "CALL ME",
        #(0, 0, 0, 0, 0): "SHUT DOWN"
    }

    return gesture_dict.get(tuple(finger_states), None)

def shutdown_system():
    # Windows shutdown command (change /s to /r if you want restart)
    os.system("shutdown /s /t 1")  # shuts down after 5 seconds


# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Horizontal flip
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'

            # Draw the hand and connections
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape
            cx, cy, cz = get_palm_center(hand_landmarks.landmark)
            cv2.circle(img, (int(cx * w), int(cy * h)), 5, (0, 0, 255), -1)


            # Get finger states
            finger_states = get_finger_states(hand_landmarks, label)
            print(f"{label} hand | Finger States: {finger_states}")

            gesture = match_gesture(finger_states)
            if gesture == "FIST":
                shutdown_system()
    
            #Optional: display states on screen
            cv2.putText(img, f"Gesture Detected: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
