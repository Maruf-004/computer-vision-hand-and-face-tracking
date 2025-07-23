import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start video capture (change 1 to 0 if you want built-in webcam)
cap = cv2.VideoCapture(1)

# Optional: try to set webcam brightness and exposure (may or may not work depending on camera)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural interaction
    frame = cv2.flip(frame, 1)

    # Adjust brightness and contrast
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 30    # Brightness control (0-100)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Draw landmarks on hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get handedness
            handedness = result.multi_handedness[idx].classification[0].label
            is_right_hand = handedness == "Right"

            finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
            finger_base = [2, 6, 10, 14, 18]  # Corresponding lower joints
            count = 0

            # Thumb detection depends on hand side
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            thumb_mcp = hand_landmarks.landmark[2]

            if is_right_hand:
                if thumb_tip.x < thumb_ip.x < thumb_mcp.x:
                    count += 1
            else:  # Left hand
                if thumb_tip.x > thumb_ip.x > thumb_mcp.x:
                    count += 1

            # Check other four fingers using Y-coordinates (tip above base means finger up)
            for tip_idx, base_idx in zip(finger_tips[1:], finger_base[1:]):
                if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[base_idx].y - 0.02:
                    count += 1

            # Display the number of fingers for each hand
            cv2.putText(frame, f'Hand {idx+1} ({handedness}): {count}', (10, 30 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
