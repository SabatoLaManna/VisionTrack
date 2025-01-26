import cv2
import mediapipe as mp
import math

# Toggles
SHOW_NUMBERS = True  # Toggle to show/hide landmark numbers for hand
SHOW_CIRCLES = True  # Toggle to show/hide circles on hand landmarks
SHOW_WIRES = True    # Toggle to show/hide connecting wires for hand
SHOW_FACE = True     # Enable facial landmarks and expression detection

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)  # Increased confidence
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt((coord1.x - coord2.x) ** 2 + 
                     (coord1.y - coord2.y) ** 2 + 
                     (coord1.z - coord2.z) ** 2)

def detect_smile(face_landmarks):
    """Detect if the face is smiling by comparing the distance between mouth corners."""
    left_corner = face_landmarks.landmark[61]  # Left corner of the mouth
    right_corner = face_landmarks.landmark[291]  # Right corner of the mouth
    mouth_top = face_landmarks.landmark[13]  # Top of the upper lip

    # Calculate distance between corners of the mouth
    mouth_width = calculate_distance(left_corner, right_corner)
    mouth_top_to_bottom = calculate_distance(mouth_top, face_landmarks.landmark[14])  # Bottom of the lower lip

    # If the mouth is stretched wide and upper lip is not too far from the lower lip, it's a smile
    return mouth_width > mouth_top_to_bottom * 2.5

def detect_frown(face_landmarks):
    """Detect if the face is frowning by analyzing the distance between the eyebrows."""
    left_eyebrow = face_landmarks.landmark[70]  # Left eyebrow
    right_eyebrow = face_landmarks.landmark[300]  # Right eyebrow

    # Check if eyebrows are close together
    distance_between_eyebrows = calculate_distance(left_eyebrow, right_eyebrow)
    return distance_between_eyebrows < 0.1  # Threshold to indicate frown

def detect_open_mouth(face_landmarks):
    """Detect if the mouth is open based on the vertical distance between upper and lower lips."""
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    mouth_open_distance = calculate_distance(upper_lip, lower_lip)
    return mouth_open_distance > 0.05  # Threshold to indicate mouth is open

def detect_closed_eyes(face_landmarks):
    """Detect if eyes are closed by comparing the aspect ratio of the eyes."""
    # Left eye: landmarks 33-133, Right eye: landmarks 362-263
    left_eye_top = face_landmarks.landmark[159]  # Top of left eye
    left_eye_bottom = face_landmarks.landmark[145]  # Bottom of left eye
    left_eye_left = face_landmarks.landmark[130]  # Left side of left eye
    left_eye_right = face_landmarks.landmark[133]  # Right side of left eye

    right_eye_top = face_landmarks.landmark[386]  # Top of right eye
    right_eye_bottom = face_landmarks.landmark[374]  # Bottom of right eye
    right_eye_left = face_landmarks.landmark[362]  # Left side of right eye
    right_eye_right = face_landmarks.landmark[263]  # Right side of right eye

    # Calculate the height and width of the eyes
    left_eye_height = calculate_distance(left_eye_top, left_eye_bottom)
    left_eye_width = calculate_distance(left_eye_left, left_eye_right)

    right_eye_height = calculate_distance(right_eye_top, right_eye_bottom)
    right_eye_width = calculate_distance(right_eye_left, right_eye_right)

    # Eye aspect ratio (EAR) formula to check if the eyes are closed
    left_eye_ratio = left_eye_height / left_eye_width
    right_eye_ratio = right_eye_height / right_eye_width

    return left_eye_ratio < 0.2 and right_eye_ratio < 0.2

def detect_neutral_expression(face_landmarks):
    """Detect if the expression is neutral."""
    # If no strong features like smile or frown are detected, it's neutral
    if not detect_smile(face_landmarks) and not detect_frown(face_landmarks) and not detect_open_mouth(face_landmarks) and not detect_closed_eyes(face_landmarks):
        return True
    return False

# Hand gesture detection functions
def detect_five(hand_landmarks):
    """Detect if the hand is open (5 fingers extended)."""
    finger_tips = [4, 8, 12, 16, 20]
    wrist = hand_landmarks.landmark[0]
    extended_fingers = []

    for tip in finger_tips:
        finger_tip = hand_landmarks.landmark[tip]
        finger_base = hand_landmarks.landmark[tip - 2] if tip != 4 else hand_landmarks.landmark[0]
        extended = calculate_distance(finger_tip, wrist) > calculate_distance(finger_base, wrist)
        extended_fingers.append(extended)

    return all(extended_fingers)

def detect_fist(hand_landmarks):
    """Detect if the hand is in a fist position."""
    finger_tips = [8, 12, 16, 20]
    wrist = hand_landmarks.landmark[0]
    fist_condition = all(
        calculate_distance(hand_landmarks.landmark[tip], wrist) < calculate_distance(hand_landmarks.landmark[tip - 2], wrist)
        for tip in finger_tips
    )
    return fist_condition

def detect_pinching(hand_landmarks):
    """Detect if the hand is pinching by comparing the distance between thumb and index finger."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    return calculate_distance(thumb_tip, index_tip) < 0.05  # Threshold for pinching

def detect_middle_finger(hand_landmarks):
    """Detect if the middle finger is extended while other fingers are curled."""
    middle_finger_tip = hand_landmarks.landmark[12]
    wrist = hand_landmarks.landmark[0]
    return calculate_distance(middle_finger_tip, wrist) > 0.2  # Middle finger extended if far enough

def detect_peace_sign(hand_landmarks):
    # Assuming the peace sign is detected when index and middle fingers are up and others are down
    index_finger_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_finger_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_finger_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_finger_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    thumb_down = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x

    return index_finger_up and middle_finger_up and ring_finger_down and pinky_finger_down and thumb_down
def detect_thumbs_up(hand_landmarks):
    # Assuming the thumbs up is detected when thumb is up and other fingers are down
    thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y
    index_finger_down = hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y
    middle_finger_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_finger_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_finger_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y

    return thumb_up and index_finger_down and middle_finger_down and ring_finger_down and pinky_finger_down

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand tracking and facial landmark detection
    result_hands = hands.process(rgb_frame)
    result_faces = face_mesh.process(rgb_frame)

    # Initialize a container for hand landmarks
    hand_landmark_positions = []

    # Draw hand landmarks and wireframe if hands are detected
    if result_hands.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(result_hands.multi_hand_landmarks):
            if SHOW_WIRES:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                       mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            if SHOW_NUMBERS:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Save hand landmarks for position printing
            landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
            hand_landmark_positions.append({'hand_index': hand_index, 'landmarks': landmarks})

            # Detect gestures and place labels near the hand
            hand_center = hand_landmarks.landmark[0]
            hand_x, hand_y = int(hand_center.x * frame.shape[1]), int(hand_center.y * frame.shape[0])

            if detect_five(hand_landmarks):
                cv2.putText(frame, "FIVE", (hand_x + 10, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif detect_fist(hand_landmarks):
                cv2.putText(frame, "FIST", (hand_x + 10, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif detect_pinching(hand_landmarks):
                cv2.putText(frame, "PINCHING", (hand_x + 10, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif detect_thumbs_up(hand_landmarks):
                cv2.putText(frame, "THUMBS UP", (hand_x + 10, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif detect_peace_sign(hand_landmarks):
                cv2.putText(frame, "PEACE SIGN", (hand_x + 10, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            elif detect_middle_finger(hand_landmarks):
                cv2.putText(frame, "MIDDLE FINGER", (hand_x + 10, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Print hand landmark positions when 'p' is pressed
    if cv2.waitKey(1) & 0xFF == ord('p'):
        for hand in hand_landmark_positions:
            print(f"Hand {hand['hand_index']} Landmarks:")
            for index, landmark in enumerate(hand['landmarks']):
                print(f"  Landmark {index}: x={landmark['x']:.4f}, y={landmark['y']:.4f}, z={landmark['z']:.4f}")
            print("-" * 50)

    # Display the frame
    cv2.imshow('VisionTrack - Coord finder (p)', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
