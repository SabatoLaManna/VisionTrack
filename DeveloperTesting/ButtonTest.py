import cv2
import mediapipe as mp
import math

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_draw = mp.solutions.drawing_utils

# Parameters
button_height = 50
button_width = 200
button_position = (320, 50)  # Position of the button (centered on the top)
circle_radius = 20
circle_position = None  # This will hold the position of the circle
is_grabbing = False
circle_attached = False
circle_created = False  # Flag to keep track of circle creation

# Function to calculate the Euclidean distance between two points
def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# Function to check if a fist is made
def is_fist(hand_landmarks):
    # Check if all fingers are curled (except thumb)
    # Thumb is at position 4 (index), the rest are at positions 8 (index), 12 (middle), 16 (ring), 20 (pinky)
    
    # Check if the y-coordinate of the fingertips is lower than the base of the finger (MCP joint)
    # These positions are 2, 6, 10, 14, and 18
    fist_condition = True
    for i in [8, 12, 16, 20]:  # Check for all fingers except thumb
        if hand_landmarks.landmark[i].y > hand_landmarks.landmark[i - 2].y:
            fist_condition = False
            break
    return fist_condition

# Open the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand tracking
    result = hands.process(rgb_frame)

    # Draw button
    cv2.rectangle(frame, (button_position[0] - button_width // 2, button_position[1] - button_height // 2),
                  (button_position[0] + button_width // 2, button_position[1] + button_height // 2), (0, 255, 0), 2)
    cv2.putText(frame, "Button", (button_position[0] - 40, button_position[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2)

    # Detect hand landmarks and check for fist gesture
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index and thumb tip coordinates
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Convert landmarks to pixel coordinates
            h, w, _ = frame.shape
            index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if either index or thumb is in the button area
            if (button_position[0] - button_width // 2 < index_tip_coords[0] < button_position[0] + button_width // 2 and
                button_position[1] - button_height // 2 < index_tip_coords[1] < button_position[1] + button_height // 2):
                
                # Create the circle at the center of the frame if it doesn't exist yet
                if not circle_created:
                    circle_position = (w // 2, h // 2)  # Initialize at the center
                    circle_created = True  # Circle is now created

                # Draw circle if it's not attached
                if not circle_attached and circle_position:
                    cv2.circle(frame, circle_position, circle_radius, (0, 0, 255), -1)  # Red circle

                # Check for fist gesture
                if is_fist(hand_landmarks):
                    # If fist is made, select the circle
                    if not is_grabbing:
                        is_grabbing = True
                        circle_attached = True
                else:
                    # Release the circle if the fist is no longer made
                    if is_grabbing:
                        is_grabbing = False
                        circle_attached = False

            # Move the circle with the thumb when grabbing
            if is_grabbing and circle_attached and circle_position:
                circle_position = thumb_tip_coords
                cv2.circle(frame, circle_position, circle_radius, (0, 0, 255), -1)  # Red circle

            elif circle_created and circle_position:  # Always keep the circle on screen if it's created
                cv2.circle(frame, circle_position, circle_radius, (0, 0, 255), -1)  # Red circle

    # Display the frame
    cv2.imshow('Hand Tracking with Button and Circle', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
