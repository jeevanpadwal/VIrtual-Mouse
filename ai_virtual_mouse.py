import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Set up webcam
cap = cv2.VideoCapture(0)

# Initialize variables for tracking hand movement
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
default_smoothening = 7  # Default smoothening factor
current_smoothening = default_smoothening
hand_speed = 0
prev_hand_pos = (0, 0)
prev_hand_time = time.time()

# Mouse control modes
MOUSE_MODE_NORMAL = 0
MOUSE_MODE_PRECISE = 1
MOUSE_MODE_FAST = 2
current_mode = MOUSE_MODE_NORMAL
mode_names = ["Normal", "Precise", "Fast"]
mode_switch_time = 0
mode_cooldown = 1.5  # Time in seconds to prevent accidental mode switches

# Initialize variables for click and scroll detection
pinch_start_time = 0
click_performed = False
scroll_active = False
prev_distance = 0

# Frame rate variables
frame_counter = 0
start_time = time.time()
fps = 0

# Mouse control area rectangle
rect_start_x = 100
rect_start_y = 100
rect_width = 400
rect_height = 300

# Precision mode settings
precision_smoothening = 12  # Higher for more precise movement
precision_scale = 0.5  # Scale factor for movement in precision mode

# Fast mode settings
fast_smoothening = 3  # Lower for faster response
fast_scale = 1.5  # Scale factor for movement in fast mode

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points"""
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def map_coordinates(hand_x, hand_y, scale_factor=1.0):
    """Map hand coordinates to screen coordinates with scaling"""
    # Calculate relative position in rectangle
    x_ratio = (hand_x - rect_start_x) / rect_width
    y_ratio = (hand_y - rect_start_y) / rect_height
    
    # Apply scaling factor based on mode
    if scale_factor != 1.0:
        # Scale around the center
        x_ratio = 0.5 + (x_ratio - 0.5) * scale_factor
        y_ratio = 0.5 + (y_ratio - 0.5) * scale_factor
    
    # Map to screen coordinates
    screen_x = int(x_ratio * screen_width)
    screen_y = int(y_ratio * screen_height)
    
    # Keep coordinates within screen bounds
    screen_x = max(0, min(screen_width, screen_x))
    screen_y = max(0, min(screen_height, screen_y))
    
    return screen_x, screen_y

def check_inside_rectangle(x, y):
    """Check if a point is inside the control rectangle"""
    return (rect_start_x <= x <= rect_start_x + rect_width and 
            rect_start_y <= y <= rect_start_y + rect_height)

def get_finger_positions(hand_landmarks, image_shape):
    """Extract finger positions from hand landmarks"""
    positions = {}
    
    # Get height and width of the image
    h, w, _ = image_shape
    
    # Index fingertip (for cursor movement)
    index_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))
    
    # Thumb tip (for click detection)
    thumb_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w),
                int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h))
    
    # Middle fingertip (for scroll detection)
    middle_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * w),
                 int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * h))
    
    # Ring fingertip (for right click)
    ring_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * w),
               int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * h))
    
    # Pinky tip (for drag and drop)
    pinky_tip = (int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * w),
                int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * h))
    
    # Index knuckle
    index_knuckle = (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * w),
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * h))
    
    # Wrist
    wrist = (int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w),
            int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h))
    
    # Store all positions
    positions['index_tip'] = index_tip
    positions['thumb_tip'] = thumb_tip
    positions['middle_tip'] = middle_tip
    positions['ring_tip'] = ring_tip
    positions['pinky_tip'] = pinky_tip
    positions['index_knuckle'] = index_knuckle
    positions['wrist'] = wrist
    
    return positions

def calculate_hand_speed(current_pos, prev_pos, time_elapsed):
    """Calculate hand movement speed in pixels per second"""
    if time_elapsed == 0:
        return 0
    distance = calculate_distance(current_pos, prev_pos)
    return distance / time_elapsed

print("Enhanced AI Virtual Mouse System Starting...")
print("Press 'q' to quit")
print("Hold up pinky finger to toggle between mouse modes")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flip the frame horizontally for a more intuitive view
    frame = cv2.flip(frame, 1)
    
    # Convert BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Draw the control rectangle
    cv2.rectangle(frame, (rect_start_x, rect_start_y), 
                  (rect_start_x + rect_width, rect_start_y + rect_height), 
                  (255, 0, 0), 2)
    
    # Calculate and display FPS
    frame_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_counter / elapsed_time
        frame_counter = 0
        start_time = time.time()
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Mode: {mode_names[current_mode]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw instructions on the frame
    cv2.putText(frame, "Index finger: Move cursor", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Thumb + Index: Left click", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Thumb + Ring: Right click", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Index + Middle: Scroll", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Pinky up: Toggle mouse mode", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Process hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Get finger positions
            finger_positions = get_finger_positions(hand_landmarks, frame.shape)
            
            # Handle mode switch with pinky finger
            pinky_up = finger_positions['pinky_tip'][1] < finger_positions['index_knuckle'][1] - 50
            current_time = time.time()
            
            # Toggle mode if pinky finger is up and cooldown time has passed
            if pinky_up and (current_time - mode_switch_time > mode_cooldown):
                current_mode = (current_mode + 1) % 3
                mode_switch_time = current_time
                print(f"Switched to {mode_names[current_mode]} mode")
            
            # Calculate hand speed
            current_time = time.time()
            time_elapsed = current_time - prev_hand_time
            hand_speed = calculate_hand_speed(finger_positions['index_tip'], prev_hand_pos, time_elapsed)
            prev_hand_pos = finger_positions['index_tip']
            prev_hand_time = current_time
            
            # Adjust smoothening and scaling based on mode
            if current_mode == MOUSE_MODE_NORMAL:
                current_smoothening = default_smoothening
                scale_factor = 1.0
            elif current_mode == MOUSE_MODE_PRECISE:
                current_smoothening = precision_smoothening
                scale_factor = precision_scale
            else:  # MOUSE_MODE_FAST
                current_smoothening = fast_smoothening
                scale_factor = fast_scale
            
            # Check if index finger is inside the control rectangle
            if check_inside_rectangle(*finger_positions['index_tip']):
                # Move cursor based on index finger position with appropriate scaling
                mapped_x, mapped_y = map_coordinates(*finger_positions['index_tip'], scale_factor)
                
                # Apply smoothening to cursor movement
                curr_x = prev_x + (mapped_x - prev_x) / current_smoothening
                curr_y = prev_y + (mapped_y - prev_y) / current_smoothening
                
                # Move the cursor
                pyautogui.moveTo(curr_x, curr_y)
                
                # Update previous coordinates
                prev_x, prev_y = curr_x, curr_y
                
                # Detect click (thumb and index finger pinch)
                thumb_index_distance = calculate_distance(finger_positions['thumb_tip'], finger_positions['index_tip'])
                
                # Left click (thumb and index finger)
                if thumb_index_distance < 30:
                    if not click_performed:
                        pyautogui.click()
                        click_performed = True
                        cv2.putText(frame, "Left Click!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    click_performed = False
                
                # Right click (thumb and ring finger)
                thumb_ring_distance = calculate_distance(finger_positions['thumb_tip'], finger_positions['ring_tip'])
                if thumb_ring_distance < 30:
                    pyautogui.rightClick()
                    cv2.putText(frame, "Right Click!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Scroll detection (index and middle finger)
                index_middle_distance = calculate_distance(finger_positions['index_tip'], finger_positions['middle_tip'])
                if index_middle_distance < 30:
                    if not scroll_active:
                        scroll_active = True
                        prev_distance = finger_positions['wrist'][1] - finger_positions['index_tip'][1]
                    else:
                        current_distance = finger_positions['wrist'][1] - finger_positions['index_tip'][1]
                        if abs(current_distance - prev_distance) > 5:
                            # Scroll up or down
                            if current_distance > prev_distance:
                                pyautogui.scroll(-5)  # Scroll down
                            else:
                                pyautogui.scroll(5)   # Scroll up
                            prev_distance = current_distance
                else:
                    scroll_active = False
    
    # Display mode information
    mode_color = (0, 255, 0) if current_mode == MOUSE_MODE_NORMAL else \
                 (255, 0, 0) if current_mode == MOUSE_MODE_PRECISE else \
                 (0, 0, 255)  # Fast mode
    cv2.putText(frame, f"Mode: {mode_names[current_mode]}", (screen_width - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    # Display the frame
    cv2.imshow('Enhanced AI Virtual Mouse', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Enhanced AI Virtual Mouse System Terminated")