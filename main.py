import cv2
import mediapipe as mp
import numpy as np
import os
from gesture_recognizer import GestureRecognizer
from segmentation_processor import SegmentationProcessor

# --- Configuration ---
CAM_WIDTH, CAM_HEIGHT = 1280, 720  # Use a higher resolution for better quality/visibility
BACKGROUNDS_DIR = 'Backgrounds'

# Placeholder for the Live Camera Background
LIVE_BG_PLACEHOLDER = 'LIVE_FEED'


# --- Utility Functions for UI ---

def load_background_images():
    """Loads and resizes background images from the directory. Inserts LIVE_FEED placeholder at index 0."""
    images = []

    # 1. Add the Live Feed Placeholder first
    images.append(LIVE_BG_PLACEHOLDER)

    # 2. Load custom images
    if os.path.isdir(BACKGROUNDS_DIR):
        for file_name in sorted(os.listdir(BACKGROUNDS_DIR)):  # Sorted for consistent indexing
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                bg_path = os.path.join(BACKGROUNDS_DIR, file_name)
                bg_img = cv2.imread(bg_path)
                if bg_img is not None:
                    # Resize the image to match the camera frame size
                    bg_img_resized = cv2.resize(bg_img, (CAM_WIDTH, CAM_HEIGHT))
                    images.append(bg_img_resized)

    # Fallback: If only the placeholder is present, add a solid color fallback
    if len(images) == 1:
        print(f"WARNING: No custom backgrounds found in '{BACKGROUNDS_DIR}'. Adding solid blue fallback.")
        solid_color_bg = np.full((CAM_HEIGHT, CAM_WIDTH, 3), (255, 100, 0), dtype=np.uint8)  # Orange/Blue
        images.append(solid_color_bg)

    return images


def draw_ui(frame, backgrounds, current_index, final_active_bg_index):
    """Draws the background selection UI with < and > indicators."""

    THUMB_WIDTH, THUMB_HEIGHT = 100, 70
    PADDING = 15
    UI_HEIGHT = THUMB_HEIGHT + 2 * PADDING
    BUTTON_WIDTH = 30

    # Custom backgrounds start from index 1.
    num_custom_bgs = len(backgrounds) - 1

    # --- UI Area Setup  ---
    y1 = frame.shape[0] - UI_HEIGHT
    y2 = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (frame.shape[1], y2), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Determine which 3 custom thumbnails to display centered around the current_index
    # The current_index (if > 0) is the background currently selected in the UI.

    # Calculate the index of the first custom background to display.
    # This keeps the current_index (the UI focus) in the center slot (i=1).
    if num_custom_bgs <= 3:
        start_display_index = 1  # Show all if 3 or fewer
    else:
        # start_display_index should be max(1, current_index - 1). This ensures
        # we don't try to show index 0 (LIVE_FEED) and keeps the focus centered.
        start_display_index = max(1, min(current_index - 1, num_custom_bgs - 2))

        # Calculate total width for 2 buttons, 3 thumbnails, and padding
    TOTAL_UI_WIDTH = (3 * THUMB_WIDTH) + (2 * PADDING) + (2 * BUTTON_WIDTH) + (
                4 * PADDING)  # Added more padding for aesthetics
    start_x = int((frame.shape[1] // 2) - (TOTAL_UI_WIDTH // 2))

    current_x = start_x
    y_thumb = y1 + PADDING

    # --- 1. Draw LEFT Button (<) ---
    cv2.rectangle(frame, (current_x, y_thumb), (current_x + BUTTON_WIDTH, y_thumb + THUMB_HEIGHT), (70, 70, 70), -1)
    cv2.putText(frame, '<', (current_x + 8, y_thumb + THUMB_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    current_x += BUTTON_WIDTH + PADDING

    # --- 2. Draw 3 Thumbnails ---
    for i in range(3):
        # The index in the full list (which starts at 1 for custom backgrounds)
        idx = start_display_index + i

        # Only draw if the index is a valid custom background
        if idx < len(backgrounds):
            bg_img = backgrounds[idx]
            thumb = cv2.resize(bg_img, (THUMB_WIDTH, THUMB_HEIGHT))

            x1_thumb = current_x
            x2_thumb = x1_thumb + THUMB_WIDTH

            # Place the thumbnail onto the frame
            frame[y_thumb: y_thumb + THUMB_HEIGHT, x1_thumb: x2_thumb] = thumb

            border_color = (255, 255, 255)  # White default
            thickness = 2

            # Highlight the currently scrolled (UI focus) item
            if idx == current_index:
                border_color = (0, 255, 255)  # Yellow
                thickness = 4

            # Highlight the actively selected item (if different)
            if idx == final_active_bg_index:
                border_color = (0, 255, 0)  # Bright Green
                thickness = 5

            cv2.rectangle(frame, (x1_thumb, y_thumb), (x2_thumb, y_thumb + THUMB_HEIGHT), border_color, thickness)

        current_x += THUMB_WIDTH + PADDING

    # --- 3. Draw RIGHT Button (>) ---
    cv2.rectangle(frame, (current_x, y_thumb), (current_x + BUTTON_WIDTH, y_thumb + THUMB_HEIGHT), (70, 70, 70), -1)
    cv2.putText(frame, '>', (current_x + 8, y_thumb + THUMB_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame


# --- 4. Main Application ---

def main():
    background_images = load_background_images()
    num_custom_bgs = len(background_images) - 1  # Number of images excluding LIVE_FEED

    # current_bg_index: The background index currently in focus in the UI (1 to num_custom_bgs)
    # final_active_bg_index: The background index actively overlaid on the screen (0 for LIVE_FEED, 1+ for custom)
    current_bg_index = 1  # Start UI focus on the first custom background
    final_active_bg_index = 0  # Start with the LIVE_FEED active

    # Initialize the handlers
    gesture_recognizer = GestureRecognizer(swipe_threshold=70)
    segmentation_processor = SegmentationProcessor()

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_segmentation = mp.solutions.selfie_segmentation

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # Use 'with' to ensure the models are properly released
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands, \
            mp_segmentation.SelfieSegmentation(
            model_selection=1) as selfie_segmentation:

        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            frame = cv2.flip(frame, 1)  # Flip for selfie-view
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Process frames
            segmentation_results = selfie_segmentation.process(image_rgb)
            hand_results = hands.process(image_rgb)

            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # --- Gesture Handling ---
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]

                # A. Handle Swipe Gesture (Scrolling)
                swipe = gesture_recognizer.detect_swipe(hand_landmarks, CAM_WIDTH, CAM_HEIGHT)

                # Scrolling is only active when a custom background has been selected (i.e., UI is active)
                if final_active_bg_index > 0:
                    if swipe == 'LEFT':
                        # Cycle from 1 to num_custom_bgs
                        current_bg_index = (current_bg_index % num_custom_bgs) + 1
                        print(f"SWIPE LEFT: New UI Index: {current_bg_index}")
                    elif swipe == 'RIGHT':
                        # Cycle backwards
                        current_bg_index = ((current_bg_index - 2 + num_custom_bgs) % num_custom_bgs) + 1
                        print(f"SWIPE RIGHT: New UI Index: {current_bg_index}")

                # B. Handle Selection Gesture
                select = gesture_recognizer.detect_selection(hand_landmarks, CAM_WIDTH, CAM_HEIGHT)
                if select == 'SELECT':
                    if final_active_bg_index == 0:
                        # Case 1: LIVE_FEED is active. SELECT toggles UI ON and selects the current UI focus.
                        final_active_bg_index = current_bg_index
                        print(f"SELECT: Toggled UI ON. Active BG: {final_active_bg_index}")
                    else:
                        # Case 2: Custom BG is active. SELECT confirms the current UI focus as the active background.
                        final_active_bg_index = current_bg_index
                        print(f"SELECT: Confirmed Active BG: {final_active_bg_index}")

                # Draw Landmarks (for visualization and debugging)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- Image Processing ---
            output_image = image.copy()  # Start with the raw camera frame

            # Check if we need to apply segmentation (i.e., if final_active_bg_index is NOT 0)
            if final_active_bg_index > 0 and segmentation_results.segmentation_mask is not None:
                # Get the actual image object from the list
                active_background = background_images[final_active_bg_index]

                output_image = segmentation_processor.process_frame(
                    image,
                    segmentation_results.segmentation_mask,
                    active_background
                )

            # --- UI Display ---
            if final_active_bg_index > 0:  # Only show UI if a custom background is active
                output_image = draw_ui(output_image, background_images, current_bg_index, final_active_bg_index)

            # --- Final Display ---
            cv2.imshow('Hand Gesture Background Changer', output_image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()