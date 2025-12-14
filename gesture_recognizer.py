import mediapipe as mp


class GestureRecognizer:
    def __init__(self, swipe_threshold=70, select_threshold=0.8):
        # Threshold for horizontal movement (in pixels) to register a swipe
        self.swipe_threshold = swipe_threshold
        # Threshold for how close the index finger must be to the camera (normalized Z-coord) for a selection
        self.select_threshold = select_threshold
        # Store the previous x-coordinate of the index finger tip for swipe detection
        self.previous_x = None
        # MediaPipe Hand Landmark IDs for the tips of the fingers
        self.index_tip_id = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
        self.middle_tip_id = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
        self.ring_tip_id = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
        self.pinky_tip_id = mp.solutions.hands.HandLandmark.PINKY_TIP
        self.wrist_id = mp.solutions.hands.HandLandmark.WRIST

    def _get_landmark_coords(self, hand_landmarks, width, height):
        """Extracts normalized and pixel coordinates of key landmarks."""
        if not hand_landmarks:
            return None, None

        index_tip = hand_landmarks.landmark[self.index_tip_id]

        # Convert normalized coordinates (0 to 1) to pixel coordinates
        pixel_x = int(index_tip.x * width)
        pixel_y = int(index_tip.y * height)
        pixel_z = index_tip.z  # Normalized depth (Z)

        # A set of key landmarks for generalized state checks
        key_landmarks = {
            'index_tip_y': hand_landmarks.landmark[self.index_tip_id].y,
            'middle_tip_y': hand_landmarks.landmark[self.middle_tip_id].y,
            'ring_tip_y': hand_landmarks.landmark[self.ring_tip_id].y,
            'pinky_tip_y': hand_landmarks.landmark[self.pinky_tip_id].y,
            'wrist_y': hand_landmarks.landmark[self.wrist_id].y
        }

        return (pixel_x, pixel_y, pixel_z), key_landmarks

    def detect_swipe(self, hand_landmarks, frame_width, frame_height):
        """
        Detects left or right swipe based on the index finger movement.
        Returns 'LEFT', 'RIGHT', or None.
        """
        coords, _ = self._get_landmark_coords(hand_landmarks, frame_width, frame_height)
        if not coords:
            self.previous_x = None
            return None

        current_x = coords[0]

        if self.previous_x is None:
            self.previous_x = current_x
            return None

        # Calculate the movement
        delta_x = current_x - self.previous_x

        gesture = None
        if delta_x > self.swipe_threshold:
            gesture = 'RIGHT'
        elif delta_x < -self.swipe_threshold:
            gesture = 'LEFT'

        # Update the previous position for the next frame
        if gesture:
            # Reset previous_x after a detection to prepare for the next distinct swipe
            self.previous_x = current_x
        else:
            # Smooth the tracking by updating every frame when not swiping
            self.previous_x = current_x

        return gesture

    def detect_selection(self, hand_landmarks, frame_width, frame_height):
        """
        Detects a selection gesture: usually a single finger pointing up.
        Returns 'SELECT' or None.
        """
        coords, key_lms = self._get_landmark_coords(hand_landmarks, frame_width, frame_height)
        if not coords:
            return None

        # Logic for "Index Finger Up" (index tip must be significantly higher (lower Y value) than other tips)
        index_y = key_lms['index_tip_y']

        # Check if the index finger tip is above (lower Y) the other three finger tips AND the wrist
        is_index_up = (index_y < key_lms['middle_tip_y']) and \
                      (index_y < key_lms['ring_tip_y']) and \
                      (index_y < key_lms['pinky_tip_y']) and \
                      (index_y < key_lms['wrist_y'])

        if is_index_up:
            return 'SELECT'

        return None