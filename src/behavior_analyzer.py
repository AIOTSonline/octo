import cv2
import numpy as np
import tensorflow as tf

import pandas as pd
import os
from collections import deque # For efficient history management in Track class

# --- Configuration ---
# Path to your TensorFlow SavedModel directory
SAVED_MODEL_PATH = '/home/pranav/coding/Arishna Internship/runs/detect/train6/weights/best_saved_model'
# Path to your video file
VIDEO_PATH = '/home/pranav/coding/Arishna Internship/data/204841--av-1.mp4'

# Detection parameters (adapted from realtime_octopus_tflite_detector.py)
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score for a detection to be considered valid
NMS_IOU_THRESHOLD = 0.45    # IoU threshold for Non-Maximum Suppression (NMS) to filter redundant boxes

# Tracking parameters (for our custom IoU tracker)
TRACKING_IOU_THRESHOLD = 0.3  # IoU threshold for matching new detections to existing tracks
MAX_MISS_COUNT = 5            # Number of consecutive frames a track can be missed before it's deleted

OCTOPUS_CLASS_ID = 0          # Assuming octopus is class 0 in your model's output
MODEL_INPUT_SIZE = (640, 640) # (width, height) expected by the SavedModel (YOLOv8 default)
CLASSES = {0: 'octopus'}      # Dictionary mapping class IDs to names (only one class here)

# Display configuration for the output window
DISPLAY_MAX_WIDTH = 1280      # Maximum width for the display window. Adjust as needed.

# Behavior classification thresholds
BRIGHTNESS_THRESHOLD = 130  # V channel threshold for "light"
DARKNESS_THRESHOLD = 70    # V channel threshold for "dark"
SPEED_THRESHOLD_LOW = 2.0  # Pixels per frame
SPEED_THRESHOLD_HIGH = 10.0  # Pixels per frame

# --- Simple Track Class for Custom IoU Tracker ---
class Track:
    """Represents a single tracked object (e.g., an octopus)."""
    next_id = 0  # Class variable to assign unique, incremental IDs to new tracks

    def __init__(self, bbox, confidence):
        """
        Initializes a new track with its initial bounding box and confidence.
        Args:
            bbox (list): Bounding box in [x1, y1, x2, y2] format.
            confidence (float): Confidence score of the initial detection.
        """
        self.bbox = bbox  # Current bounding box of the track
        self.track_id = Track.next_id
        Track.next_id += 1  # Increment for the next new track
        self.confidence = confidence
        self.frames_since_last_detection = 0  # Counter for consecutive frames without a new detection
        # Use deque for efficient history of bounding box centers
        self.center_history = deque([self._get_center(bbox)], maxlen=10)
        self.average_hsv = (0, 0, 0)  # Average (Hue, Saturation, Value)
        self.speed = 0.0  # Instantaneous speed in pixels per frame
        self.speed_history = deque(maxlen=15) # History of speeds for averaging (e.g., 0.5 seconds at 30fps)
        # History for stabilizing behavior classification
        self.state_history = deque(maxlen=30)  # Stores last 30 classifications (~1 sec at 30fps)
        self.dominant_behavior = "pending"  # The most common behavior in the recent history
        self.previous_behavior = "pending"

    def _get_center(self, bbox):
        """Calculates the center of a bounding box."""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, new_bbox, new_confidence):
        """
        Updates the track with a new, matched detection.
        Args:
            new_bbox (list): New bounding box for the track from the current frame.
            new_confidence (float): Confidence of the new detection.
        """
        # Calculate speed before updating history
        if len(self.center_history) > 1:
            prev_center = self.center_history[-1]
            curr_center = self._get_center(new_bbox)
            # Euclidean distance
            self.speed = np.sqrt(
                (curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            self.speed_history.append(self.speed) # Add current speed to history

        self.bbox = new_bbox
        self.confidence = new_confidence
        self.frames_since_last_detection = 0  # Reset miss count on successful update
        self.center_history.append(self._get_center(new_bbox))

    def mark_missed(self):
        """Increments the miss counter when no new detection is found for this track in the current frame."""
        self.frames_since_last_detection += 1
        self.speed = 0.0  # Reset speed if track is missed

# --- Utility Function: Intersection Over Union (IoU) ---
def calculate_iou(box1, box2):
    """Calculates Intersection Over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format (top-left and bottom-right coordinates).
    Returns a float between 0.0 and 1.0.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # If there is no overlap (intersection area is zero or negative), return 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area (sum of individual areas minus intersection area)
    union_area = float(box1_area + box2_area - intersection_area)

    # Avoid division by zero if union_area is 0 (e.g., one or both boxes have zero area)
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def analyze_color(octopus_crop):
    """
    Analyzes the average color of an octopus crop in HSV space.
    Args:
        octopus_crop (np.array): The cropped image of the octopus in BGR format.
    Returns:
        tuple: (average_hue, average_saturation, average_value)
    """
    if octopus_crop is None or octopus_crop.size == 0:
        return (0, 0, 0)

    # Convert the cropped image to HSV color space
    hsv_crop = cv2.cvtColor(octopus_crop, cv2.COLOR_BGR2HSV)

    # Calculate the average Hue, Saturation, and Value
    # Note: Hue is circular (0-179 in OpenCV), so taking a direct mean can be misleading
    # if colors span across the red boundary (e.g., magenta and red).
    # For a simple start, a direct mean is acceptable.
    avg_hsv = cv2.mean(hsv_crop)

    # avg_hsv will be a tuple of 4 values (H, S, V, alpha if present), so we take the first 3.
    return (int(avg_hsv[0]), int(avg_hsv[1]), int(avg_hsv[2]))

def get_brightness_state(average_hsv):
    """
    Determines if the octopus is 'light' or 'dark' based on its average HSV Value.
    Args:
        average_hsv (tuple): (hue, saturation, value) tuple.
    Returns:
        str: "light" or "dark" based on value threshold.
    """
    _, _, val = average_hsv
    if val > BRIGHTNESS_THRESHOLD:
        return "light"
    else:
        return "dark"


def classify_behavior(track):
    """
    Classifies octopus behavior based on movement speed.
    Args:
        track (Track): The track object containing octopus data.
    Returns:
        str: The classified movement behavior state.
    """
    # Use average speed from history for more stable classification
    # If speed_history is empty (e.g., first frame), default to 0.0
    speed = np.mean(track.speed_history) if track.speed_history else 0.0

    if speed > SPEED_THRESHOLD_HIGH:
        return "exploring"
    elif speed < SPEED_THRESHOLD_LOW:
        return "calm"
    else:
        return "observing"


# --- Helper Functions adapted from realtime_octopus_tflite_detector.py ---
def non_max_suppression(boxes, scores, iou_threshold):
    """
    Performs Non-Maximum Suppression (NMS) on the bounding boxes.
    Assumes boxes are in (x1, y1, x2, y2) format.
    Returns indices of the boxes to keep after NMS.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep

def preprocess_frame(frame, input_size):
    """
    Preprocesses a single video frame for model inference.
    Resizes, converts BGR to RGB, adds batch dimension, and sets data type.
    """
    # Resize to model input size (width, height)
    resized_frame = cv2.resize(frame, input_size, interpolation=cv2.INTER_AREA)
    # Convert BGR (OpenCV default) to RGB (common for models)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Expand dimensions to create a batch of 1: (H, W, C) -> (1, H, W, C)
    input_tensor = np.expand_dims(rgb_frame, axis=0)
    # Convert data type to float32 and normalize to 0-1 range
    input_tensor = input_tensor.astype(np.float32) / 255.0
    return input_tensor

def postprocess_output(output_data, original_frame_shape, model_input_size, conf_threshold, iou_threshold):
    """
    Post-processes the raw model output to extract and filter detections.
    Assumes output is in a dictionary format from a SavedModel, with shape (1, 5, 8400).
    """
    # Standard SavedModel output from TF is often a dictionary. Let's get the tensor.
    # The key is 'output_0' based on our debug print.
    output_tensor = output_data['output_0']

    # Squeeze the batch dimension and transpose: (1, 5, 8400) -> (5, 8400) -> (8400, 5)
    # This reorients the data so we can iterate over each of the 8400 potential detections.
    predictions = np.squeeze(output_tensor).T

    boxes = []
    scores = []
    class_ids = []

    original_h, original_w, _ = original_frame_shape

    for pred in predictions:
        # Assuming pred format: [x_center, y_center, width, height, confidence_score]
        conf = pred[4]

        if conf >= conf_threshold:
            x_center, y_center, box_width, box_height = pred[0], pred[1], pred[2], pred[3]

            # The model output coordinates are normalized to the [0, 1] range.
            # We scale them to the original frame's dimensions.
            x1 = int((x_center - box_width / 2) * original_w)
            y1 = int((y_center - box_height / 2) * original_h)
            x2 = int((x_center + box_width / 2) * original_w)
            y2 = int((y_center + box_height / 2) * original_h)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))
            class_ids.append(OCTOPUS_CLASS_ID)

    if boxes:
        keep_indices = non_max_suppression(boxes, scores, iou_threshold)
        final_boxes = [boxes[i] for i in keep_indices]
        final_scores = [scores[i] for i in keep_indices]
    else:
        final_boxes, final_scores = [], []

    return final_boxes, final_scores

# --- Main Logic ---
def main():
    # --- Check paths and load model ---
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: SavedModel directory not found at {SAVED_MODEL_PATH}")
        return

    try:
        print("Loading TensorFlow SavedModel... This may take a moment.")
        model = tf.saved_model.load(SAVED_MODEL_PATH)


        if model is None:
            print("Error: TensorFlow model failed to load and is None.")
            return

        print("TensorFlow SavedModel loaded successfully. GPU should be utilized if available.")

        # Check if the model has the expected 'serving_default' signature
        if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
            infer = model.signatures['serving_default']
            print("Using 'serving_default' signature for inference.")
            # Also check if the signature has structured_outputs before printing
            if hasattr(infer, 'structured_outputs'):
                print("Model output signature:", infer.structured_outputs)
        # As a fallback, check if the model itself is callable
        elif callable(model):
            infer = model
            print("Warning: 'serving_default' signature not found. Falling back to using the model object directly.")
        # If neither, we can't proceed
        else:
            print("Error: Model is not a SavedModel with a 'serving_default' signature and is not directly callable.")
            return
    except Exception as e:
        print(f"Error loading SavedModel: {e}")
        return

    # --- Open video stream ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    print(f"Processing video: {VIDEO_PATH}")

    # --- Initialize Custom IoU Tracker State ---
    active_tracks = []
    Track.next_id = 0

    # --- Initialize Logging ---
    log_data = []



    print("\n--- Starting Octopus Detection and Tracking with TensorFlow SavedModel (GPU) ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        input_tensor = preprocess_frame(frame, MODEL_INPUT_SIZE)
        input_tensor_tf = tf.convert_to_tensor(input_tensor)
        output_data = infer(input_tensor_tf)
        nms_filtered_boxes, nms_filtered_scores = postprocess_output(
            output_data,
            frame.shape,
            MODEL_INPUT_SIZE,
            CONFIDENCE_THRESHOLD,
            NMS_IOU_THRESHOLD
        )

        current_detections_for_tracker = []
        for i in range(len(nms_filtered_boxes)):
            box = nms_filtered_boxes[i]
            score = nms_filtered_scores[i]
            current_detections_for_tracker.append([box[0], box[1], box[2], box[3], score])

        # --- Custom IoU-based Tracking Algorithm ---
        matched_detections_indices = [False] * len(current_detections_for_tracker)
        matched_tracks_indices = [False] * len(active_tracks)

        for i, track in enumerate(active_tracks):
            best_match_iou = 0.0
            best_detection_idx = -1
            for j, detection_data in enumerate(current_detections_for_tracker):
                if matched_detections_indices[j]:
                    continue
                det_bbox = detection_data[0:4]
                iou = calculate_iou(track.bbox, det_bbox)
                if iou > best_match_iou and iou >= TRACKING_IOU_THRESHOLD:
                    best_match_iou = iou
                    best_detection_idx = j
            if best_detection_idx != -1:
                det_bbox = current_detections_for_tracker[best_detection_idx][0:4]
                det_confidence = current_detections_for_tracker[best_detection_idx][4]
                track.update(det_bbox, det_confidence)
                matched_detections_indices[best_detection_idx] = True
                matched_tracks_indices[i] = True

        tracks_to_remove = []
        for i, track in enumerate(active_tracks):
            if not matched_tracks_indices[i]:
                track.mark_missed()
                if track.frames_since_last_detection > MAX_MISS_COUNT:
                    tracks_to_remove.append(i)
        for i in sorted(tracks_to_remove, reverse=True):
            del active_tracks[i]

        for j, detection_data in enumerate(current_detections_for_tracker):
            if not matched_detections_indices[j]:
                det_bbox = detection_data[0:4]
                det_confidence = detection_data[4]
                new_track = Track(det_bbox, det_confidence)
                active_tracks.append(new_track)

        # --- Behavior Analysis and Drawing ---
        for track in active_tracks:
            # --- Get BBox and Crop for Analysis ---
            x1, y1, x2, y2 = track.bbox
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(frame.shape[1], x2), min(frame.shape[0], y2)
            octopus_crop = frame[y1_c:y2_c, x1_c:x2_c]

            # --- Analyze Features and Classify Behavior ---
            track.average_hsv = analyze_color(octopus_crop)

            # Get individual classifications
            movement_state = classify_behavior(track)
            brightness_state = get_brightness_state(track.average_hsv)

            # Combine brightness and movement to determine behavior state
            if brightness_state == "dark" and movement_state == "calm":
                behavior_state = "aggressive"
            elif brightness_state == "light" and movement_state == "exploring":
                behavior_state = "alert"
            else:
                behavior_state = movement_state  # Default to movement state

            # Add the classified behavior state to the history deque for stabilization
            track.state_history.append(behavior_state)

            # Determine the most common (dominant) movement behavior
            if track.state_history:
                track.dominant_behavior = max(set(track.state_history), key=list(track.state_history).count)
            else:
                track.dominant_behavior = "pending"

            # --- Log Behavior State Changes ---
            if track.dominant_behavior != track.previous_behavior:
                # Get video timestamp in milliseconds
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                # Format to HH:MM:SS.ms
                seconds = int(msec / 1000)
                milliseconds = int(msec % 1000)
                minutes = int(seconds / 60)
                hours = int(minutes / 60)
                timestamp = f"{hours:02d}:{minutes % 60:02d}:{seconds % 60:02d}.{milliseconds:03d}"
                log_entry = {
                    'timestamp': timestamp,
                    'track_id': track.track_id,
                    'previous_state': track.previous_behavior,
                    'new_state': track.dominant_behavior
                }
                log_data.append(log_entry)
                track.previous_behavior = track.dominant_behavior # Update previous behavior

            # --- Drawing Logic ---
            # Set color based on the dominant behavior state for stable visualization
            state_color = (0, 255, 255)  # Yellow for default/observing
            if track.dominant_behavior == "calm":
                state_color = (255, 255, 0)  # Cyan
            elif track.dominant_behavior == "exploring":
                state_color = (255, 0, 255)  # Magenta
            elif track.dominant_behavior == "aggressive":
                state_color = (0, 0, 255)  # Red
            elif track.dominant_behavior == "alert":
                state_color = (0, 165, 255)  # Orange
            elif track.dominant_behavior == "aggressive":
                state_color = (0, 0, 255)  # Red
            elif track.dominant_behavior == "alert":
                state_color = (0, 165, 255)  # Orange

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare text for display using the stabilized dominant behavior
            track_id_text = f"ID: {track.track_id}"
            behavior_text = f"State: {track.dominant_behavior}"
            speed_text = f"Speed: {track.speed:.1f}px/f"

            cv2.putText(frame, behavior_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2, cv2.LINE_AA)
            cv2.putText(frame, speed_text, (x1, y1 - 35),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, track_id_text, (x1, y1 - 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Resize frame for display ---
        display_height, display_width, _ = frame.shape
        if display_width > DISPLAY_MAX_WIDTH:
            scale = DISPLAY_MAX_WIDTH / display_width
            display_width = int(display_width * scale)
            display_height = int(display_height * scale)
            frame = cv2.resize(frame, (display_width, display_height))

        cv2.imshow("Octopus Behavior Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Save Log to Excel ---
    if log_data:
        try:
            log_df = pd.DataFrame(log_data)
            output_path = "behavior_log.xlsx"
            log_df.to_excel(output_path, index=False, engine="openpyxl")
            print(f"\nBehavior log saved to {output_path}")
        except ImportError:
            print("\n[Warning] Could not save log. To enable logging to Excel, please install pandas and openpyxl:")
            print("pip install pandas openpyxl")
        except Exception as e:
            print(f"\nAn error occurred while saving the log: {e}")



    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished.")

if __name__ == '__main__':
    main()
