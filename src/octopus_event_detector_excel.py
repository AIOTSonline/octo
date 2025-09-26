import cv2
import numpy as np
import pandas as pd
from config import DetectionConfig

class OctopusEventDetector:
    def __init__(self, video_path, output_file=None, config=None):
        self.video_path = video_path
        self.config = config or DetectionConfig()
        self.output_file = output_file or self.config.DEFAULT_OUTPUT_FILE
        self.events = []

        # Initialize detection parameters from config
        self.color_history = []
        self.color_change_threshold = self.config.COLOR_CHANGE_THRESHOLD
        self.min_color_change_duration = self.config.MIN_COLOR_CHANGE_DURATION

        # Ink spray detection parameters
        self.ink_detection_threshold = self.config.INK_DARKNESS_THRESHOLD
        self.ink_area_threshold = self.config.INK_AREA_THRESHOLD
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        # Tracking variables
        self.last_dominant_color = None
        self.last_color_change_time = 0
        self.ink_spray_active = False
        self.last_ink_time = 0

    def extract_dominant_colors(self, frame):
        """Extract dominant color from frame using simple averaging"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate average color values
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])

        return np.array([h_mean, s_mean, v_mean])

    def calculate_color_distance(self, color1, color2):
        """Calculate distance between two HSV colors"""
        # Handle hue wraparound (0 and 180 are close)
        h_diff = abs(color1[0] - color2[0])
        if h_diff > 90:
            h_diff = 180 - h_diff

        # Calculate weighted distance
        distance = np.sqrt((h_diff * 2)**2 + (color1[1] - color2[1])**2 + (color1[2] - color2[2])**2)
        return distance



    def detect_color_change(self, frame, timestamp):
        """Detect significant color changes in the octopus"""
        # Extract dominant color
        dominant_color = self.extract_dominant_colors(frame)

        if self.last_dominant_color is not None:
            color_distance = self.calculate_color_distance(dominant_color, self.last_dominant_color)

            if color_distance > self.color_change_threshold:
                if timestamp - self.last_color_change_time > self.min_color_change_duration:
                    # Get color descriptions for both current and previous
                    current_description = self.describe_color(dominant_color)
                    previous_description = self.describe_color(self.last_dominant_color)

                    # Only record if there's a meaningful dark/light change
                    if current_description != previous_description:
                        self.events.append({
                            'Time': timestamp,
                            'Subject': self.config.DEFAULT_SUBJECT_NAME,
                            'Behavior': self.config.BEHAVIOR_TYPES['color_change'],
                            'Modifier': f"To {current_description}",
                            'Comment': f'From {previous_description} to {current_description}' if self.config.INCLUDE_DEBUG_INFO else ''
                        })

                        self.last_color_change_time = timestamp
                        print(f"Color change detected at {timestamp:.2f}s - {previous_description} to {current_description}")

        self.last_dominant_color = dominant_color

    def describe_color(self, hsv_color):
        """Convert HSV color to simple dark/light classification"""
        h, s, v = hsv_color

        # Simple dark/light classification based on brightness (value)
        if v < self.config.BRIGHTNESS_THRESHOLD:
            return "Dark"
        else:
            return "Light"

    def detect_ink_spray(self, frame, timestamp):
        """Detect ink spray events using background subtraction and darkness detection"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)

        # Convert frame to grayscale for darkness detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create mask for very dark areas (potential ink)
        _, dark_mask = cv2.threshold(gray, self.ink_detection_threshold, 255, cv2.THRESH_BINARY_INV)

        # Combine foreground mask with dark areas
        ink_mask = cv2.bitwise_and(fg_mask, dark_mask)

        # Find contours of potential ink clouds
        contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ink_detected = False
        total_ink_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.ink_area_threshold:
                ink_detected = True
                total_ink_area += area

        # Record ink spray events (with debouncing)
        if ink_detected and not self.ink_spray_active:
            if timestamp - self.last_ink_time > self.config.MIN_INK_DURATION:
                self.events.append({
                    'Time': timestamp,
                    'Subject': self.config.DEFAULT_SUBJECT_NAME,
                    'Behavior': self.config.BEHAVIOR_TYPES['ink_spray'],
                    'Modifier': 'Start',
                    'Comment': f'Area: {total_ink_area}' if self.config.INCLUDE_DEBUG_INFO else ''
                })
                self.ink_spray_active = True
                self.last_ink_time = timestamp
                print(f"Ink spray detected at {timestamp:.2f}s")

        elif not ink_detected and self.ink_spray_active:
            self.events.append({
                'Time': timestamp,
                'Subject': self.config.DEFAULT_SUBJECT_NAME,
                'Behavior': self.config.BEHAVIOR_TYPES['ink_spray'],
                'Modifier': 'End',
                'Comment': 'Dissipated'
            })
            self.ink_spray_active = False

        return ink_mask



    def process_video(self):
        """Main function to process video and detect octopus behaviors"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        print(f"Processing video: {self.video_path}")
        print(f"Duration: {duration:.2f}s, FPS: {fps}, Frames: {frame_count}")

        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Resize frame for faster processing
            height, width = frame.shape[:2]
            if width > self.config.VIDEO_RESIZE_WIDTH:
                scale = self.config.VIDEO_RESIZE_WIDTH / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Apply detection algorithms
            self.detect_color_change(frame, timestamp)
            ink_mask = self.detect_ink_spray(frame, timestamp)

            # Display frame with annotations (optional)
            if self.config.DISPLAY_VIDEO:
                display_frame = frame.copy()

                # Overlay ink detection
                if ink_mask is not None:
                    ink_overlay = cv2.applyColorMap(ink_mask, cv2.COLORMAP_JET)
                    display_frame = cv2.addWeighted(display_frame, 0.8, ink_overlay, 0.2, 0)

                # Add timestamp
                cv2.putText(display_frame, f"Time: {timestamp:.2f}s",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show current dominant color
                if self.last_dominant_color is not None:
                    color_rect = np.full((50, 100, 3), self.last_dominant_color, dtype=np.uint8)
                    display_frame[10:60, display_frame.shape[1]-110:display_frame.shape[1]-10] = color_rect

                cv2.imshow('Octopus Behavior Detection', display_frame)

            # Progress indicator
            if frame_number % 30 == 0:  # Every 30 frames
                progress = (frame_number / frame_count) * 100
                print(f"Progress: {progress:.1f}% - Events detected: {len(self.events)}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_number += 1

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Save results
        self.save_events()
        print(f"\nProcessing complete! Detected {len(self.events)} events.")
        print(f"Events saved to {self.output_file}")

    def save_events(self):
        """Save detected events to CSV file compatible with BORIS"""
        if self.events:
            df = pd.DataFrame(self.events)
            df = df.sort_values('Time')  # Sort by timestamp
            df.to_excel(self.output_file, index=False, sheet_name="Octopus_Behaviors")

            # Print summary
            behavior_counts = df['Behavior'].value_counts()
            print("\nEvent Summary:")
            for behavior, count in behavior_counts.items():
                print(f"  {behavior}: {count} events")
        else:
            print("No events detected to save.")

def main():
    """Main function to run the octopus behavior detector"""
    video_path = "../videoplayback.mp4"  # Update with your video path
    output_csv = "octopus_behavior_events.xlsx"

    # Create custom config if needed
    config = DetectionConfig()

    # Create detector instance
    detector = OctopusEventDetector(video_path, output_csv, config)

    # Process the video
    detector.process_video()

if __name__ == "__main__":
    main()
