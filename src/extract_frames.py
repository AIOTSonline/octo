import cv2 as cv
import os
from tqdm import tqdm

VIDEO_PATH = "/Users/himanshu/Downloads/204750--vv-1.mp4"
OUTPUT_PATH = "octopus_frames"

START_TIME = 0
END_TIME = 2*60+50
SKIP_FRAMES = 5

CONTOUR_THRESHOLD = 8000

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

existing_files = [f for f in os.listdir(OUTPUT_PATH) if f.startswith("frame_") and f.endswith(".jpg")]

if existing_files:
    last_index = max(int(f.split("_")[1].split(".")[0]) for f in existing_files)
    saved_count = last_index + 1
else:
    saved_count = 0

cap = cv.VideoCapture(VIDEO_PATH)

fps = cap.get(cv.CAP_PROP_FPS)
total_video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

start_frame = int(START_TIME * fps)

if END_TIME == -1:
    end_frame = total_video_frames
else:
    end_frame = int(END_TIME * fps)

cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)

ret, prev_frame = cap.read()
if not ret:
    print("Video could not be opened")
    exit()

prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
prev_gray = cv.GaussianBlur(prev_gray, (21, 21), 0)

frame_id = start_frame

total_frames = end_frame - start_frame

with tqdm(total=total_frames, desc="Processing Video") as pbar:

    while frame_id < end_frame:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (21, 21), 0)

        diff = cv.absdiff(prev_gray, gray)

        thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)[1]
        thresh = cv.dilate(thresh, None, iterations=2)

        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            if cv.contourArea(contour) > CONTOUR_THRESHOLD:
                motion_detected = True
                break

        if motion_detected and frame_id % SKIP_FRAMES == 0:
            save_path = os.path.join(OUTPUT_PATH, f"frame_{saved_count}.jpg")
            cv.imwrite(save_path, frame, [cv.IMWRITE_JPEG_QUALITY, 85])
            saved_count += 1

        prev_gray = gray
        frame_id += 1
        pbar.update(1)

cap.release()

print("Saved frames:", saved_count)
