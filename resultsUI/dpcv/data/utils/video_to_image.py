
import cv2
import os
import zipfile
from pathlib import Path
import argparse
from multiprocessing import Pool
from tqdm import tqdm
# from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

import numpy as np

def crop_to_square(img):
    h, w, _ = img.shape
    c_x, c_y = int(w / 2), int(h / 2)
    img = img[:, c_x - c_y: c_x + c_y]
    return img


def frame_extract(video_path, save_dir, resize=(456, 256), transform=None):
    """
    Extract frames from video at 15 fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
 
    # Retrieve the original FPS from the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        print("Error: Failed to retrieve frame rate.")
        return
 
    # Calculate the frame skip rate to approximate 15 fps extraction
    skip_rate = max(1, int(round(original_fps / 15)))
 
    # Extract the base filename from the video path
    file_name = Path(video_path).stem
 
    # Construct the directory path where the frames will be saved
    save_path = Path(save_dir).joinpath(file_name)
    os.makedirs(save_path, exist_ok=True)
 
    frame_count = 0  # Counter for frames processed
    saved_frame_count = 0  # Counter for frames saved
 
    while True:
        ret, frame = cap.read()
 
        # Break the loop if no frame is read
        if not ret:
            break
 
        # Only save every skip_rate'th frame
        if frame_count % skip_rate == 0:
            if transform:
                frame = transform(frame)
 
            # Resize the frame
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
 
            # Construct the filename for the saved frame
            frame_filename = f"{save_path}/frame_{saved_frame_count + 1}.jpg"
 
            # Save the frame
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            
 
        frame_count += 1
 
 
    # Release the video capture object and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Extraction completed, {saved_frame_count} frames saved.")

def long_time_task(video, parent_dir):
        print(f"execute {video} ...")
        return frame_extract(video_path=video, save_dir=parent_dir, resize=(256, 256), transform=crop_to_square)

def convert_videos_to_frames(video_dir, output_dir):
    p = Pool(8)
    path = Path(video_dir)
    i = 0
    video_pts = list(path.rglob("*.mp4"))
    print("Making frames ")
    for video in tqdm(video_pts):
        i += 1
        video_path = str(video)
        if output_dir is not None:
            saved_dir = output_dir
        else:
            saved_dir = output_dir
        p.apply_async(long_time_task, args=(video_path, saved_dir))
        # frame_extract(video_path=video_path, save_dir=saved_dir, resize=(256, 256), transform=crop_to_square)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(f"processed {i} videos")

