import cv2
import os
import tempfile
import shutil
import re # For sort_nicely if needed
from logs import log, warn
# --- New Function to Extract Frames ---
def extract_frames_from_video(video_path, output_folder):
    """
    Reads a video file and saves each frame as a PNG image in the output folder.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.

    Returns:
        list: A list of paths to the extracted image frames, sorted numerically.
              Returns an empty list if the video cannot be opened or no frames are read.
    """
    extracted_frame_paths = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        warn(f"Error opening video file: {video_path}")
        return extracted_frame_paths

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video or error

        # Format frame number with leading zeros for proper sorting
        frame_filename = f"frame_{frame_count:06d}.png"
        frame_path = os.path.join(output_folder, frame_filename)

        try:
            cv2.imwrite(frame_path, frame)
            extracted_frame_paths.append(frame_path)
            frame_count += 1
        except Exception as e:
            warn(f"Error writing frame {frame_count} to {frame_path}: {e}")
            # Decide whether to continue or stop on write error

        if frame_count % 100 == 0:
             log(f"Extracted {frame_count} frames...")

    cap.release()
    log(f"Finished extracting {frame_count} frames to {output_folder}")

    # Ensure paths are sorted correctly before returning (though sequential naming helps)
    # extracted_frame_paths = sort_nicely(extracted_frame_paths) # Usually not needed with zero-padding
    return extracted_frame_paths
