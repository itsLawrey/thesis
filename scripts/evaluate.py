import cv2
import os
import numpy as np # Useful for handling keypoints array
import re
from pathlib import Path
from logs import log, warn

def sort_nicely(files):
    """Sort the given list of files in a human-friendly way (numerically)."""
    def try_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    def alphanum_key(s):
        return [try_int(c) for c in re.split('([0-9]+)', s)]

    return sorted(files, key=alphanum_key)




def create_video_from_image_frames(folder_path, output_folder, fps=25, flag=False):
    """
    Creates a video from image frames stored in a specified folder.

    The output video file will be named after the input folder (e.g.,
    if folder_path is '/path/to/my_images', the output will be
    '<output_folder>/my_images.mp4').

    Args:
        folder_path (str): The path to the folder containing the image frames.
        output_folder (str): The path to the folder where the output video
                            will be saved.
        fps (int): The desired frame rate for the output video. Defaults to 25.
    """
    
    if not flag:
        return
    
    
    print(f"[INFO] Starting video creation from folder: {folder_path}")
    print(f"[INFO] Output will be saved in: {output_folder}")

    # --- Validate Input Folder ---
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Input folder not found or is not a directory: {folder_path}")
        return

    # --- Get Image Files ---
    try:
        # List files, ensuring they are actual files and have image extensions
        image_files = [f for f in os.listdir(folder_path)
                    if os.path.isfile(os.path.join(folder_path, f)) and
                        f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    except OSError as e:
        print(f"[ERROR] Could not read directory {folder_path}: {e}")
        return

    if not image_files:
        print(f"[WARNING] No suitable image files (.png, .jpg, .jpeg, .bmp, .tiff) found in {folder_path}.")
        return

    # --- Sort Files ---
    print(f"[INFO] Found {len(image_files)} image files. Sorting numerically...")
    # Use sort_nicely for human-friendly numeric sorting (e.g., frame_1, frame_2, frame_10)
    image_files = sort_nicely(image_files)
    # print(f"[DEBUG] Sorted files (first 10): {image_files[:10]}") # Optional debug print

    # --- Prepare Output Path ---
    # Ensure the output directory exists
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"[INFO] Ensured output directory exists: {output_folder}")
    except OSError as e:
        print(f"[ERROR] Could not create output directory {output_folder}: {e}")
        return

    # Get the base name of the input folder
    input_folder_basename = os.path.basename(os.path.normpath(folder_path))
    if not input_folder_basename: # Handle cases like '/' or '.'
        input_folder_basename = "output_video" # Provide a default name
        print(f"[WARNING] Could not determine input folder name, using default: {input_folder_basename}")

    # Construct the final output video path
    output_video_filename = f"{input_folder_basename}.mp4"
    final_output_video_path = os.path.join(output_folder, output_video_filename)
    print(f"[INFO] Final output video path set to: {final_output_video_path}")


    # --- Get Video Properties from First Image ---
    first_image_path = os.path.join(folder_path, image_files[0])
    print(f"[INFO] Reading first image for dimensions: {first_image_path}")
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"[ERROR] Failed to load the first image: {first_image_path}. Cannot determine video dimensions.")
        return

    height, width, layers = first_image.shape
    if layers != 3:
        print(f"[WARNING] First image has {layers} channels. Video writer expects 3 (BGR). Problems might occur.")

    print(f"[INFO] Video dimensions set to: Width={width}, Height={height}")

    # --- Initialize Video Writer ---
    # Using 'mp4v' codec for .mp4 files. Alternatives include 'XVID'.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        out = cv2.VideoWriter(final_output_video_path, fourcc, float(fps), (width, height))
        if not out.isOpened():
            print(f"[ERROR] Failed to open VideoWriter for path: {final_output_video_path}. Check codec availability and permissions.")
            return
        print(f"[INFO] VideoWriter initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Exception during VideoWriter initialization: {e}")
        return


    # --- Process Images and Write Video ---
    print(f"[INFO] Writing {len(image_files)} frames to video...")
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[WARNING] Skipping frame: Error loading image {image_path}")
            continue

        # Ensure frame dimensions match the video dimensions
        if image.shape[0] != height or image.shape[1] != width:
            print(f"[WARNING] Resizing frame {image_file} from {image.shape[:2]} to {(height, width)}")
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        elif image.shape[2] != layers:
            print(f"[WARNING] Frame {image_file} has {image.shape[2]} channels, expected {layers}. Skipping frame.")
            continue # Skip frames with wrong number of channels


        # Write the frame
        out.write(image)

        # Optional: Progress indicator
        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_files):
            print(f"  Processed frame {idx + 1}/{len(image_files)}: {image_file}")


    # --- Release Resources ---
    print("[INFO] Releasing VideoWriter...")
    out.release()
    print(f"[INFO] Video creation complete. File saved at: {final_output_video_path}")




def darker_color(color, amount=50):
    """Creates a darker version of the input BGR color."""
    return tuple(max(0, c - amount) for c in color)

def draw_corrected_predictions(corrected_predictions_dict, output_folder=None, save_images=False):
    """
    Draws corrected keypoints and skeletons onto images based on the input dictionary.
    Returns a list of annotated image objects (NumPy arrays). Optionally saves
    the annotated images to disk.

    Args:
        corrected_predictions_dict (dict): Dictionary mapping image paths to keypoint data.
                                           Format: {img_path: {kp_idx: {'x': x, 'y': y}, ...}}
                                           kp_idx can be string or int, will be converted to int.
        output_folder (str or Path, optional): Path to the folder where annotated images
                                               will be saved *if* save_images is True.
                                               Defaults to None. Required if save_images=True.
        save_images (bool, optional): If True, saves the annotated images to the
                                      specified output_folder. Defaults to False.

    Returns:
        list: A list of NumPy arrays, where each array represents an image with
              annotations drawn on it. Returns an empty list if no images
              were successfully processed.
    """
    # --- Input Validation and Initialization ---
    if save_images and output_folder is None:
        warn("Output folder must be provided when save_images is True. Images will not be saved.")
        save_images = False # Disable saving if no folder provided

    annotated_image_objects = [] # <--- List to store annotated image NumPy arrays
    output_path_obj = Path(output_folder) if output_folder else None
    successfully_saved_count = 0 # Counter for saved images if save_images is True

    # Keypoint color mapping and drawing settings
    color_mapping = {
        "default": (0, 255, 0),       # Green
        "left_leg": (0, 255, 255),    # Yellow
        "right_leg": (255, 255, 0),   # Aqua
        "skeleton": (0, 0, 0)         # Black for the skeleton lines
    }
    left_leg_keypoints = {7, 8, 9}
    right_leg_keypoints = {10, 11, 12}
    connections = [
        (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (3, 7), (7, 8), (8, 9),
        (3, 10), (10, 11), (11, 12)
    ]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)
    TEXT_OFFSET_X = 5
    TEXT_OFFSET_Y = -5

    if not corrected_predictions_dict:
        warn("Input dictionary is empty. Nothing to draw.")
        return annotated_image_objects

    # Create output directory only if saving is enabled
    if save_images and output_path_obj:
        try:
            output_path_obj.mkdir(parents=True, exist_ok=True)
            log(f"Saving annotated images enabled. Target folder: {output_path_obj}")
        except OSError as e:
            warn(f"Error creating output directory {output_path_obj}: {e}. Disabling image saving.")
            save_images = False # Disable saving if directory creation fails

    processed_count = 0
    for image_path_str, keypoints_data in corrected_predictions_dict.items():
        image_path = Path(image_path_str)
        log(f"Processing: {image_path.name}")

        # Load the original image
        img = cv2.imread(str(image_path))
        if img is None:
            warn(f"Failed to load image: {image_path}. Skipping.")
            continue

        processed_count += 1 # Count as processed even if no keypoints are drawn later

        if not keypoints_data:
            warn(f"No keypoint data found for {image_path.name}. Returning original image.")
            annotated_image_objects.append(img) # Add original image if no keypoints
            # Optionally save the original image if saving is enabled
            if save_images and output_path_obj:
                 out_filename = f"{image_path.stem}_original_no_kpts.png" # Indicate no kpts
                 output_img_path = output_path_obj / out_filename
                 try:
                     cv2.imwrite(str(output_img_path), img)
                     successfully_saved_count += 1
                 except Exception as e:
                     warn(f"  [Error] Failed to save original image {output_img_path}: {e}")
            continue # Move to next image

        # --- Prepare keypoints list (same logic as before) ---
        max_idx = -1
        keypoints_int_dict = {}
        if keypoints_data:
            try:
                keypoints_int_dict = {int(k): v for k, v in keypoints_data.items()}
                if keypoints_int_dict: max_idx = max(keypoints_int_dict.keys())
            except (ValueError, TypeError) as e:
                 warn(f"Error converting keypoint indices for {image_path.name}: {e}. Skipping drawing.")
                 annotated_image_objects.append(img) # Add original image
                 continue
        if max_idx < 0:
             warn(f"No valid keypoints after processing for {image_path.name}. Skipping drawing.")
             annotated_image_objects.append(img) # Add original image
             continue

        keypoints_list = [(0, 0)] * (max_idx + 1)
        valid_point_found = False
        for idx, data in keypoints_int_dict.items():
            if 0 <= idx <= max_idx:
                try:
                    x, y = int(data['x']), int(data['y'])
                    keypoints_list[idx] = (x, y)
                    if x > 0 or y > 0: valid_point_found = True
                except (KeyError, ValueError, TypeError) as e:
                    warn(f"Invalid data format for keypoint {idx} in {image_path.name}: {e}")
            else: warn(f"Keypoint index {idx} out of range (0-{max_idx}) for {image_path.name}")

        if not valid_point_found:
            warn(f"No valid keypoint coordinates > (0,0) found for {image_path.name}. Skipping drawing.")
            annotated_image_objects.append(img) # Add original image
            continue

        # --- Draw Skeleton Connections (on img) ---
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx <= max_idx and end_idx <= max_idx:
                start_point, end_point = keypoints_list[start_idx], keypoints_list[end_idx]
                if (start_point[0] <= 0 and start_point[1] <= 0) or (end_point[0] <= 0 and end_point[1] <= 0): continue
                is_left = start_idx in left_leg_keypoints or end_idx in left_leg_keypoints
                is_right = start_idx in right_leg_keypoints or end_idx in right_leg_keypoints
                line_color = darker_color(color_mapping["left_leg"]) if is_left else \
                             darker_color(color_mapping["right_leg"]) if is_right else \
                             color_mapping["skeleton"]
                cv2.line(img, start_point, end_point, line_color, 2)
            # else: warn(...) # Warning for out-of-range connections can be noisy

        # --- Draw Keypoints and Annotate (on img) ---
        for keypoint_idx, point_coords in enumerate(keypoints_list):
            x, y = point_coords
            if x <= 0 and y <= 0: continue
            color = color_mapping["left_leg"] if keypoint_idx in left_leg_keypoints else \
                    color_mapping["right_leg"] if keypoint_idx in right_leg_keypoints else \
                    color_mapping["default"]
            cv2.circle(img, (x, y), 4, color, -1)
            cv2.putText(img, str(keypoint_idx), (x + TEXT_OFFSET_X, y + TEXT_OFFSET_Y),
                        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # --- Store the annotated image object ---
        annotated_image_objects.append(img) # <--- Add the modified img object

        # --- Save the annotated image IF flag is set ---
        if save_images and output_path_obj:
            out_filename = f"{image_path.stem}_corrected_annotated.png"
            output_img_path = output_path_obj / out_filename
            try:
                cv2.imwrite(str(output_img_path), img)
                successfully_saved_count += 1
            except Exception as e:
                warn(f"  [Error] Failed to save annotated image {output_img_path}: {e}")

    # --- Final Log Message ---
    log(f"Finished processing {processed_count} images.")
    if save_images:
        log(f"Attempted to save {successfully_saved_count} annotated images to {output_path_obj}.")

    return annotated_image_objects # <--- Return the list of image objects


def draw_uncorrected_predictions(yolo_results_list, output_folder=None, save_images=False):
    """
    Draws UNCORRECTED keypoints and skeletons onto images based on YOLO Results list.
    Returns a list of annotated image objects (NumPy arrays). Optionally saves
    the annotated images to disk.

    Args:
        yolo_results_list (list): List of ultralytics.engine.results.Results objects.
        output_folder (str or Path, optional): Path to the folder where annotated images
                                               will be saved *if* save_images is True.
                                               Defaults to None. Required if save_images=True.
        save_images (bool, optional): If True, saves the annotated images to the
                                      specified output_folder. Defaults to False.

    Returns:
        list: A list of NumPy arrays, where each array represents an image with
              annotations drawn on it. Returns an empty list if no results
              were processed.
    """
    # --- Input Validation and Initialization ---
    if save_images and output_folder is None:
        warn("Output folder must be provided when save_images is True. Images will not be saved.")
        save_images = False # Disable saving

    annotated_image_objects = [] # <--- List to store annotated image NumPy arrays
    output_path_obj = Path(output_folder) if output_folder else None
    successfully_saved_count = 0

    # --- Settings (Keep identical to draw_corrected_predictions) ---
    color_mapping = {
        "default": (0, 255, 0), "left_leg": (0, 255, 255),
        "right_leg": (255, 255, 0), "skeleton": (0, 0, 0)
    }
    left_leg_keypoints = {7, 8, 9}    # Using sets
    right_leg_keypoints = {10, 11, 12}
    connections = [
        (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), # Head, spine, tail
        (3, 7), (7, 8), (8, 9),                         # Left leg
        (3, 10), (10, 11), (11, 12)                      # Right leg
    ]
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    TEXT_COLOR = (255, 255, 255)
    TEXT_OFFSET_X = 5
    TEXT_OFFSET_Y = -5
    # --- End Settings ---

    if not yolo_results_list:
        warn("Input YOLO results list is empty. Nothing to draw.")
        return annotated_image_objects

    # Create output directory only if saving is enabled
    if save_images and output_path_obj:
        try:
            output_path_obj.mkdir(parents=True, exist_ok=True)
            log(f"Saving UNCORRECTED annotated images enabled. Target folder: {output_path_obj}")
        except OSError as e:
            warn(f"Error creating output directory {output_path_obj}: {e}. Disabling image saving.")
            save_images = False

    processed_count = 0
    # Iterate through the list of Results objects
    for result in yolo_results_list:
        # Ensure result has a path attribute
        if not hasattr(result, 'path') or result.path is None:
            warn("Result object missing 'path' attribute. Skipping.")
            continue

        image_path = Path(result.path) # Use Path object
        log(f"Processing Uncorrected: {image_path.name}")

        # Load the original image
        img = cv2.imread(str(image_path)) # cv2 needs string
        if img is None:
            warn(f"Failed to load image: {image_path}. Skipping.")
            continue

        processed_count += 1
        drawing_done = False # Flag to track if any drawing happened on this image

        # Check if keypoints exist in the result object
        # Using more robust checks
        keypoints_present = (
            hasattr(result, 'keypoints') and
            result.keypoints is not None and
            hasattr(result.keypoints, 'xy') and
            result.keypoints.xy is not None and
            result.keypoints.xy.numel() > 0
        )

        if not keypoints_present:
            warn(f"No keypoint data found in YOLO result for {image_path.name}. Returning original image.")
            # No drawing needed, handled later by appending img if drawing_done is False
        else:
            # Get keypoints tensor (potentially multiple instances)
            keypoints_tensor = result.keypoints.xy.cpu() # Move to CPU

            # --- Loop through each detected instance on the image ---
            num_instances = keypoints_tensor.shape[0]
            for i in range(num_instances):
                instance_keypoints_np = keypoints_tensor[i].numpy()
                keypoints_list = [tuple(map(int, kp)) for kp in instance_keypoints_np]
                num_keypoints_detected = len(keypoints_list)
                instance_has_valid_point = any(kp[0] > 0 or kp[1] > 0 for kp in keypoints_list)

                if not instance_has_valid_point:
                    # warn(f"Instance {i} in {image_path.name} has no valid keypoints > (0,0). Skipping drawing for this instance.")
                    continue # Skip drawing this instance if all points are 0,0

                drawing_done = True # Mark that we are drawing something

                # --- Draw Skeleton Connections for this instance ---
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < num_keypoints_detected and end_idx < num_keypoints_detected:
                        start_point, end_point = keypoints_list[start_idx], keypoints_list[end_idx]
                        if (start_point[0] <= 0 and start_point[1] <= 0) or (end_point[0] <= 0 and end_point[1] <= 0): continue
                        is_left = start_idx in left_leg_keypoints or end_idx in left_leg_keypoints
                        is_right = start_idx in right_leg_keypoints or end_idx in right_leg_keypoints
                        line_color = darker_color(color_mapping["left_leg"]) if is_left else \
                                     darker_color(color_mapping["right_leg"]) if is_right else \
                                     color_mapping["skeleton"]
                        cv2.line(img, start_point, end_point, line_color, 2)
                    # else: warn(...) # Optional warning

                # --- Draw Keypoints and Annotate for this instance ---
                for keypoint_idx, point_coords in enumerate(keypoints_list):
                    x, y = point_coords
                    if x <= 0 and y <= 0: continue
                    color = color_mapping["left_leg"] if keypoint_idx in left_leg_keypoints else \
                            color_mapping["right_leg"] if keypoint_idx in right_leg_keypoints else \
                            color_mapping["default"]
                    cv2.circle(img, (x, y), 4, color, -1)
                    cv2.putText(img, str(keypoint_idx), (x + TEXT_OFFSET_X, y + TEXT_OFFSET_Y),
                                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            # --- End instance loop ---

        # --- Store the (potentially modified) image object ---
        annotated_image_objects.append(img) # <--- Add the img object (original or annotated)

        # --- Save the annotated image IF flag is set AND drawing occurred ---
        # Decide if you want to save even if no keypoints were found/drawn
        # Current logic only saves if drawing_done is True
        if save_images and output_path_obj and drawing_done:
            # Use a different suffix to distinguish from corrected images
            out_filename = f"{image_path.stem}_uncorrected_annotated.png"
            output_img_path = output_path_obj / out_filename
            try:
                cv2.imwrite(str(output_img_path), img)
                successfully_saved_count += 1
            except Exception as e:
                warn(f"  [Error] Failed to save uncorrected image {output_img_path}: {e}")
        elif save_images and output_path_obj and not drawing_done:
             # Optionally save the original if no keypoints were drawn
             out_filename = f"{image_path.stem}_uncorrected_original_no_kpts.png"
             output_img_path = output_path_obj / out_filename
             try:
                 cv2.imwrite(str(output_img_path), img)
                 # Don't increment successfully_saved_count here if you only count annotated ones
             except Exception as e:
                 warn(f"  [Error] Failed to save original uncorrected image {output_img_path}: {e}")


    # --- Final Log Message ---
    log(f"Finished processing {processed_count} uncorrected results.")
    if save_images:
        log(f"Attempted to save {successfully_saved_count} annotated uncorrected images to {output_path_obj}.")

    return annotated_image_objects # <--- Return the list of image objects

def is_long_enough(folder_path, threshold=250):

    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    frame_count = 0
  
    frame_count = sum(1 for item in os.listdir(folder_path)
                      if os.path.isfile(os.path.join(folder_path, item)) and
                         item.lower().endswith(IMAGE_EXTENSIONS))


    is_enough = frame_count > threshold
    return is_enough

# if __name__ == "__main__":
#     create_video_from_image_frames('/home/dalloslorand/YOLO_lori/cut_video_images_4_pred/R5W1_500',
#                                    "/home/dalloslorand/YOLO_lori/predicted_videos/sample_videos_for_prediction",
#                                    25)