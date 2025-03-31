import cv2
import os
import numpy as np # Useful for handling keypoints array
from pipeline import log, warn

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
#TODO: output path csak folder legyen es az input neve vege legyen az mp4 neve
def create_video_from_image_frames(folder_path, output_video_path, fps=30):
    # Get a list of image files in the folder, explicitly excluding subdirectories
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                   and os.path.isfile(os.path.join(folder_path, f))]
    
    # Sort files in numerical order using the custom sort function
    image_files = sort_nicely(image_files)
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    # Create the output folder if it doesn't exist
    output_folder = os.path.dirname(output_video_path)
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the first image to get video properties
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Error loading the first image {first_image_path}")
        return
    
    height, width, _ = first_image.shape

    # Initialize video writer (with frame rate and proper codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each image and write it to the video
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image {image_path}")
            continue
        
        # Resize image to match the first image's resolution (to avoid dimension mismatches)
        resized_image = cv2.resize(image, (width, height))

        # Write the resized image as a frame to the video
        out.write(resized_image)

        # Logging to verify the correct order of processing
        print(f"Writing frame {idx + 1} of {len(image_files)}: {image_file}")

    # Release the video writer
    out.release()
    print(f"Video saved at {output_video_path}")
#TODO:
def create_video_from_image_lists(imlist, output_video_path, fps=30):
    # Sort files in numerical order using the custom sort function
    image_files = sort_nicely(image_files)
    
    if not image_files:
        print("No images found in the folder.")
        return
    
    # Create the output folder if it doesn't exist
    output_folder = os.path.dirname(output_video_path)
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the first image to get video properties
    #TODO FIX
    first_image_path = os.path.join(folder_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Error loading the first image {first_image_path}")
        return
    
    height, width, _ = first_image.shape

    # Initialize video writer (with frame rate and proper codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each image and write it to the video
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image {image_path}")
            continue
        
        # Resize image to match the first image's resolution (to avoid dimension mismatches)
        resized_image = cv2.resize(image, (width, height))

        # Write the resized image as a frame to the video
        out.write(resized_image)

        # Logging to verify the correct order of processing
        print(f"Writing frame {idx + 1} of {len(image_files)}: {image_file}")

    # Release the video writer
    out.release()
    print(f"Video saved at {output_video_path}")

def darker_color(color, amount=50):
    """Creates a darker version of the input BGR color."""
    return tuple(max(0, c - amount) for c in color)

def draw_corrected_predictions(corrected_predictions_dict, output_folder):
    """
    Draws corrected keypoints and skeletons onto images based on the input dictionary.

    Args:
        corrected_predictions_dict (dict): Dictionary mapping image paths to keypoint data.
                                           Format: {img_path: {kp_idx: {'x': x, 'y': y}, ...}}
        output_folder (str): Path to the folder where annotated images will be saved.
    """
    # Keypoint color mapping and drawing settings
    color_mapping = {
        "default": (0, 255, 0),       # Green
        "left_leg": (0, 255, 255),    # Yellow
        "right_leg": (255, 255, 0),   # Aqua
        "skeleton": (0, 0, 0)         # Black for the skeleton lines (can be adjusted)
    }

    # Define left and right leg keypoints (used for drawing color)
    # Make sure these indices match the keypoint indices in your corrected_predictions_dict
    left_leg_keypoints = [7, 8, 9]
    right_leg_keypoints = [10, 11, 12]

    # Define skeleton connections (pairs of keypoint indices)
    # Ensure these indices are valid for your keypoint data
    connections = [
        (0, 2),   # Left ear to nose
        (1, 2),   # Right ear to nose
        (2, 3),   # Nose to spine
        (3, 4),   # Spine to tail 01
        (4, 5),   # Tail 01 to tail 02
        (5, 6),   # Tail 02 to tail 03
        (3, 7),   # Spine to left leg hip
        (7, 8),   # Left leg hip to left knee
        (8, 9),   # Left knee to left toe
        (3, 10),  # Spine to right leg hip
        (10, 11), # Right leg hip to right knee
        (11, 12), # Right knee to right toe
    ]

    # Text settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1 # Adjusted for potentially smaller keypoint numbers
    TEXT_COLOR = (255, 255, 255) # White text for visibility
    TEXT_OFFSET_X = 5
    TEXT_OFFSET_Y = -5
    if not corrected_predictions_dict:
        warn("Input dictionary is empty. Nothing to draw.")
        return

    os.makedirs(output_folder, exist_ok=True)
    log(f"Saving annotated images to: {output_folder}")

    processed_count = 0
    for image_path, keypoints_data in corrected_predictions_dict.items():
        log(f"Processing: {os.path.basename(image_path)}")

        # Load the original image
        img = cv2.imread(image_path)
        if img is None:
            warn(f"Failed to load image: {image_path}. Skipping.")
            continue

        if not keypoints_data:
            warn(f"No keypoint data found for {image_path}. Saving original image.")
            # Optionally save the original image or skip entirely
            # out_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_corrected_annotated.png"
            # output_img_path = os.path.join(output_folder, out_filename)
            # cv2.imwrite(output_img_path, img)
            continue # Skip drawing if no keypoints

        # --- Prepare keypoints list from the dictionary ---
        # Find the maximum keypoint index to determine the size of our list
        max_idx = -1
        if keypoints_data:
             # Ensure keys are integers if they aren't already
            int_keys = {int(k): v for k, v in keypoints_data.items()}
            max_idx = max(int_keys.keys())
            keypoints_data = int_keys # Use the dictionary with integer keys

        # Create a list to hold keypoint coordinates (x, y), indexed by keypoint index.
        # Initialize with (0, 0) for potentially missing indices up to max_idx.
        keypoints_list = [(0, 0)] * (max_idx + 1)
        for idx, data in keypoints_data.items():
            # Ensure index is within bounds (should be if max_idx is correct)
            if 0 <= idx <= max_idx:
                keypoints_list[idx] = (int(data['x']), int(data['y']))
            else:
                 warn(f"Keypoint index {idx} out of expected range for {image_path}")


        # --- Draw Skeleton Connections ---
        for connection in connections:
            start_idx, end_idx = connection

            # Check if indices are valid within our prepared list
            if start_idx <= max_idx and end_idx <= max_idx:
                start_point = keypoints_list[start_idx]
                end_point = keypoints_list[end_idx]

                # Check if keypoints were detected (not the default 0,0)
                # Using a small threshold > 0 might be safer than exact 0 check
                if (start_point[0] <= 0 and start_point[1] <= 0) or \
                   (end_point[0] <= 0 and end_point[1] <= 0):
                    continue # Skip connection if either point is missing

                # Determine color (similar logic to previous script)
                # Check if *either* point in the connection belongs to a leg
                is_left = start_idx in left_leg_keypoints or end_idx in left_leg_keypoints
                is_right = start_idx in right_leg_keypoints or end_idx in right_leg_keypoints

                if is_left:
                    line_color = darker_color(color_mapping["left_leg"])
                elif is_right:
                    line_color = darker_color(color_mapping["right_leg"])
                else:
                    line_color = color_mapping["skeleton"] # Use the defined skeleton color

                # Draw the line
                cv2.line(img, start_point, end_point, line_color, 2) # Thickness 2
            else:
                 warn(f"Connection indices {connection} out of range (max: {max_idx}) for {image_path}")


        # --- Draw Keypoints and Annotate ---
        for keypoint_idx, point_coords in enumerate(keypoints_list):
            x, y = point_coords

            # Skip drawing if keypoint was not detected (still 0,0)
            if x <= 0 and y <= 0:
                continue

            # Determine color based on leg assignment
            if keypoint_idx in left_leg_keypoints:
                color = color_mapping["left_leg"]
            elif keypoint_idx in right_leg_keypoints:
                color = color_mapping["right_leg"]
            else:
                color = color_mapping["default"]

            # Draw the keypoint circle
            cv2.circle(img, (x, y), 4, color, -1) # Radius 4, filled circle

            # Draw the keypoint index number
            cv2.putText(img, str(keypoint_idx), (x + TEXT_OFFSET_X, y + TEXT_OFFSET_Y),
                        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


        # --- Save the annotated image ---
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_filename = f"{base_name}_corrected_annotated.png"
        output_img_path = os.path.join(output_folder, out_filename)

        try:
            cv2.imwrite(output_img_path, img)
            processed_count += 1
        except Exception as e:
            warn(f"  [Error] Failed to save image {output_img_path}: {e}")

    log(f"Finished processing. Saved {processed_count} annotated images to {output_folder}.")

def draw_uncorrected_predictions(yolo_results_list, output_folder):
    """
    Draws UNCORRECTED keypoints and skeletons onto images based on YOLO Results list.

    Args:
        yolo_results_list (list): List of ultralytics.engine.results.Results objects.
        output_folder (str): Path to the folder where annotated images will be saved.
    """
    # --- Settings (Keep identical to draw_corrected_predictions) ---
    color_mapping = {
        "default": (0, 255, 0), "left_leg": (0, 255, 255),
        "right_leg": (255, 255, 0), "skeleton": (0, 0, 0)
    }
    left_leg_keypoints = [7, 8, 9]
    right_leg_keypoints = [10, 11, 12]
    connections = [
        (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7),
        (7, 8), (8, 9), (3, 10), (10, 11), (11, 12),
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
        return

    os.makedirs(output_folder, exist_ok=True)
    log(f"Saving UNCORRECTED annotated images to: {output_folder}")

    processed_count = 0
    # Iterate through the list of Results objects
    for result in yolo_results_list:
        image_path = result.path
        image_basename = os.path.basename(image_path)
        log(f"Processing Uncorrected: {image_basename}") # Less verbose

        # Load the original image
        img = cv2.imread(image_path)
        if img is None:
            warn(f"Failed to load image: {image_path}. Skipping.")
            continue

        # Check if keypoints exist in the result object
        if not hasattr(result, 'keypoints') or result.keypoints is None or result.keypoints.xy is None or result.keypoints.xy.numel() == 0:
            warn(f"No keypoint data found in YOLO result for {image_basename}. Skipping draw.")
            # Optionally save the original image if no keypoints were detected
            # base_name = os.path.splitext(image_basename)[0]
            # out_filename = f"{base_name}_uncorrected_annotated.png"
            # output_img_path = os.path.join(output_folder, out_filename)
            # cv2.imwrite(output_img_path, img)
            continue

        # Get keypoints tensor (potentially multiple instances)
        # Shape: [num_instances, num_keypoints, 2]
        keypoints_tensor = result.keypoints.xy.cpu() # Move to CPU

        # --- Loop through each detected instance on the image ---
        num_instances = keypoints_tensor.shape[0]
        for i in range(num_instances):
            # Get keypoints for the current instance as a NumPy array
            # Shape: [num_keypoints, 2]
            instance_keypoints_np = keypoints_tensor[i].numpy()

            # Convert to list of integer tuples for drawing
            # This list directly corresponds to keypoint indices 0, 1, 2,...
            keypoints_list = [tuple(map(int, kp)) for kp in instance_keypoints_np]
            num_keypoints_detected = len(keypoints_list)

            # --- Draw Skeleton Connections for this instance ---
            for connection in connections:
                start_idx, end_idx = connection

                # Check if indices are valid for the number of keypoints detected
                if start_idx < num_keypoints_detected and end_idx < num_keypoints_detected:
                    start_point = keypoints_list[start_idx]
                    end_point = keypoints_list[end_idx]

                    # Check if keypoints were detected (not 0,0)
                    if (start_point[0] <= 0 and start_point[1] <= 0) or \
                       (end_point[0] <= 0 and end_point[1] <= 0):
                        continue # Skip connection if either point is missing

                    # Determine color (same logic as corrected function)
                    is_left = start_idx in left_leg_keypoints or end_idx in left_leg_keypoints
                    is_right = start_idx in right_leg_keypoints or end_idx in right_leg_keypoints
                    if is_left: line_color = darker_color(color_mapping["left_leg"])
                    elif is_right: line_color = darker_color(color_mapping["right_leg"])
                    else: line_color = color_mapping["skeleton"]

                    # Draw the line
                    cv2.line(img, start_point, end_point, line_color, 2)
                else: warn(f"Uncorrected Connection indices {connection} out of range (num_kps: {num_keypoints_detected}) for {image_basename}") # Less verbose


            # --- Draw Keypoints and Annotate for this instance ---
            for keypoint_idx, point_coords in enumerate(keypoints_list):
                x, y = point_coords

                # Skip drawing if keypoint was not detected (0,0)
                if x <= 0 and y <= 0:
                    continue

                # Determine color based on leg assignment
                if keypoint_idx in left_leg_keypoints: color = color_mapping["left_leg"]
                elif keypoint_idx in right_leg_keypoints: color = color_mapping["right_leg"]
                else: color = color_mapping["default"]

                # Draw the keypoint circle
                cv2.circle(img, (x, y), 4, color, -1)

                # Draw the keypoint index number
                cv2.putText(img, str(keypoint_idx), (x + TEXT_OFFSET_X, y + TEXT_OFFSET_Y),
                            FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # --- Save the annotated image (after drawing all instances) ---
        base_name = os.path.splitext(image_basename)[0]
        # Use a different suffix to distinguish from corrected images
        out_filename = f"{base_name}_uncorrected_annotated.png"
        output_img_path = os.path.join(output_folder, out_filename)

        try:
            cv2.imwrite(output_img_path, img)
            processed_count += 1
        except Exception as e:
            warn(f"  [Error] Failed to save uncorrected image {output_img_path}: {e}")

    log(f"Finished processing uncorrected. Saved {processed_count} annotated images.")
