import numpy as np
from skimage.morphology import skeletonize
from pipeline import warn, log
import cv2
import os
import torch # ensure torch is imported
import gc # ensure gc is imported
import math # For sqrt


def get_skeletons(masks):
    skeletons = []
    for mask in masks:
        if mask is None: # Handle potential None masks
             skeletons.append(None)
             continue
        mask = mask.astype(np.uint8)
        skeleton = skeletonize(mask).astype(np.uint8)
        skeletons.append(skeleton)
    return skeletons

# --- Helper Function for Dynamic Radius ---
def determine_radius(prediction_result, default_radius, scaling_factor):
    """Calculates search radius based on the bounding box diagonal."""
    try:
        if prediction_result.boxes and prediction_result.boxes.xywh.numel() > 0:
            # Assuming the first box is the relevant one if multiple exist
            box_xywh = prediction_result.boxes.xywh[0] # Get tensor [cx, cy, w, h]
            w = box_xywh[2].item() # Extract width
            h = box_xywh[3].item() # Extract height

            if w > 0 and h > 0:
                diagonal = math.sqrt(w**2 + h**2)
                calculated_radius = int(round(diagonal * scaling_factor))
                # Add a minimum radius check if desired, e.g., max(5, calculated_radius)
                return max(5, calculated_radius) # Ensure radius is at least 5 pixels
            else:
                warn(f"Invalid bounding box dimensions (w={w}, h={h}) for {os.path.basename(prediction_result.path)}. Using default radius.")
                return default_radius
        else:
            warn(f"No bounding box found for {os.path.basename(prediction_result.path)}. Using default radius.")
            return default_radius
    except Exception as e:
        warn(f"Error calculating radius for {os.path.basename(prediction_result.path)}: {e}. Using default radius.")
        return default_radius
    
def process_toe(coord, search_radius, endpoints):
    toe_x, toe_y = coord
    if toe_x == 0 and toe_y == 0: return None, []
    if not endpoints: return None, []

    # Use the passed search_radius here
    nearby = [(ex, ey) for ex, ey in endpoints
              if np.linalg.norm(np.array([ex, ey]) - np.array([toe_x, toe_y])) <= search_radius]

    chosen_point = None
    if not nearby: return None, []
    if len(nearby) == 1: chosen_point = nearby[0]
    else:
        arr = np.array(nearby)
        avg_x = int(np.mean(arr[:, 0]))
        avg_y = int(np.mean(arr[:, 1]))
        chosen_point = (avg_x, avg_y)
    return chosen_point, nearby    

# --- Main Correction Function ---
def correct_toes_skeleton(yolo_predictions,
                          skeletons,
                          save_images_flag=True,
                          output_image_folder=None):
    """
    Corrects toe keypoints using skeleton endpoints within a DYNAMICALLY calculated radius
    based on the detection's bounding box size. Optionally saves visualization images.
    
    For each YOLO prediction, attempts to correct toe keypoints (indices 12 and 9)
    by searching for skeleton endpoints within a fixed radius in the corresponding skeleton.
    Generates and saves a composite image overlaying the skeleton and final keypoints.

    Args:
        yolo_predictions: An iterable of YOLO Results objects. Each object must have
                          .path and .keypoints attributes.
        skeletons: A list or iterable of skeleton images (NumPy arrays) corresponding
                   to each prediction, or None if no skeleton is available.

    Returns:
        dict: A dictionary where keys are image file paths (from pred.path) and
              values are sub-dictionaries. Each sub-dictionary maps a keypoint
              index (int) to {'x': int, 'y': int} representing the final coordinates
              (potentially corrected for toes).

    Assumptions:
      - Each prediction in yolo_predictions has:
          .path (string): image filepath.
          .keypoints: a Keypoints object with property .xy (a torch.Tensor)
      - Constants TOE_CIRCLE_RADIUS, FONT, etc. are defined globally.
      - OUTPUT_IMAGE_FOLDER is defined globally.
    
    """
    # --- Configuration & Constants ---
    # Default radius if bbox is unavailable or invalid
    DEFAULT_TOE_CIRCLE_RADIUS = 20
    # Factor to scale bounding box diagonal to get radius (NEEDS TUNING)
    BBOX_DIAGONAL_SCALING_FACTOR = 0.06 #szazalek

    # Visualization constants
    LIGHT_BLUE = (230, 216, 173)
    ORIGINAL_KP_COLOR = (0, 0, 255)
    SKELETON_COLOR = (0, 255, 0)
    TOE_CIRCLE_COLOR = (0, 255, 255)
    ENDPOINT_COLOR = (255, 0, 255)
    AVG_ENDPOINT_COLOR = LIGHT_BLUE
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_COLOR = (255, 255, 255)
    FONT_THICKNESS = 1
    TEXT_OFFSET_Y = 10

    # --- Parameter Validation ---
    if save_images_flag and output_image_folder is None:
        raise ValueError("output_image_folder must be provided when save_images_flag is True.")
    if save_images_flag:
        os.makedirs(output_image_folder, exist_ok=True)

    # Initialize the dictionary to store the results
    corrected_keypoints_data = {}

    # --- Main Processing Loop ---
    for pred, skeleton in zip(yolo_predictions, skeletons):
        
        image_path = pred.path
        image_basename = os.path.basename(image_path) # Get basename for logging
        
        image = cv2.imread(image_path)
        if image is None:
            warn(f"Could not load image {image_path}. Skipping prediction.")
            continue

        # --- Get Keypoints ---
        if not hasattr(pred, 'keypoints') or pred.keypoints is None or pred.keypoints.xy is None:
             warn(f"No keypoints found for image {image_path}. Skipping correction.")
             continue
        try:
            kp_tensor = pred.keypoints.xy.cpu()
            original_kp_np = kp_tensor.numpy().copy()
            kp_np = original_kp_np.copy()
        except Exception as e:
            warn(f"Error accessing keypoints for {image_path}: {e}. Skipping correction.")
            continue

        if original_kp_np.ndim == 3 and original_kp_np.shape[0] == 1: original_kp_np = original_kp_np[0]
        if kp_np.ndim == 3 and kp_np.shape[0] == 1: kp_np = kp_np[0]

        has_enough_keypoints = kp_np.shape[0] >= 13

        # --- Calculate Dynamic Radius for THIS image ---
        current_toe_radius = determine_radius(pred, DEFAULT_TOE_CIRCLE_RADIUS, BBOX_DIAGONAL_SCALING_FACTOR)
        # Optional: Log the calculated radius for debugging
        log(f"Image: {os.path.basename(image_path)}, Calculated Radius: {current_toe_radius}")

        # --- Correction Logic ---
        endpoints = []
        if skeleton is not None:
            # (Endpoint finding logic - unchanged)
            if skeleton.dtype != np.uint8: skeleton = skeleton.astype(np.uint8)
            binary_skeleton = (skeleton > 0).astype(np.uint8)
            if np.any(binary_skeleton):
                h, w = binary_skeleton.shape
                for y, x in np.argwhere(binary_skeleton):
                    is_endpoint = False
                    if y == 0 or y >= h - 1 or x == 0 or x >= w - 1:
                        y_min, y_max = max(0, y-1), min(h, y+2)
                        x_min, x_max = max(0, x-1), min(w, x+2)
                        neighbors = np.sum(binary_skeleton[y_min:y_max, x_min:x_max]) - binary_skeleton[y, x]
                        if neighbors == 1: is_endpoint = True
                    else:
                        neighbors = np.sum(binary_skeleton[y - 1:y + 2, x - 1:x + 2]) - 1
                        if neighbors == 1: is_endpoint = True
                    if is_endpoint: endpoints.append((int(x), int(y)))


        # --- Apply Correction ---
        new_left, nearby_left_endpoints = None, []
        new_right, nearby_right_endpoints = None, []
        if has_enough_keypoints:
            original_left_toe = tuple(original_kp_np[12].astype(int))
            original_right_toe = tuple(original_kp_np[9].astype(int))

            # Pass the calculated current_toe_radius to process_toe
            new_left, nearby_left_endpoints = process_toe(original_left_toe, current_toe_radius, endpoints)
            new_right, nearby_right_endpoints = process_toe(original_right_toe, current_toe_radius, endpoints)

            # *** ADDED LOGGING HERE ***
            if new_left is not None:
                kp_np[12] = np.array(new_left) # Update the working copy
                log(f"{image_basename}: Left toe UPDATED {original_left_toe} -> {new_left}")
            else:
                 # Only log if the original toe was detected (not 0,0)
                 if original_left_toe != (0, 0):
                     log(f"{image_basename}: Left toe NOT corrected from {original_left_toe}")
                 # else: pass # Optionally log skipped (0,0) toes

            if new_right is not None:
                kp_np[9] = np.array(new_right) # Update the working copy
                log(f"{image_basename}: Right toe UPDATED {original_right_toe} -> {new_right}")
            else:
                 # Only log if the original toe was detected (not 0,0)
                 if original_right_toe != (0, 0):
                     log(f"{image_basename}: Right toe NOT corrected from {original_right_toe}")
                 # else: pass # Optionally log skipped (0,0) toes
            # *** END OF ADDED LOGGING ***

        # --- Store Final Keypoints ---
        image_keypoints_dict = {}
        for idx, (x, y) in enumerate(kp_np):
            image_keypoints_dict[idx] = {'x': int(round(x)), 'y': int(round(y))}
        corrected_keypoints_data[image_path] = image_keypoints_dict

        # --- Create and Save Composite Image (Conditional Visualization) ---
        if save_images_flag:
            comp = image.copy()

            # 1. Draw Skeleton Overlay (unchanged)
            if skeleton is not None:
                 kernel = np.ones((3, 3), np.uint8)
                 if skeleton.dtype != np.uint8: skeleton_vis = skeleton.astype(np.uint8)
                 else: skeleton_vis = skeleton
                 skeleton_vis = (skeleton_vis > 0).astype(np.uint8) * 255
                 thick_skel = cv2.dilate(skeleton_vis, kernel, iterations=1)
                 overlay = np.zeros_like(comp)
                 overlay[thick_skel == 255] = SKELETON_COLOR
                 comp = cv2.addWeighted(comp, 0.8, overlay, 0.2, 0)
                 del skeleton_vis, thick_skel, overlay

            # 2. Draw ALL ORIGINAL keypoints (unchanged)
            for idx, (x, y) in enumerate(original_kp_np):
                px, py = int(round(x)), int(round(y))
                if px == 0 and py == 0: continue
                cv2.circle(comp, (px, py), 5, ORIGINAL_KP_COLOR, -1)
                cv2.putText(comp, str(idx), (px, py - TEXT_OFFSET_Y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            # 3. Draw Toe Correction Visualization (using current_toe_radius)
            if has_enough_keypoints:
                # --- Left Toe (Index 12) ---
                orig_lx, orig_ly = tuple(original_kp_np[12].astype(int))
                if orig_lx != 0 or orig_ly != 0:
                    # Use current_toe_radius for drawing the circle
                    cv2.circle(comp, (orig_lx, orig_ly), current_toe_radius, TOE_CIRCLE_COLOR, 1)
                    for ex, ey in nearby_left_endpoints: cv2.circle(comp, (ex, ey), 3, ENDPOINT_COLOR, -1)
                    if new_left is not None:
                        chosen_lx, chosen_ly = new_left
                        is_average = len(nearby_left_endpoints) > 1
                        chosen_color = AVG_ENDPOINT_COLOR if is_average else ENDPOINT_COLOR
                        cv2.circle(comp, (chosen_lx, chosen_ly), 4, chosen_color, -1)
                        cv2.line(comp, (orig_lx, orig_ly), (chosen_lx, chosen_ly), chosen_color, 1)

                # --- Right Toe (Index 9) ---
                orig_rx, orig_ry = tuple(original_kp_np[9].astype(int))
                if orig_rx != 0 or orig_ry != 0:
                    # Use current_toe_radius for drawing the circle
                    cv2.circle(comp, (orig_rx, orig_ry), current_toe_radius, TOE_CIRCLE_COLOR, 1)
                    for ex, ey in nearby_right_endpoints: cv2.circle(comp, (ex, ey), 3, ENDPOINT_COLOR, -1)
                    if new_right is not None:
                        chosen_rx, chosen_ry = new_right
                        is_average = len(nearby_right_endpoints) > 1
                        chosen_color = AVG_ENDPOINT_COLOR if is_average else ENDPOINT_COLOR
                        cv2.circle(comp, (chosen_rx, chosen_ry), 4, chosen_color, -1)
                        cv2.line(comp, (orig_rx, orig_ry), (chosen_rx, chosen_ry), chosen_color, 1)

            # 4. Save the composite image (unchanged)
            out_filename = os.path.basename(image_path)
            out_path = os.path.join(output_image_folder, out_filename)
            try:
                cv2.imwrite(out_path, comp)
            except Exception as e:
                 warn(f"Could not save composite image {out_path}: {e}")
            del comp

        # --- Memory Cleanup ---
        del image
        del kp_np, original_kp_np
        gc.collect()

    log("Keypoint correction processing complete.")
    if save_images_flag: log(f"Visualization images saved to {output_image_folder}")
    else: log("Visualization image saving was skipped.")

    return corrected_keypoints_data