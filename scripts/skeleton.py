import numpy as np
from skimage.morphology import skeletonize
from pipeline import warn, log
import cv2
import os


def get_skeletons(masks):
    skeletons = []
    
    for mask in masks:
        mask = mask.astype(np.uint8)

        skeleton = skeletonize(mask).astype(np.uint8)
        
        skeletons.append(skeleton)
        
        
    return skeletons


#TODO ADD GLOBAL VARIABLES INSIDE FUNC BODY AFTER FIXING DRAWINGS
#TODO ADD SAVE FUNCTION
def correct_toe_with_skeleton(yolo_predictions, skeletons):
    """
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
    # Initialize the dictionary to store the results
    corrected_keypoints_data = {}

    # Iterate through original predictions and skeletons
    # Note: No deep copy is made of yolo_predictions
    for pred, skeleton in zip(yolo_predictions, skeletons):
        image_path = pred.path # Get image path for the dictionary key

        # Load image from the prediction's path
        image = cv2.imread(image_path)
        if image is None:
            warn(f"Could not load image {image_path}. Skipping prediction.")
            continue # Skip this prediction entirely

        # Check if keypoints exist
        if not hasattr(pred, 'keypoints') or pred.keypoints is None or pred.keypoints.xy is None:
             warn(f"No keypoints found for image {image_path}. Skipping correction for this image.")
             continue

        # Get keypoints from the Results object's keypoints.xy tensor
        # Make a NumPy copy to work with and potentially modify locally
        try:
            kp_tensor = pred.keypoints.xy.cpu()  # ensure on CPU for numpy conversion
            kp_np = kp_tensor.numpy().copy() # Use .copy()
        except Exception as e:
            warn(f"Error accessing keypoints for {image_path}: {e}. Skipping correction.")
            continue

        # Handle potential extra batch dimension: assume shape (1, N, 2) -> (N,2)
        if kp_np.ndim == 3 and kp_np.shape[0] == 1:
            kp_np = kp_np[0]

        # Check if enough keypoints exist for toe correction (indices 9 and 12)
        if kp_np.shape[0] < 13:
            warn(f"Not enough keypoints ({kp_np.shape[0]}) in prediction for image {image_path}. Skipping correction.")
            # Still process the image for the dictionary, but without correction
            # Populate the dictionary with original keypoints for this image
            image_keypoints_dict = {}
            for idx, (x, y) in enumerate(kp_np):
                 image_keypoints_dict[idx] = {'x': int(x), 'y': int(y)}
            corrected_keypoints_data[image_path] = image_keypoints_dict
            # Skip the rest of the correction logic for this image
            continue


        # --- Start Correction Logic (Only if enough keypoints and skeleton exists) ---
        endpoints = []
        if skeleton is not None:
            # Compute skeleton endpoints from the non-dilated skeleton.
            # Ensure skeleton is uint8
            if skeleton.dtype != np.uint8:
                skeleton = skeleton.astype(np.uint8)

            # Ensure skeleton is binary (0 or non-zero) for endpoint logic
            binary_skeleton = (skeleton > 0).astype(np.uint8)

            if np.any(binary_skeleton): # Proceed only if skeleton is not empty
                for y, x in np.argwhere(binary_skeleton):
                    # Boundary check
                    if y == 0 or y >= binary_skeleton.shape[0] - 1 or x == 0 or x >= binary_skeleton.shape[1] - 1:
                        # Simple boundary handling: consider border pixels potential endpoints
                        # A more robust method might be needed depending on skeleton quality
                        is_endpoint = True # Tentatively
                        # Refinement: Check neighbors even for border pixels if possible
                        h, w = binary_skeleton.shape
                        y_min, y_max = max(0, y-1), min(h, y+2)
                        x_min, x_max = max(0, x-1), min(w, x+2)
                        if np.sum(binary_skeleton[y_min:y_max, x_min:x_max]) - 1 == 1:
                            is_endpoint = True
                        else:
                            # If a border pixel has >1 neighbor, maybe not a true endpoint
                            # Keep it simple for now, stick to original logic's implication
                             pass # Keep is_endpoint=True if on border for simplicity match

                    else: # Inner pixels
                        # Count neighbors in 3x3 window (excluding center)
                        neighbors = np.sum(binary_skeleton[y - 1:y + 2, x - 1:x + 2]) - 1
                        if neighbors == 1:
                            is_endpoint = True
                        else:
                            is_endpoint = False

                    if is_endpoint:
                        endpoints.append((int(x), int(y))) # Store as (x, y)

        # Helper: Process a toe coordinate (find nearby endpoint)
        def process_toe(coord):
            toe_x, toe_y = coord
            # Skip if original keypoint is at origin (likely undetected)
            if toe_x == 0 and toe_y == 0:
                return None
            if not endpoints: # No endpoints found from skeleton
                 return None

            nearby = [(ex, ey) for ex, ey in endpoints
                      if np.linalg.norm(np.array([ex, ey]) - np.array([toe_x, toe_y])) <= TOE_CIRCLE_RADIUS]

            if not nearby:
                return None # No endpoints within radius
            if len(nearby) == 1:
                return nearby[0] # Return the single endpoint tuple (x, y)

            # If multiple endpoints, average them (as per original logic)
            arr = np.array(nearby)
            avg_x = int(np.mean(arr[:, 0]))
            avg_y = int(np.mean(arr[:, 1]))
            return (avg_x, avg_y)

        # Process toe keypoints: left toe (index 12) and right toe (index 9)
        # Use the kp_np array for modification
        original_left_toe = tuple(kp_np[12].astype(int))
        original_right_toe = tuple(kp_np[9].astype(int))

        new_left = process_toe(original_left_toe)
        new_right = process_toe(original_right_toe)

        if new_left is not None:
            kp_np[12] = np.array(new_left) # Update the numpy array
            log(f"{os.path.basename(image_path)}: Left toe updated {original_left_toe} -> {new_left}")
        else:
             log(f"{os.path.basename(image_path)}: Left toe not corrected from {original_left_toe}")

        if new_right is not None:
            kp_np[9] = np.array(new_right) # Update the numpy array
            log(f"{os.path.basename(image_path)}: Right toe updated {original_right_toe} -> {new_right}")
        else:
             log(f"{os.path.basename(image_path)}: Right toe not corrected from {original_right_toe}")

        # --- End Correction Logic ---


        # --- Store final keypoints (potentially modified kp_np) in the dictionary ---
        image_keypoints_dict = {}
        for idx, (x, y) in enumerate(kp_np):
            # Store coordinates as integers
            image_keypoints_dict[idx] = {'x': int(round(x)), 'y': int(round(y))}
        corrected_keypoints_data[image_path] = image_keypoints_dict


        # --- Create and Save Composite Image (using the final kp_np) ---
        if skeleton is not None:
             # Dilate the skeleton (to make it 2-pixel wide) for visualization
             kernel = np.ones((3, 3), np.uint8)
             # Ensure skeleton is uint8 before dilate
             if skeleton.dtype != np.uint8:
                 skeleton_vis = skeleton.astype(np.uint8)
             else:
                 skeleton_vis = skeleton
             # Ensure skeleton is 0 or 255 for visualization overlay
             skeleton_vis = (skeleton_vis > 0).astype(np.uint8) * 255
             thick_skel = cv2.dilate(skeleton_vis, kernel, iterations=1)

             overlay = np.zeros_like(image)
             overlay[thick_skel == 255] = (0, 255, 0)  # green overlay (BGR)
             comp = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
        else:
             # If no skeleton, just use the original image
             comp = image.copy()


        # Draw all keypoints (using the final kp_np values)
        for idx, (x, y) in enumerate(kp_np):
            px, py = int(round(x)), int(round(y))
            # Don't draw if keypoint is at origin (likely undetected)
            if px == 0 and py == 0 and idx in [9, 12]: # Check specifically for toes if needed
                 continue
            elif px == 0 and py == 0: # Or skip any 0,0 keypoint
                 continue

            cv2.circle(comp, (px, py), 5, (0, 0, 255), -1) # Red circle
            cv2.putText(comp, str(idx), (px, py - TEXT_OFFSET_Y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Save the composite image to the OUTPUT_IMAGE_FOLDER
        out_filename = os.path.basename(image_path)
        out_path = os.path.join(OUTPUT_IMAGE_FOLDER, out_filename)
        try:
            cv2.imwrite(out_path, comp)
            # log(f"Saved composite image to {out_path}") # Keep logging minimal if preferred
        except Exception as e:
             warn(f"Could not save composite image {out_path}: {e}")


    log("Keypoint correction processing complete for all images.")
    # Return the dictionary containing the final keypoint data
    return corrected_keypoints_data

#search for skeleton renaissance

def save_correction_images(saveflag):
    pass