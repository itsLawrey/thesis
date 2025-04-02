import cv2
import numpy as np

def compute_optical_flow(prev_img, next_img):
    """Compute dense optical flow using Farneback method."""
    return cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def compute_flow_magnitude(flow):
    """Compute the magnitude (speed) of the optical flow vectors."""
    return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

def make_frame_decisions(yolo_predictions, sam_masks, flow_diff_threshold):
    """
    Determines which frames should be kept or discarded based on movement detection.
    
    This implementation uses a tolerance counter to allow up to TOLERANCE consecutive out-of-spec frames
    in the 25-frame buffer without clearing it.
    
    :param yolo_predictions: List of YOLO detection results.
    :param sam_masks: List of segmentation masks corresponding to each frame.
    :param flow_diff_threshold: Threshold for distinguishing movement.
    :return: Dictionary mapping frame paths to "keep" or "discard".
    """
    BUF_SIZE = 25
    TOLERANCE = 5  # Maximum consecutive out-of-spec frames allowed in the buffer
    frame_decision = {}  # Stores {frame_path: "keep" / "discard"}
    buffer = []  # Stores tuples (img_path, decision) for the current buffer
    tolerance_counter = 0  # Count of consecutive out-of-spec frames in the buffer
    state = "rat_still"  # Start in the moving still
    prev_gray = None  # Previous frame in grayscale
    prev_union_mask = None  # Previous frame's SAM mask
    
    print("[INFO] Processing optical flow and marking frames to keep or discard...")
    for idx, result in enumerate(yolo_predictions):
        img_path = result.path
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Could not load image {img_path}. Skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get SAM mask
        union_mask = sam_masks[idx]
        if union_mask is None:
            print(f"[WARNING] No SAM mask for image {img_path}. Skipping flow computation.")
            frame_decision[img_path] = "discard"
            prev_gray = gray
            prev_union_mask = None
            # Clear the buffer as well, since we don't want to accumulate uncertainty.
            buffer.clear()
            tolerance_counter = 0
            continue
        
        if prev_gray is not None and prev_union_mask is not None:
            flow = compute_optical_flow(prev_gray, gray)
            flow_mag = compute_flow_magnitude(flow)
            
            # Compute average optical flow inside and outside the previous mask
            rat_flow = np.mean(flow_mag[prev_union_mask > 0]) if np.count_nonzero(prev_union_mask) > 0 else 0
            bg_flow = np.mean(flow_mag[prev_union_mask == 0]) if np.count_nonzero(prev_union_mask == 0) > 0 else 0
            diff_flow = rat_flow - bg_flow
            
            print(f"Processing frame {idx}")
            print(f"Current state: {state}")
            print(f"diff_flow for frame {idx}: {diff_flow}")
            
            # ---------------------------
            # In state: rat_moving
            # ---------------------------
            if state == "rat_moving":
                # Default decision in rat_moving is to "keep" the frame.
                frame_decision[img_path] = "keep"
                
                # Expected condition for discarding: diff_flow <= threshold.
                if diff_flow <= flow_diff_threshold:
                    # In-spec for discarding: add to buffer as "discard" and reset tolerance.
                    buffer.append((img_path, "discard"))
                    tolerance_counter = 0
                    print(f"{img_path} added to buffer with value 'discard' in state {state}")
                else:
                    # Out-of-spec: this frame would normally be "keep"
                    # Instead of immediately clearing the buffer, increment tolerance.
                    buffer.append((img_path, "discard"))
                    tolerance_counter += 1
                    print(f"{img_path} added to buffer (out-of-spec) in state {state}; tolerance count: {tolerance_counter}")
                    # If tolerance exceeded, clear the buffer.
                    if tolerance_counter > TOLERANCE:
                        print(f"Tolerance exceeded in state {state} at frame {idx}, clearing buffer.")
                        buffer.clear()
                        tolerance_counter = 0
                
                # If the buffer has reached BUF_SIZE, we consider this a valid sequence of discardable frames.
                if len(buffer) >= BUF_SIZE:
                    print(f"Buffer full in state {state} at frame {idx}. Marking buffered frames as 'discard' and transitioning to rat_still.")
                    for path, _ in buffer:
                        frame_decision[path] = "discard"
                        print(f"{path} retroactively marked as 'discard'.")
                    buffer.clear()
                    tolerance_counter = 0
                    state = "rat_still"
                    print(f"[INFO] Transition to rat_still at frame {idx}.")
            
            # ---------------------------
            # In state: rat_still
            # ---------------------------
            elif state == "rat_still":
                # Default decision in rat_still is to "discard" the frame.
                frame_decision[img_path] = "discard"
                
                # Expected condition for keeping: diff_flow > threshold.
                if diff_flow > flow_diff_threshold:
                    # In-spec for keeping: add to buffer as "keep" and reset tolerance.
                    buffer.append((img_path, "keep"))
                    tolerance_counter = 0
                    print(f"{img_path} added to buffer with value 'keep' in state {state}")
                else:
                    # Out-of-spec: this frame would normally be "discard"
                    buffer.append((img_path, "keep"))
                    tolerance_counter += 1
                    print(f"{img_path} added to buffer (out-of-spec) in state {state}; tolerance count: {tolerance_counter}")
                    if tolerance_counter > TOLERANCE:
                        print(f"Tolerance exceeded in state {state} at frame {idx}, clearing buffer.")
                        buffer.clear()
                        tolerance_counter = 0
                
                if len(buffer) >= BUF_SIZE:
                    print(f"Buffer full in state {state} at frame {idx}. Marking buffered frames as 'keep' and transitioning to rat_moving.")
                    for path, _ in buffer:
                        frame_decision[path] = "keep"
                        print(f"{path} retroactively marked as 'keep'.")
                    buffer.clear()
                    tolerance_counter = 0
                    state = "rat_moving"
                    print(f"[INFO] Transition to rat_moving at frame {idx}.")
        else:
            # For the very first frame, we mark it as "discard" (since no comparison is possible)
            frame_decision[img_path] = "discard"
        
        prev_gray = gray
        prev_union_mask = union_mask

    # At end of video, handle any remaining frames in the buffer:
    if buffer:
        if state == "rat_moving":
            # If we're in rat_moving, mark remaining buffered frames as "discard"
            for path, _ in buffer:
                frame_decision[path] = "discard"
            print(f"[INFO] End of video: remaining {len(buffer)} frames in buffer marked as 'discard' (rat_moving state).")
        elif state == "rat_still":
            # If we're in rat_still, mark remaining buffered frames as "keep"
            for path, _ in buffer:
                frame_decision[path] = "keep"
            print(f"[INFO] End of video: remaining {len(buffer)} frames in buffer marked as 'keep' (rat_still state).")
        buffer.clear()
        tolerance_counter = 0

    return frame_decision # Dictionary to store frame decisions ("keep" or "discard")
 
 # --------------------------------------------------------------------------

def discard_unwanted_frames(yolo_predictions, sam_masks, frame_decision):
    """
    Filters YOLO predictions and SAM masks based on a pre-computed decision dictionary.

    Iterates through the input lists and creates new lists containing only the
    elements corresponding to paths marked "keep" in the frame_decision dictionary.
    Preserves the original relative order and object structure.

    Args:
        yolo_predictions: The original list of YOLO detection result objects.
                          Each object must have a '.path' attribute.
        sam_masks: The original list of segmentation masks (e.g., NumPy arrays),
                   corresponding element-wise to yolo_predictions.
        frame_decision (dict): A dictionary mapping image paths (str) to
                               decision strings ("keep" or "discard"). This is
                               typically generated by the filter_frames function.

    Returns:
        tuple: A tuple containing two new lists:
            - filtered_yolo_predictions: List containing only the YOLO results
                                         for frames marked "keep".
            - filtered_sam_masks: List containing only the SAM masks
                                  corresponding to the kept frames.
            Returns ([], []) if input lists are empty or if errors occur.

    Raises:
        ValueError: If input lists `yolo_predictions` and `sam_masks` have
                    mismatched non-zero lengths.
    """
    if not yolo_predictions or not sam_masks:
        print("[INFO] (discard_unwanted_frames) Input lists (yolo_predictions or sam_masks) are empty. Returning empty lists.")
        return [], []

    # Check for length mismatch - crucial for index alignment
    if len(yolo_predictions) != len(sam_masks):
        raise ValueError(f"Input list length mismatch: "
                         f"yolo_predictions ({len(yolo_predictions)}) != sam_masks ({len(sam_masks)})")

    filtered_yolo_predictions = []
    filtered_sam_masks = []
    kept_count = 0
    discarded_count = 0
    missing_path_count = 0
    missing_decision_count = 0

    print("[INFO] (discard_unwanted_frames) Filtering predictions and masks based on decisions...")

    for i, yolo_result in enumerate(yolo_predictions):
        # --- Get Path ---
        if not hasattr(yolo_result, 'path'):
            print(f"[WARNING] (discard_unwanted_frames) Item at index {i} in yolo_predictions lacks '.path' attribute. Discarding.")
            missing_path_count += 1
            discarded_count += 1
            continue # Skip this item

        img_path = yolo_result.path

        # --- Get Decision ---
        # Use .get() for safety: defaults to "discard" if path not in dictionary
        decision = frame_decision.get(img_path, "discard")

        if img_path not in frame_decision:
             # Only print warning if path was expected but missing
             # print(f"[WARNING] (discard_unwanted_frames) Path '{img_path}' not found in frame_decision dictionary. Defaulting to discard.")
             missing_decision_count +=1


        # --- Append if "keep" ---
        if decision == "keep":
            # Double-check index validity for sam_masks (should be fine due to initial length check)
            if i < len(sam_masks):
                filtered_yolo_predictions.append(yolo_result)
                filtered_sam_masks.append(sam_masks[i])
                kept_count += 1
            else:
                # This should not happen if the initial length check passed
                print(f"[ERROR] (discard_unwanted_frames) Index {i} out of bounds for sam_masks despite initial check for path {img_path}. Discarding.")
                discarded_count += 1
        elif decision == "discard":
            discarded_count += 1
        else:
            # Handle unexpected decision values
            print(f"[WARNING] (discard_unwanted_frames) Invalid decision '{decision}' found for path '{img_path}'. Discarding.")
            discarded_count += 1

    print(f"[INFO] (discard_unwanted_frames) Filtering complete.")
    print(f"  Original items: {len(yolo_predictions)}")
    print(f"  Items kept: {kept_count}")
    print(f"  Items discarded: {discarded_count}")
    if missing_path_count > 0:
        print(f"  Items discarded due to missing '.path': {missing_path_count}")
    if missing_decision_count > 0:
         # This count includes items discarded because their path wasn't in the decision dict
         print(f"  Items discarded due to missing decision entry: {missing_decision_count}")


    # Final sanity check on output lengths
    if len(filtered_yolo_predictions) != len(filtered_sam_masks):
         print(f"[CRITICAL WARNING] (discard_unwanted_frames) Final filtered list lengths do not match! "
               f"Preds: {len(filtered_yolo_predictions)}, Masks: {len(filtered_sam_masks)}")
         # Depending on severity, you might want to return empty lists or raise an error here
         # return [], [] # Safer option

    return filtered_yolo_predictions, filtered_sam_masks

def filter_data(yolo_predictions, sam_masks):
    
    flow_diff_threshold = 0.2
    
    decision_dict = make_frame_decisions(yolo_predictions, sam_masks, flow_diff_threshold)
    
    f_yolo, f_masks = discard_unwanted_frames(yolo_predictions, sam_masks, decision_dict)
    
    return f_yolo, f_masks