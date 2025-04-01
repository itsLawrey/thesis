import os
import pandas as pd
import json
import shutil
import cv2 # To get image dimensions
from pathlib import Path # For easier path handling
import time # Added for potential timing/logging
import random
import yaml # Requires PyYAML (pip install pyyaml)
from logs import log, warn



# --- Logging Functions ---
def log(msg): print(f"[LOG] {msg}")
def warn(msg): print(f"[WARNING] {msg}")

def create_and_split_yolo_dataset(input_folder_list, output_split_dir, train_ratio=0.7, seed=42, save_consolidated_json=False):
    """
    Processes biologist data, splits it, and saves directly into train/val/test
    YOLO dataset structure, minimizing intermediate disk I/O.

    Args:
        input_folder_list (list): List of paths to the input folders containing
                                  images and a single CSV file each.
        output_split_dir (str or Path): The base directory where 'train', 'val',
                                      'test', and 'config' folders will be created.
        train_ratio (float): Proportion of the dataset for training (0.0 to 1.0).
                             Remainder split equally between validation and test.
        seed (int): Random seed for shuffling.
        save_consolidated_json (bool): If True, saves a 'consolidated_annotations.json'
                                       in the output_split_dir (contains all data before split).

    Returns:
        bool: True if the process was successful, False otherwise.
        dict: Paths needed for data.yaml if successful, None otherwise.
    """
    # --- Constants ---
    YOLO_CLASS_ID = 0 # Assuming 'rat' is the only class, index 0
    VISIBILITY_ABSENT_OR_INVALID = 0
    VISIBILITY_PRESENT = 2
    YOLO_VISIBILITY_NOT_LABELED = 0
    YOLO_VISIBILITY_VISIBLE = 2
    HEADER_ROWS_TO_SKIP = 3 # Skip scorer, bodyparts, coords rows
    log(f"\n--- Starting Integrated Dataset Creation and Splitting ---")
    output_path = Path(output_split_dir)
    consolidated_json_path = output_path / "consolidated_annotations.json" # Optional JSON path

    # --- Data storage before splitting ---
    processed_data_items = [] # List to hold dicts of processed data before saving
    master_annotations_json = {} # For optional consolidated JSON
    master_unique_bodyparts = None
    total_skipped_entries = 0

    # --- Phase 1: Process all data and store in memory ---
    log("--- Phase 1: Processing all input data ---")
    for input_folder_path in input_folder_list:
        log(f"\n--- Processing Source Folder: {input_folder_path} ---")
        input_folder = Path(input_folder_path)
        if not input_folder.is_dir():
            warn(f"Input path is not a valid directory: {input_folder_path}. Skipping.")
            continue

        input_folder_name = input_folder.name

        # Find and Read CSV
        csv_files = list(input_folder.glob('*.csv'))
        if not csv_files: warn(f"No CSV file found in {input_folder_path}. Skipping folder."); continue
        if len(csv_files) > 1: warn(f"Multiple CSV files found. Using: {csv_files[0].name}")
        csv_path = csv_files[0]
        log(f"Processing CSV: {csv_path.name}")

        try:
            # Read and verify bodyparts
            bodyparts_df_row = pd.read_csv(csv_path, header=None, skiprows=1, nrows=1)
            raw_bodyparts_header = [str(bp).strip() for bp in bodyparts_df_row.iloc[0, 3:].tolist()]
            current_unique_bodyparts = list(dict.fromkeys(raw_bodyparts_header))
            if not current_unique_bodyparts: raise ValueError("Could not extract body part names.")

            if master_unique_bodyparts is None:
                master_unique_bodyparts = current_unique_bodyparts
                log(f"Established master bodypart list: {master_unique_bodyparts}")
            elif master_unique_bodyparts != current_unique_bodyparts:
                warn(f"Inconsistent bodyparts in {csv_path.name}. Expected: {master_unique_bodyparts}, Found: {current_unique_bodyparts}. Skipping folder.")
                continue

            # Read main data
            df = pd.read_csv(csv_path, skiprows=HEADER_ROWS_TO_SKIP, header=None)
            if len(df.columns) < 5: warn(f"CSV {csv_path.name} has < 5 columns. Cannot process."); continue
            df = df.dropna(subset=[df.columns[2]])
            df = df[df.iloc[:, 2] != '']

        except Exception as e:
            warn(f"Error reading CSV file {csv_path}: {e}. Skipping folder.")
            continue

        log(f"Processing {len(df)} entries from CSV for folder {input_folder_name}...")
        folder_skipped_count = 0

        # Process each row (image annotation)
        for index, row in df.iterrows():
            try: original_filename = str(row.iloc[2]).strip()
            except IndexError: warn(f"Skipping row {index}: Cannot access filename."); folder_skipped_count += 1; continue
            if not original_filename or not original_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                warn(f"Row {index}: Invalid filename '{original_filename}'. Skipping."); folder_skipped_count += 1; continue

            source_image_path = input_folder / original_filename
            if not source_image_path.is_file():
                warn(f"Image file not found: {source_image_path}. Skipping row {index}."); folder_skipped_count += 1; continue

            prefixed_filename = f"{input_folder_name}_{original_filename}"
            label_filename = Path(prefixed_filename).with_suffix('.txt').name

            # --- Get Image Dimensions (Read temporarily) ---
            try:
                # We need dimensions for normalization, so we read the source image
                img = cv2.imread(str(source_image_path))
                if img is None: raise ValueError("cv2.imread returned None")
                img_h, img_w = img.shape[:2]
                if img_h <= 0 or img_w <= 0: raise ValueError(f"Invalid image dimensions: H={img_h}, W={img_w}")
            except Exception as e:
                warn(f"Error reading image dimensions for {source_image_path}: {e}. Skipping row {index}.")
                folder_skipped_count += 1
                continue

            # --- Process Keypoints and Calculate BBox ---
            keypoints_for_yolo = []
            keypoints_for_json = []
            min_x_pix, max_x_pix = float('inf'), float('-inf')
            min_y_pix, max_y_pix = float('inf'), float('-inf')
            valid_kps_found = False
            col_idx = 3 # Reset for each row

            for bp_name in master_unique_bodyparts:
                if col_idx + 1 >= len(row):
                    warn(f"Row {index} ({original_filename}) shorter than expected for '{bp_name}'. Filling remaining as not labeled.")
                    remaining_count = len(master_unique_bodyparts) - len(keypoints_for_yolo)
                    for i in range(remaining_count):
                        bp_add = master_unique_bodyparts[len(keypoints_for_yolo) + i]
                        keypoints_for_yolo.append((0.0, 0.0, YOLO_VISIBILITY_NOT_LABELED))
                        keypoints_for_json.append({"bodypart": bp_add, "x": 0.0, "y": 0.0, "v": YOLO_VISIBILITY_NOT_LABELED})
                    break # Exit inner loop

                x_val, y_val = row.iloc[col_idx], row.iloc[col_idx + 1]
                try:
                    x_pix, y_pix = float(x_val), float(y_val)
                    if pd.isna(x_pix) or pd.isna(y_pix): raise ValueError("NaN coordinate")
                    input_v = VISIBILITY_PRESENT
                except (ValueError, TypeError):
                    x_pix, y_pix, input_v = 0.0, 0.0, VISIBILITY_ABSENT_OR_INVALID

                yolo_v = YOLO_VISIBILITY_VISIBLE if input_v == VISIBILITY_PRESENT else YOLO_VISIBILITY_NOT_LABELED
                norm_x = x_pix / img_w if input_v == VISIBILITY_PRESENT else 0.0
                norm_y = y_pix / img_h if input_v == VISIBILITY_PRESENT else 0.0

                keypoints_for_yolo.append((norm_x, norm_y, yolo_v))
                keypoints_for_json.append({"bodypart": bp_name, "x": norm_x, "y": norm_y, "v": yolo_v})

                if input_v == VISIBILITY_PRESENT:
                    min_x_pix, max_x_pix = min(min_x_pix, x_pix), max(max_x_pix, x_pix)
                    min_y_pix, max_y_pix = min(min_y_pix, y_pix), max(max_y_pix, y_pix)
                    valid_kps_found = True
                col_idx += 2

            # Calculate BBox
            if valid_kps_found:
                padding_x = (max_x_pix - min_x_pix) * 0.05
                padding_y = (max_y_pix - min_y_pix) * 0.05
                min_x_pix, max_x_pix = max(0.0, min_x_pix - padding_x), min(float(img_w), max_x_pix + padding_x)
                min_y_pix, max_y_pix = max(0.0, min_y_pix - padding_y), min(float(img_h), max_y_pix + padding_y)
                bb_w_pix, bb_h_pix = max(0.0, max_x_pix - min_x_pix), max(0.0, max_y_pix - min_y_pix)
                bb_cx_pix, bb_cy_pix = min_x_pix + bb_w_pix / 2, min_y_pix + bb_h_pix / 2
                norm_cx, norm_cy = bb_cx_pix / img_w, bb_cy_pix / img_h
                norm_w, norm_h = bb_w_pix / img_w, bb_h_pix / img_h
                norm_cx, norm_cy = max(0.0, min(1.0, norm_cx)), max(0.0, min(1.0, norm_cy))
                norm_w, norm_h = max(0.0, min(1.0, norm_w)), max(0.0, min(1.0, norm_h))
                bbox_json = {"center": {"x": norm_cx, "y": norm_cy}, "width": norm_w, "height": norm_h}
            else:
                warn(f"No valid keypoints for {prefixed_filename}. Using zero bbox.")
                norm_cx, norm_cy, norm_w, norm_h = 0.0, 0.0, 0.0, 0.0
                bbox_json = {"center": {"x": 0.0, "y": 0.0}, "width": 0.0, "height": 0.0}

            # Format YOLO String
            yolo_elements = [str(YOLO_CLASS_ID), f"{norm_cx:.6f}", f"{norm_cy:.6f}", f"{norm_w:.6f}", f"{norm_h:.6f}"]
            for kx, ky, kv in keypoints_for_yolo:
                kx, ky = max(0.0, min(1.0, kx)), max(0.0, min(1.0, ky))
                yolo_elements.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
            yolo_string = " ".join(yolo_elements)

            # --- Store processed data in memory ---
            processed_data_items.append({
                'source_path': source_image_path,
                'image_name': prefixed_filename,
                'label_name': label_filename,
                'yolo_data': yolo_string
            })

            # Store for optional consolidated JSON
            if save_consolidated_json:
                 if prefixed_filename in master_annotations_json: warn(f"Duplicate key '{prefixed_filename}' encountered for JSON.")
                 master_annotations_json[prefixed_filename] = {"bodyparts": keypoints_for_json, "bounding_box": bbox_json}

        # End row loop
        total_skipped_entries += folder_skipped_count
        log(f"Finished processing folder {input_folder_name}. Valid entries found: {len(df)-folder_skipped_count}, Skipped: {folder_skipped_count}")
    # End folder loop
    log("--- Phase 1 Finished: All input data processed ---")

    # --- Validation before splitting ---
    if not processed_data_items:
        warn("No valid data items were processed. Cannot proceed with splitting. Aborting.")
        return False, None
    if not master_unique_bodyparts:
         warn("Master bodypart list could not be established. Aborting.")
         return False, None

    log(f"Total valid data items to split: {len(processed_data_items)}")
    log(f"Total entries skipped across all folders: {total_skipped_entries}")

    # --- Phase 2: Shuffle, Split, and Save ---
    log("\n--- Phase 2: Shuffling, Splitting, and Saving Data ---")
    random.seed(seed)
    random.shuffle(processed_data_items)

    # Calculate split counts
    total_files = len(processed_data_items)
    train_count = int(total_files * train_ratio)
    remaining_count = total_files - train_count
    val_count = remaining_count // 2
    test_count = remaining_count - val_count
    log(f"Splitting into: Train={train_count}, Validation={val_count}, Test={test_count}")
    assert train_count + val_count + test_count == total_files

    # Define output directories
    split_dirs_info = {
        "train": {"start": 0, "end": train_count, "path": output_path / "train"},
        "val":   {"start": train_count, "end": train_count + val_count, "path": output_path / "val"},
        "test":  {"start": train_count + val_count, "end": total_files, "path": output_path / "test"}
    }

    # Create directories
    try:
        output_path.mkdir(parents=True, exist_ok=True) # Ensure base output dir exists
        for split_name, info in split_dirs_info.items():
            (info["path"] / "images").mkdir(parents=True, exist_ok=True)
            (info["path"] / "labels").mkdir(parents=True, exist_ok=True)
        log("Created train/val/test directory structures.")
    except OSError as e:
        warn(f"Error creating output directories: {e}. Aborting.")
        return False, None

    # Save files to respective splits
    files_copied = 0
    labels_written = 0
    copy_errors = 0
    write_errors = 0

    for i, item in enumerate(processed_data_items):
        # Determine split
        split_name = None
        if i < split_dirs_info["train"]["end"]: split_name = "train"
        elif i < split_dirs_info["val"]["end"]: split_name = "val"
        else: split_name = "test"

        dest_img_dir = split_dirs_info[split_name]["path"] / "images"
        dest_lbl_dir = split_dirs_info[split_name]["path"] / "labels"
        dest_img_path = dest_img_dir / item['image_name']
        dest_lbl_path = dest_lbl_dir / item['label_name']

        # Copy image
        try:
            shutil.copy2(item['source_path'], dest_img_path)
            files_copied += 1
        except Exception as e:
            warn(f"Error copying image {item['source_path']} to {dest_img_path}: {e}")
            copy_errors += 1
            continue # Skip label writing if image copy failed

        # Write label
        try:
            with open(dest_lbl_path, 'w') as f:
                f.write(item['yolo_data'])
            labels_written += 1
        except IOError as e:
            warn(f"Error writing label file {dest_lbl_path}: {e}")
            write_errors += 1
            # Optionally remove the copied image if label write fails
            # if dest_img_path.exists(): try: dest_img_path.unlink() except OSError: pass

    log(f"Finished saving files. Images copied: {files_copied}, Labels written: {labels_written}")
    if copy_errors > 0 or write_errors > 0:
        warn(f"Encountered errors during saving: {copy_errors} image copy errors, {write_errors} label write errors.")
        # Decide if this is a critical failure? Let's return True but warn.

    # --- Phase 3: Create data.yaml and Save Optional JSON ---
    log("\n--- Phase 3: Finalizing ---")

    # Prepare paths for data.yaml
    yaml_paths = {
        'path': str(output_path.resolve()), # Absolute path
        'train': str(Path("train") / "images"),
        'val': str(Path("val") / "images"),
        'test': str(Path("test") / "images")
    }

    # Create data.yaml
    yaml_success = create_data_yaml(output_path, yaml_paths, master_unique_bodyparts)
    if not yaml_success:
        warn("Failed to create data.yaml file.")
        # Return False if YAML creation is critical
        # return False, None

    # Save Consolidated JSON (Optional)
    json_saved = True
    if save_consolidated_json:
        if master_annotations_json:
            log(f"Saving consolidated annotations ({len(master_annotations_json)} entries)...")
            try:
                with open(consolidated_json_path, 'w') as f:
                    json.dump(master_annotations_json, f, indent=4)
                log(f"Successfully saved consolidated JSON to {consolidated_json_path}")
            except Exception as e:
                warn(f"Error writing consolidated JSON file {consolidated_json_path}: {e}")
                json_saved = False
        else:
            log("No annotations were processed, consolidated JSON file not saved.")

    log(f"\n--- Integrated Dataset Creation and Splitting Finished ---")
    final_success = (copy_errors == 0 and write_errors == 0 and yaml_success and json_saved) # Define success criteria
    return final_success, yaml_paths


def create_data_yaml(output_dir, yaml_paths, bodypart_names):
    """
    Creates the data.yaml file for YOLO pose training. (Now includes bodypart names)

    Args:
        output_dir (str or Path): The base output directory where 'config' goes.
        yaml_paths (dict): Paths dictionary from the main function.
        bodypart_names (list): List of unique bodypart names in order.

    Returns:
        bool: True if YAML creation was successful, False otherwise.
    """
    log(f"\n--- Creating data.yaml ---")
    output_path = Path(output_dir)
    config_dir = output_path / "config"
    yaml_file_path = config_dir / "data.yaml"

    if not yaml_paths: warn("YAML paths data missing."); return False
    if not bodypart_names: warn("Bodypart names missing."); return False

    # Define YAML Content
    data_yaml_content = {
        'path': yaml_paths['path'],
        'train': yaml_paths['train'],
        'val': yaml_paths['val'],
        'test': yaml_paths['test'],
        'kpt_shape': [len(bodypart_names), 3], # Use actual count
        'flip_idx': [1, 0, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9], # IMPORTANT: Verify this matches your bodypart order!
        'names': { 0: 'rat' }
        # Optional: Add keypoint names for clarity/visualization tools
        # 'keypoints': { 'names': {i: name for i, name in enumerate(bodypart_names)}, 'skeleton': [] } # Define skeleton if needed
    }

    # --- Verify flip_idx length ---
    if len(data_yaml_content['flip_idx']) != len(bodypart_names):
        warn(f"CRITICAL WARNING: Length of 'flip_idx' ({len(data_yaml_content['flip_idx'])}) does not match the number of keypoints ({len(bodypart_names)})!")
        warn("You MUST manually correct 'flip_idx' in the generated data.yaml based on your specific keypoint order:")
        warn(f"Keypoints: {bodypart_names}")
        # Example: If 'left ear' is index 0 and 'right ear' is index 1, flip_idx should start [1, 0, ...]
        # You need to define the corresponding index for *each* keypoint after flipping.

    # Create Config Directory and Write YAML
    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        log(f"Ensured config directory exists: {config_dir}")
        with open(yaml_file_path, 'w') as f:
            yaml.dump(data_yaml_content, f, sort_keys=False, default_flow_style=None)
        log(f"Successfully created data.yaml at: {yaml_file_path}")
        log("YAML Content:")
        print("-" * 20)
        with open(yaml_file_path, 'r') as f: print(f.read())
        print("-" * 20)
    except Exception as e:
        warn(f"An error occurred during YAML creation: {e}")
        return False

    log("--- data.yaml Creation Finished ---")
    return True


# --- Example Usage ---
if __name__ == "__main__":
    # Input folders from biologists
    BIOLOGIST_DATA_FOLDER_LIST = [
        r"C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\in\demo_from_scratch\Rat3_week1",
        r'C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\in\demo_from_scratch\Rat31_week19'
    ]
    # Directory where the final split dataset (train/val/test/config) will be created
    FINAL_SPLIT_OUTPUT_DIR = r"C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\models\training_data\datasets\ds_integrated_1"

    # Ratio for the training set (e.g., 80%)
    TRAIN_SPLIT_RATIO = 0.8

    # Random seed for reproducible splits
    RANDOM_SEED = 42

    # --- Run the integrated processing, splitting, and saving ---
    overall_success, _ = create_and_split_yolo_dataset(
        input_folder_list=BIOLOGIST_DATA_FOLDER_LIST,
        output_split_dir=FINAL_SPLIT_OUTPUT_DIR,
        train_ratio=TRAIN_SPLIT_RATIO,
        seed=RANDOM_SEED,
        save_consolidated_json=False # Set to True if you want the extra JSON file
    )

    # --- Final Status ---
    if overall_success:
        log("\nIntegrated dataset creation and splitting completed successfully!")
        log(f"Split dataset located at: {Path(FINAL_SPLIT_OUTPUT_DIR).resolve()}")
    else:
        log("\nProcess finished with errors or warnings.")

    log("Script finished.")