#IMPORTS
import sys
import pprint
import os
import shutil
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
import time
start_time = time.time()




#GLOBAL VARIABLES
SAM_MODEL_PATH = r"C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = 'vit_h'

YOLO_MODEL_PATH = r"C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\best.pt"
YOLO_INPUT_PATH = r'C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\data_2_predict_on\a'

OUTPUT_PATH = r"C:/Users/loran/OneDrive - elte.hu/ELTE/szakdolgozat/program/out"
LABELS_PATH = f"{OUTPUT_PATH}/labels"
MASKS_PATH = f"{OUTPUT_PATH}/masks"
PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

TOE_CIRCLE_RADIUS = 20
TOE_CIRCLE_COLOR = (0, 255, 255)
ENDPOINT_COLOR = (255, 0, 255)

COLOR_MAPPING = {
        "default": (0, 255, 0),       # Green
        "left_leg": (0, 255, 255),    # Yellow
        "right_leg": (255, 255, 0),   # Aqua
        "skeleton": (0, 0, 0)         # Black for the skeleton
    }

LEFT_LEG_KEYPOINTS_IDX = [7, 8, 9]
RIGHT_LEG_KEYPOINTS_IDX = [10, 11, 12]

SKELETON_CONNECTIONS = [
        (0, 2),   # Left ear to nose
        (1, 2),   # Right ear to nose
        (2, 3),   # Nose to spine
        (3, 4),   # Spine to tail 01
        (4, 5),   # Tail 01 to tail 02
        (5, 6),   # Tail 02 to tail 03
        (3, 7),   # Spine to left leg
        (7, 8),   # Left leg to left knee
        (8, 9),   # Left knee to left toe
        (3, 10),  # Spine to right leg
        (10, 11), # Right leg to right knee
        (11, 12), # Right knee to right toe
    ]

SAVE_INITIAL_PREDICTIONS = False
SAVE_MASKS = True
SAVE_SKELETON_CORRECTION_IMAGES = False


#FUNCTION DEFNINTIONS -- REWRITE EVERYTHING TO WORK
def log(msg):
    print("\n[LOG]:",msg)

def warn(msg):
    print("\n[WARNING]:",msg)

def create_folders():
    for folder in [OUTPUT_PATH, LABELS_PATH, PREDICTIONS_PATH, MASKS_PATH]:
        os.makedirs(folder, exist_ok=True)
        log(f"Created {folder} for output.")

def predict_with_yolo(input_picture_folder, conf_threshold = 0.7, batch_size = 1, stream = False):
    
    log("Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH).to("cpu")
    
    log("Performing YOLO predictions...")
    predictions = model.predict(
        source=input_picture_folder,
        conf=conf_threshold,
        batch=batch_size,
        stream=stream,
        project=PREDICTIONS_PATH,
        verbose=True
    )
    
    log("YOLO predictions complete...")
    return predictions

def extract_prediction_keypoints(predictions):
    
        # print(result.keypoints.xy)
        # print("\n")
        # print(result.keypoints.xy[0][4][1])
        # print("\n")
        # print(result.keypoints.xy[0][4][1].item())
        # print("\n")
        
        # #object 1, keypoint 5, coordinate y
        # #result.keypoints.xy[0][4][1].item() --> python float
        # print(result.keypoints.xy[0][4][1].numpy())
        # print(result.keypoints.xy[0][4][1].item())
    
    
    
    
    
    
    
    
    
    
    #65. kep 3. kulcspontjanak coordinatai: keypoints['x'][65][3]
    keypoints = {
        "x": [],
        "y": []
    }
    
    for result in predictions:
        
        
        
        
        
        
        
        result = result.summary()
            
        keypoints["x"].append(result[0]["keypoints"]["x"])
        keypoints["y"].append(result[0]["keypoints"]["y"])
        
        
        
        
        
        
    return keypoints

def save_initial_prediction(result):
    saved_path = result.save()
    
    shutil.move(saved_path, PREDICTIONS_PATH)
    
    log(f"Saved a prediction to: {PREDICTIONS_PATH}/{saved_path}")

def save_initial_predictions(predictions):
    
    if not SAVE_INITIAL_PREDICTIONS:
        return
    
    for result in predictions:
        save_initial_prediction(result)

def union_masks(masks):
    """Take a list of binary masks and return their union (logical OR)."""
    if len(masks) == 0:
        return None
    union = masks[0].copy().astype(bool)
    for m in masks[1:]:
        union = np.logical_or(union, m.astype(bool))
    return union.astype(np.uint8)

def load_sam_model():
    """
    Loads the SAM model from the specified checkpoint and moves it to the chosen device.
    Returns a SamPredictor instance ready for segmentation.
    """
    log("Loading SAM model...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
    sam.to(DEVICE)
    log(f"SAM model loaded and moved to {DEVICE}.")
    return SamPredictor(sam)
    
def get_masks_with_sam(predictions):
    
    sam_predictor = load_sam_model()
    
    log("Computing SAM masks for each image...")
    
    sam_masks = []
    
    for result in predictions:
        
        #original images!!!
        img_path = result.path
        img = cv2.imread(img_path)

        if img is None:
            warn(f"Could not load image {img_path} for SAM segmentation. Skipping.")
            sam_masks.append(None)
            continue

        if not hasattr(result, 'keypoints') or result.keypoints is None or result.keypoints.xy is None:
            warn(f"No keypoints for image {img_path}. Skipping SAM segmentation.")
            sam_masks.append(None)
            continue

        keypoints = result.keypoints.xy.cpu().numpy()
        
        if keypoints.shape[1] < 5:
            warn(f"Not enough keypoints for image {img_path}. Skipping SAM segmentation.")
            sam_masks.append(None)
            continue

        # spine es tailbase SAM-nek
        kp3 = keypoints[0][3]
        kp4 = keypoints[0][4]
        prompt1 = np.array([kp3])
        prompt2 = np.array([kp4])
        prompt3 = np.array([kp3, kp4])
        masks = []
        
        for prompt in [prompt1, prompt2, prompt3]:
            sam_predictor.set_image(img)
            pts_labels = np.ones(len(prompt))#amelyik indexen van egy kulcspont, ha ugyanitt 1 van azt jeletni hogy ez foreground
            m, _, _ = sam_predictor.predict(point_coords=prompt, point_labels=pts_labels)
            masks.append(m[0])
        union_mask = union_masks(masks)
        sam_masks.append(union_mask)

    return sam_masks

def save_sam_masks(masks):
    if SAVE_MASKS:
        for idx, union_mask in enumerate(masks):
            mask_vis = (union_mask * 255).astype(np.uint8)
            mask_save_path = os.path.join(MASKS_PATH, f"mask_{idx:05d}.png")
            cv2.imwrite(mask_save_path, mask_vis)
            log(f"Saved SAM mask: {mask_save_path}")

def get_skeletons(masks_for_every_image):
    return skeletons_for_every_image

def correct_toes_with_skeletons(predictions, skeletons_for_every_image):
    return corrected_predictions

def filter_frames_with_optical_flow(masks_for_every_image):
    return filtered_predictions

#TODO REWRITE THIS
def draw_predictions(label_folder, input_picture_folder, output_folder):    
    
    #also save images but keep them in a folder and return it
    
    for result in label_folder:
        # Get image path and base filename
        image_path = result.path
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_img_path = os.path.join(output_folder, f"{img_name}_predicted.png")
        
        # Load the image and get dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_path}")
            continue
        img_h, img_w = img.shape[:2]
        
        label_lines = []  # List to hold label lines for this image
        
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints_all = result.keypoints.xy
            try:
                keypoints_conf = result.keypoints.conf
            except AttributeError:
                keypoints_conf = None
                
            # Process each detection
            for i in range(len(result.boxes)):
                # Get bounding box in xyxy format
                bbox = result.boxes.xyxy[i].cpu().numpy().flatten()  # [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = bbox
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # Normalize bounding box values
                norm_center_x = center_x / img_w
                norm_center_y = center_y / img_h
                norm_width = width / img_w
                norm_height = height / img_h
                
                # Get object id if available, else default to 0
                try:
                    object_id = int(result.boxes.cls[i].cpu().numpy())
                except Exception:
                    object_id = 0
                
                # Get keypoints for this detection
                keypoints = keypoints_all[i].cpu().numpy()
                if keypoints_conf is not None:
                    confs = keypoints_conf[i].cpu().numpy()
                else:
                    confs = [1.0] * len(keypoints)
                
                # Build the label line with normalized values (6 decimal places)
                line = f"{object_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
                for keypoint_idx, kp in enumerate(keypoints):
                    norm_kp_x = kp[0] / img_w
                    norm_kp_y = kp[1] / img_h
                    norm_conf = confs[keypoint_idx]
                    line += f" {norm_kp_x:.6f} {norm_kp_y:.6f} {norm_conf:.6f}"
                label_lines.append(line)
                
                # Draw skeleton connections on the image
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(keypoints) and end_idx < len(keypoints):
                        start_point = keypoints[start_idx]
                        end_point = keypoints[end_idx]
                        if (start_point[0] == 0 and start_point[1] == 0) or (end_point[0] == 0 and end_point[1] == 0):
                            continue
                        start_pt = (int(start_point[0]), int(start_point[1]))
                        end_pt = (int(end_point[0]), int(end_point[1]))
                        if start_idx in left_leg_keypoints or end_idx in left_leg_keypoints:
                            line_color = darker_color(color_mapping["left_leg"])
                        elif start_idx in right_leg_keypoints or end_idx in right_leg_keypoints:
                            line_color = darker_color(color_mapping["right_leg"])
                        else:
                            line_color = color_mapping["skeleton"]
                        cv2.line(img, start_pt, end_pt, line_color, 2)
                # Draw keypoints and annotate them
                for keypoint_idx, keypoint in enumerate(keypoints):
                    x, y = int(keypoint[0]), int(keypoint[1])
                    if x == 0 and y == 0:
                        continue
                    if keypoint_idx in left_leg_keypoints:
                        color = color_mapping["left_leg"]
                    elif keypoint_idx in right_leg_keypoints:
                        color = color_mapping["right_leg"]
                    else:
                        color = color_mapping["default"]
                    cv2.circle(img, (x, y), 3, color, -1)
                    cv2.putText(img, str(keypoint_idx), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the annotated image
        cv2.imwrite(output_img_path, img)
        print(f"Image saved to {output_img_path}")
        
        # Write the normalized label file for this image
        label_file_path = os.path.join(labels_folder, f"{img_name}.txt")
        with open(label_file_path, "w") as f:
            for line in label_lines:
                f.write(line + "\n")
        print(f"Label file saved to {label_file_path}")
        
        
    return predicted_images

#update this to work with a list instead
def create_video_from_images(predicted_images, output_video_path, fps=30):
    # # Get a list of image files in the folder, explicitly excluding subdirectories
    # image_files = predicted_images
    
    # # Sort files in numerical order using the custom sort function
    # image_files = sort_nicely(image_files)
    
    # if not image_files:
    #     print("No images found in the folder.")
    #     return
    
    # # Load the first image to get video properties
    # first_image_path = os.path.join(folder_path, image_files[0])
    # first_image = cv2.imread(first_image_path)
    # if first_image is None:
    #     print(f"Error loading the first image {first_image_path}")
    #     return
    
    # height, width, _ = first_image.shape

    # # Initialize video writer (with frame rate and proper codec)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # # Process each image and write it to the video
    # for idx, image_file in enumerate(image_files):
    #     image_path = os.path.join(folder_path, image_file)
    #     image = cv2.imread(image_path)
        
    #     if image is None:
    #         print(f"Error loading image {image_path}")
    #         continue
        
    #     # Resize image to match the first image's resolution (to avoid dimension mismatches)
    #     resized_image = cv2.resize(image, (width, height))

    #     # Write the resized image as a frame to the video
    #     out.write(resized_image)

    #     # Logging to verify the correct order of processing
    #     print(f"Writing frame {idx + 1} of {len(image_files)}: {image_file}")

    # # Release the video writer
    # out.release()
    # print(f"Video saved at {output_video_path}")
    pass

def plot_results():
    pass










if __name__ == "__main__":
    create_folders()
    


#YOLO INFERENCE
    #1o IMAGE MODE
    #predictions = predict_with_yolo(YOLO_INPUT_PATH)
    
    #SINGLE IMAGE MODE
    predictions = predict_with_yolo(r'C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\data_2_predict_on\a\frame_00101.png')
    
    save_initial_predictions(predictions)
            


#SAM MASK IDENTIFICATION
    masks = get_masks_with_sam(predictions)
    
    save_sam_masks(masks)






#OPTICAL FLOW FILTERING

#SKELETONIZE TOE CORRECTION

#SUMMING UP DATA - PLOTS, VIDEOS


    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nTotal execution time: {runtime:.2f} seconds")