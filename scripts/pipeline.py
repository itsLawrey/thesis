#IMPORTS
import os
import torch
import time
start_time = time.time()
import predict
import masks
import skeleton
import evaluate
import opticalflow
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################

#GLOBAL VARIABLES
SAM_MODEL_PATH = r"C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = 'vit_h'

YOLO_MODEL_PATH = r"C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\best.pt"
YOLO_INPUT_PATH = r'C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\data_2_predict_on\a'

OUTPUT_PATH = r"C:/Users/loran/OneDrive - elte.hu/ELTE/szakdolgozat/program/out"
LABELS_PATH = f"{OUTPUT_PATH}/labels"
MASKS_PATH = f"{OUTPUT_PATH}/masks"
CORRECTED_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/corrected"
ORIGINAL_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/original"
VIDEO_PREDICTION_PATH = f"{OUTPUT_PATH}/predictions/videos"
SKELETONS_PATH = f"{OUTPUT_PATH}/skeletons"

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

#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################

#FUNCTION DEFNINTIONS
def log(msg):
    print("\n[LOG]:",msg)

def warn(msg):
    print("\n[WARNING]:",msg)

def create_folders():
    for folder in [OUTPUT_PATH, LABELS_PATH, PREDICTIONS_PATH, MASKS_PATH, SKELETONS_PATH]:
        os.makedirs(folder, exist_ok=True)
        log(f"Created {folder} for output.")


#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################

if __name__ == "__main__":
    
    create_folders()
    

#YOLO INFERENCE
    #SINGLE IMAGE MODE
    predictions = predict.predict_with_yolo(r'C:\Users\loran\OneDrive - elte.hu\ELTE\szakdolgozat\program\data_2_predict_on\a\frame_00101.png', YOLO_MODEL_PATH)
    
    predict.save_initial_predictions(predictions, SAVE_INITIAL_PREDICTIONS, ORIGINAL_PREDICTIONS_PATH)

#SAM MASK IDENTIFICATION
    prediction_masks = masks.get_masks_with_sam(predictions)
    
    masks.save_sam_masks(prediction_masks, SAVE_MASKS, MASKS_PATH)

#OPTICAL FLOW FILTERING
    #TODO FILTER FRAMEZZZ
    
#SKELETONIZE TOE CORRECTION
    #TODO HANDLE FACT THAT SOME FRAMES GOT FILTERED...
    skeletons = skeleton.get_skeletons(prediction_masks)
    
    corrected_predictions_dict = skeleton.correct_toes_skeleton(predictions,skeletons, SAVE_SKELETON_CORRECTION_IMAGES, SKELETONS_PATH)

#SUMMING UP DATA - PLOTS, VIDEOS

    evaluate.draw_corrected_predictions(corrected_predictions_dict, CORRECTED_PREDICTIONS_PATH)
    
    evaluate.draw_uncorrected_predictions(predictions, ORIGINAL_PREDICTIONS_PATH)
    
    evaluate.create_video_from_image_frames(CORRECTED_PREDICTIONS_PATH, VIDEO_PREDICTION_PATH, 25)
    
    evaluate.create_video_from_image_frames(ORIGINAL_PREDICTIONS_PATH, VIDEO_PREDICTION_PATH, 25)
    
    #TODO: use corrected coordinated obv of the files that remain after the filtering
    end_time = time.time()
    runtime = end_time - start_time
    print(f"\nTotal execution time: {runtime:.2f} seconds")