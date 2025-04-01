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
from logs import log, warn
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################

#GLOBAL VARIABLES
SAM_MODEL_PATH = "/home/dalloslorand/YOLO_lori/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = 'vit_h'

YOLO_MODEL_PATH = "/home/dalloslorand/YOLO_lori/runs/pose/full_v8/weights/best.pt"

OUTPUT_PATH = "/home/dalloslorand/YOLO_lori/BSC THESIS/thesis/scripts/out"
LABELS_PATH = f"{OUTPUT_PATH}/labels"
MASKS_PATH = f"{OUTPUT_PATH}/masks"
CORRECTED_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/corrected"
ORIGINAL_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/original"
VIDEO_PREDICTION_PATH = f"{OUTPUT_PATH}/predictions/videos"
SKELETONS_PATH = f"{OUTPUT_PATH}/skeletons"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

SAVE_INITIAL_PREDICTIONS = False
SAVE_MASKS = False
SAVE_SKELETON_CORRECTION_IMAGES = False

#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################

#FUNCTION DEFNINTIONS


def create_folders():
    for folder in [OUTPUT_PATH, LABELS_PATH, CORRECTED_PREDICTIONS_PATH,ORIGINAL_PREDICTIONS_PATH,  MASKS_PATH, SKELETONS_PATH]:
        os.makedirs(folder, exist_ok=True)
        log(f"Created {folder} for output.")


#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################
#######################################################################################################################################################################################################################################

def execute_task(input_file):
    
    
    create_folders()
    

#YOLO INFERENCE
    #SINGLE IMAGE MODE
    predictions = predict.predict_with_yolo(input_file, YOLO_MODEL_PATH)
    
    predict.save_initial_predictions(predictions, SAVE_INITIAL_PREDICTIONS, ORIGINAL_PREDICTIONS_PATH)

#SAM MASK IDENTIFICATION
    prediction_masks = masks.get_masks_with_sam(predictions, SAM_MODEL_TYPE, SAM_MODEL_PATH, DEVICE)
    
    masks.save_sam_masks(prediction_masks, SAVE_MASKS, MASKS_PATH)

#OPTICAL FLOW FILTERING
    #TODO FILTER FRAMEZZZ
    
#SKELETONIZE TOE CORRECTION
    #TODO HANDLE FACT THAT SOME FRAMES GOT FILTERED...
    skeletons = skeleton.get_skeletons(prediction_masks)
    
    corrected_predictions_dict = skeleton.correct_toes_skeleton(predictions,skeletons, SAVE_SKELETON_CORRECTION_IMAGES, SKELETONS_PATH)

#SUMMING UP DATA - PLOTS, VIDEOS

    images = evaluate.draw_corrected_predictions(corrected_predictions_dict, CORRECTED_PREDICTIONS_PATH)
    
    # evaluate.draw_uncorrected_predictions(predictions, ORIGINAL_PREDICTIONS_PATH)
    
    # evaluate.create_video_from_image_frames(CORRECTED_PREDICTIONS_PATH, VIDEO_PREDICTION_PATH, 25)
    
    # evaluate.create_video_from_image_frames(ORIGINAL_PREDICTIONS_PATH, VIDEO_PREDICTION_PATH, 25)
    
    #TODO: use corrected coordinated obv of the files that remain after the filtering



    #TEMP
    return images




end_time = time.time()
runtime = end_time - start_time
print(f"\nTotal execution time: {runtime:.2f} seconds")