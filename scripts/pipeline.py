#IMPORTS
import os
import torch
import predict
import mask
import skeleton
import evaluate
import opticalflow
from logs import log, warn

#GLOBAL VARIABLES
SAM_MODEL_PATH = "/home/dalloslorand/YOLO_lori/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = 'vit_h'

YOLO_MODEL_PATH = "/home/dalloslorand/YOLO_lori/runs/pose/full_v8/weights/best.pt"

FLOW_DIFF_THRESHOLD = 0.2

OUTPUT_PATH = "/home/dalloslorand/YOLO_lori/BSC THESIS/thesis/scripts/out_videoinputtal"
LABELS_PATH = f"{OUTPUT_PATH}/labels"
MASKS_PATH = f"{OUTPUT_PATH}/masks"
CORRECTED_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/corrected"
ORIGINAL_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/original"
VIDEO_PREDICTION_PATH = f"{OUTPUT_PATH}/predictions/videos"
SKELETONS_PATH = f"{OUTPUT_PATH}/skeletons"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 

SAVE_INITIAL_PREDICTIONS = False
SAVE_CORRECTED_PREDICTIONS = True
SAVE_MASKS = False
SAVE_SKELETON_CORRECTION_IMAGES = False

def create_folders():
    for folder in [OUTPUT_PATH, LABELS_PATH, CORRECTED_PREDICTIONS_PATH,ORIGINAL_PREDICTIONS_PATH,  MASKS_PATH, SKELETONS_PATH]:
        os.makedirs(folder, exist_ok=True)
        log(f"Created {folder} for output.")

def execute(input_file):
    
    create_folders()
    

#YOLO INFERENCE
    
    predictions = predict.predict_with_yolo(input_file, YOLO_MODEL_PATH)
    
    predict.save_initial_predictions(predictions, SAVE_INITIAL_PREDICTIONS, ORIGINAL_PREDICTIONS_PATH)

#SAM MASK IDENTIFICATION
    
    masks = mask.get_masks_with_sam(predictions, SAM_MODEL_TYPE, SAM_MODEL_PATH, DEVICE)
    
    mask.save_sam_masks(masks, SAVE_MASKS, MASKS_PATH)

#OPTICAL FLOW FILTERING
    
    predictions_filtered, masks_filtered = opticalflow.filter_data(predictions, masks, FLOW_DIFF_THRESHOLD)
    
#SKELETONIZE TOE CORRECTION
    
    skeletons = skeleton.get_skeletons(masks_filtered)
    
    corrected_predictions_dict = skeleton.correct_toes_skeleton(predictions_filtered, skeletons, SAVE_SKELETON_CORRECTION_IMAGES, SKELETONS_PATH)

#SUMMING UP DATA - PLOTS, VIDEOS

    images = evaluate.draw_corrected_predictions(corrected_predictions_dict, CORRECTED_PREDICTIONS_PATH, SAVE_CORRECTED_PREDICTIONS)
    
    # evaluate.draw_uncorrected_predictions(predictions, ORIGINAL_PREDICTIONS_PATH)
    
    evaluate.create_video_from_image_frames(CORRECTED_PREDICTIONS_PATH, VIDEO_PREDICTION_PATH, 25)
    
    # evaluate.create_video_from_image_frames(ORIGINAL_PREDICTIONS_PATH, VIDEO_PREDICTION_PATH, 25)
    
    
if __name__ == "__main__":
    execute('/home/dalloslorand/YOLO_lori/predicted_videos/sample_videos_for_prediction/R5W1_500.mp4')