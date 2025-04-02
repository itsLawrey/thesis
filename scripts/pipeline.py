#IMPORTS
import os
import torch
import predict
import mask
import skeleton
import evaluate
import opticalflow
import slice
import tempfile
from logs import log, warn

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 



def execute(input_file, 
            output_folder, 
            save_initial_predictions=False,
            save_original_predictions=False,
            save_corrected_predictions=False,
            save_masks=False,
            save_skeletons=False,
            save_original_vid=False,
            save_corrected_vid=False):
    
    OUTPUT_PATH = output_folder
    LABELS_PATH = f"{OUTPUT_PATH}/labels"
    MASKS_PATH = f"{OUTPUT_PATH}/masks"
    CORRECTED_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/corrected"
    ORIGINAL_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/original"
    INITIAL_PREDICTIONS_PATH = f"{OUTPUT_PATH}/predictions/initial"
    VIDEO_PREDICTION_PATH = f"{OUTPUT_PATH}/predictions/videos"
    SKELETONS_PATH = f"{OUTPUT_PATH}/skeletons"
    
    for folder in [OUTPUT_PATH, 
                   LABELS_PATH, 
                   CORRECTED_PREDICTIONS_PATH,
                   ORIGINAL_PREDICTIONS_PATH,  
                   MASKS_PATH, 
                   SKELETONS_PATH, 
                   VIDEO_PREDICTION_PATH, 
                   INITIAL_PREDICTIONS_PATH]:
        os.makedirs(folder, exist_ok=True)
        log(f"Created {folder} for output.")
        
    SAVE_INITIAL_PREDICTIONS = save_initial_predictions
    SAVE_ORIGINAL_PREDICTIONS = save_original_predictions
    SAVE_CORRECTED_PREDICTIONS = save_corrected_predictions
    SAVE_MASKS = save_masks
    SAVE_SKELETON_CORRECTION_IMAGES = save_skeletons
    SAVE_ORIGINAL_PRED_VIDEO = save_original_vid
    SAVE_CORRECTED_PRED_VIDEO = save_corrected_vid
    
    #TODO: ha nem eleg hosszu akkor nem is futtatjuk a programot
    
    with tempfile.TemporaryDirectory(prefix="pipeline_frames_") as temp_dir:
        frame_paths = slice.extract_frames_from_video(input_file, temp_dir)

    #YOLO INFERENCE
        
        predictions = predict.predict_with_yolo(frame_paths, DEVICE)
        
        predict.save_initial_predictions(predictions, SAVE_INITIAL_PREDICTIONS, INITIAL_PREDICTIONS_PATH)

    #SAM MASK IDENTIFICATION
        
        masks = mask.get_masks_with_sam(predictions, DEVICE)
        
        mask.save_sam_masks(masks, SAVE_MASKS, MASKS_PATH)

    #OPTICAL FLOW FILTERING
        
        predictions_filtered, masks_filtered = opticalflow.filter_data(predictions, masks)
        
    #SKELETONIZE TOE CORRECTION
        
        skeletons = skeleton.get_skeletons(masks_filtered)
        
        corrected_predictions_dict = skeleton.correct_toes_skeleton(predictions_filtered, skeletons, SAVE_SKELETON_CORRECTION_IMAGES, SKELETONS_PATH)

    #SUMMING UP DATA - PLOTS, VIDEOS

        images = evaluate.draw_corrected_predictions(corrected_predictions_dict, CORRECTED_PREDICTIONS_PATH, SAVE_CORRECTED_PREDICTIONS)
        
        evaluate.draw_uncorrected_predictions(predictions_filtered, ORIGINAL_PREDICTIONS_PATH, SAVE_ORIGINAL_PREDICTIONS)
        
        evaluate.create_video_from_image_frames(CORRECTED_PREDICTIONS_PATH, VIDEO_PREDICTION_PATH, SAVE_CORRECTED_PRED_VIDEO)
        
        evaluate.create_video_from_image_frames(ORIGINAL_PREDICTIONS_PATH, VIDEO_PREDICTION_PATH, SAVE_ORIGINAL_PRED_VIDEO)
    
# TESTING PURPOSES
if __name__ == "__main__":
    execute('/home/dalloslorand/YOLO_lori/predicted_videos/sample_videos_for_prediction/R5W1_500.mp4',"/home/dalloslorand/YOLO_lori/BSC THESIS/thesis/out/out_st")