#IMPORTS
import os
import torch
import tempfile

from pages.backend import predict, mask, skeleton, evaluate, opticalflow, slice
from pages.backend.logs import log, warn

import streamlit as st



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
    
    st.title("Execute the pipeline")
    
    st.markdown("### Generates data based on configuration.")
    
    if 'file_to_predict' in st.session_state and 'output_location' in st.session_state:
        
        ss = st.session_state

        file_bytes = ss.file_to_predict.getvalue()
        file_type = ss.file_to_predict.type
        file_name = ss.file_to_predict.name        
        
        if st.button(label="PREDICT", help="press this to start predicting", use_container_width=True):
            input_temp_path = None 
            suffix = os.path.splitext(file_name)[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                tfile.write(file_bytes)
                input_temp_path = tfile.name
            #TODO PROGRESS FORGOS CUCCLI
            execute(input_temp_path,
                   st.session_state["output_location"],
                   st.session_state["save_init_pred"],
                   st.session_state["save_orig_pred"],
                   st.session_state["save_corr_pred"],
                   st.session_state["save_masks"],
                   st.session_state["save_skel"],
                   st.session_state["save_orig_vid"],
                   st.session_state['save_corr_vid'],
                   )#TODO CONFIDENCE RATE AND FLOW THRESHOLD AND STUFF
            
            op = st.session_state["output_location"]
            st.success(f"✅✅✅okay big boy go see {op}✅✅✅")
        
            if input_temp_path and os.path.exists(input_temp_path):
                os.remove(input_temp_path)
                print(f"Temporary file deleted: {input_temp_path}")
    else:
        st.markdown("# Uh oh...👀")
        st.markdown("### Upload a video and configure settings to be able to predict.")
        st.subheader("Video and output path mandatory.")
    