import streamlit as st
from streamlit_option_menu import option_menu

import math
import pipeline 
import io
import os
import tempfile
from PIL import Image

if __name__ == "__main__":
    st.set_page_config(page_title="soterats project")
    st.title("Rat Pose Analyzer")
    st.write("---")



    
    
    with st.sidebar:
        selected = option_menu("Main Menu", 
                               ["Home","Upload", 'Settings',"Predict"], 
                                icons=['house', 'upload', 'gear', 'list-task'], 
                                menu_icon="code-slash", default_index=0)
    
    #if selected == "Home":
    st.markdown("# Welcome")
    st.markdown("### Upload a file and configure settings to begin")
        
        
        
    #if selected == "Upload":
    file_to_predict = st.file_uploader(
        "Upload video to use as prediction data",
         type=["mp4", "avi", "mov", "mkv"]
    )
    st.write("---")
        
        
        
    #if selected == "Settings":
    st.markdown(" ### Settings")
    output_location = st.text_input("Full Output Path", placeholder="full path to save outputs to...")
    col1, col2 = st.columns(2)
    with col1:
        save_init_pred = st.toggle("Save Initial Predictions")
        save_orig_pred = st.toggle("Save Original Predictions")
        save_corr_pred = st.toggle("Save Corrected Predictions")
        save_masks = st.toggle("Save Mask Visualizations")
    with col2:
        save_skel = st.toggle("Save Skeleton Correction Visualizations")
        save_orig_vid = st.toggle("Save Original Predictions In Video Form")
        save_corr_vid = st.toggle("Save Corrected Predictions In Video Form")
    
    st.write("---")
    
        
    
    #if selected == "Predict":
    if file_to_predict:
        file_details = {"FileName": file_to_predict.name, "FileType": file_to_predict.type, "FileSize": file_to_predict.size}
        
        st.write("Uploaded File Details:")
        st.json(file_details)

        file_bytes = file_to_predict.getvalue()
        file_type = file_to_predict.type
        file_name = file_to_predict.name        
        
        if st.button(label="PREDICT", help="press this to start predicting"):
            input_temp_path = None 
            suffix = os.path.splitext(file_name)[1]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                tfile.write(file_bytes)
                input_temp_path = tfile.name
                
            pipeline.execute(input_temp_path,
                            output_location,
                            save_init_pred,
                            save_orig_pred,
                            save_corr_pred,
                            save_masks,
                            save_skel,
                            save_orig_vid,
                            save_corr_vid,
                            )#TODO CONFIDENCE RATE AND FLOW THRESHOLD AND STUFF
            
            st.success("YEEEEEEEEEEEEEES now check results lil boi")
        
            if input_temp_path and os.path.exists(input_temp_path):
                os.remove(input_temp_path)
                print(f"Temporary file deleted: {input_temp_path}")
    else:
        st.markdown("# upload a video first")
                
                
     


  