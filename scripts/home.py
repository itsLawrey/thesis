import streamlit as st
import math
import pipeline 
import io
import os
import tempfile
from PIL import Image

if __name__ == "__main__":
    
    
    st.set_page_config(page_title="soterats project")
    st.title("Hello World. from nipg")
    
    
    file_to_predict = st.file_uploader(
        "Upload video...",
        type=["mp4", "avi", "mov", "mkv"]
    )
    
    if file_to_predict:
        # file_details = {"FileName": file_to_predict.name, "FileType": file_to_predict.type, "FileSize": file_to_predict.size}
        # st.write("---")
        
        
        # st.write("Uploaded File Details:")
        # st.json(file_details)

        # file_bytes = file_to_predict.getvalue()
        # file_type = file_to_predict.type
        # file_name = file_to_predict.name
        
        st.video(file_to_predict)

        st.write("---")
        
        
        if st.button(label="PREDICT", help="press this to start predicting"):
            
            
            input_temp_path = None # Path for the video file fed *into* the pipeline
            output_video_path = None # Path for the video file *returned* by the pipeline
            
            
            suffix = os.path.splitext(file_name)[1]
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(file_bytes)
                    input_temp_path = tfile.name

                
                
            output_video_path = pipeline.execute_task(input_temp_path)
           
            with open(output_video_path, 'rb') as f:
                processed_video_bytes = f.read()
                
            st.video(processed_video_bytes)
                            
            st.success("YEEEEEEEEEEEEEES")
        
            if input_temp_path and os.path.exists(input_temp_path):
                os.remove(input_temp_path)
                print(f"Temporary file deleted: {input_temp_path}")
                
    