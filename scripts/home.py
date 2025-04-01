import streamlit as st
import math
import pipeline 
import io
import os
import tempfile
from PIL import Image



def calculate_square(user_input):
    try:
        number = float(user_input)
        return number ** 2
    except ValueError:
        return math.nan
    
def func_testing():
    st.title("Square Calculator")
    
    # Input field
    user_input = st.text_input("Enter a number:", "")
    
    # Button to compute square
    if st.button("Calculate Square"):
        result = calculate_square(user_input)
        st.write(f"Result: {result}")

if __name__ == "__main__":
    st.set_page_config(page_title="soterats project")
    st.title("Hello World. from nipg")
    file_to_predict = st.file_uploader(
        "Upload an image or video...",
        type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"] # Add more if needed
    )
    
    if file_to_predict:
        file_details = {"FileName": file_to_predict.name, "FileType": file_to_predict.type, "FileSize": file_to_predict.size}
        st.write("---")
        st.write("Uploaded File Details:")
        st.json(file_details) # Show details in a nice format

        file_bytes = file_to_predict.getvalue()
        file_type = file_to_predict.type
        file_name = file_to_predict.name
        
        original_image = Image.open(io.BytesIO(file_bytes))
        st.image(original_image, caption='Uploaded Image', use_container_width=True)

        st.write("---")
        
        image_for_prediction = Image.open(io.BytesIO(file_bytes))
        
        if st.button(label="PREDICT", help="press this to start predicting"):
            temp_file_path = None
            suffix = os.path.splitext(file_name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
                    tfile.write(file_bytes)
                    temp_file_path = tfile.name # Get the path to the temporary file
                # -------------------------------

                
                
            predicted_images = pipeline.execute_task(temp_file_path)
            st.image(predicted_images, channels="BGR")
            st.success("YEEEEEEEEEEEEEES")
        
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Temporary file deleted: {temp_file_path}")
                
    