import streamlit as st

#INIT BLOCK

if __name__ == "__main__":

    st.title("Settings")
    
    file_to_predict = st.file_uploader("Upload video to use as prediction data",type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=False)
    if file_to_predict:
        st.session_state["file_to_predict"] = file_to_predict
    
    output_location = st.text_input("Full Output Path", placeholder="full path to save outputs to...", value = st.session_state["output_location"] if "output_location" in st.session_state else "")
    if output_location:
        st.session_state["output_location"] = output_location
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["save_init_pred"] = st.toggle("Save Initial Predictions")
        st.session_state["save_orig_pred"] = st.toggle("Save Original Predictions")
        st.session_state["save_corr_pred"] = st.toggle("Save Corrected Predictions")
        st.session_state["save_masks"] = st.toggle("Save Mask Visualizations")
      
            
    with col2:
        st.session_state["save_skel"] = st.toggle("Save Skeleton Correction Visualizations")
        st.session_state["save_orig_vid"] = st.toggle("Save Original Predictions In Video Form")
        st.session_state['save_corr_vid'] = st.toggle("Save Corrected Predictions In Video Form")
        
    