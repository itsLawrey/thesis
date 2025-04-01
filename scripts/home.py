import streamlit as st
import math
import pipeline 


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
    st.set_page_config(page_title="soterats project", layout="wide")
    st.write("# Hello World.")
    
    func_testing()