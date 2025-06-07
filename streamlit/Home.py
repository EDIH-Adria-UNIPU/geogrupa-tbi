import streamlit as st
import time
import pandas as pd
import numpy as np
import tempfile
from create_image_dataset import create_dataset
from get_telemetry import telemetry
from run_detections_2 import run_detection
from triangulate_objects import triangulate_objects

st.set_page_config(
    page_title="Home",
    page_icon="🚦",
)

#st.image("logo.svg")
st.title("🛰️ GeoGrupa – Traffic Signs and Streetlight Poles Detection System")

st.markdown("""
        Welcome to the application developed for GeoGrupa – an advanced system for automatic detection of traffic signs and streetlight poles based on real-world video footage.

        ## 🎯 About the Project  
        The goal of this project is to enable easy and efficient collection of spatial data about traffic infrastructure through video analysis. The system uses computer vision to:

        - Detect traffic signs and streetlight poles  
        - Classify types of traffic signs  
        - Accurately determine the geographic location (lat/lon) of each object  
        - Display results on an interactive map  
        - Generate a CSV file with data about all detected objects  

        ## 📽️ How to Use the Application?  
        Upload a video file (size may be up to **2GB**).  
        The application will automatically process the video and:

        - Detect traffic signs and streetlight poles  
        - Classify signs ("Stop", "Priority road sign" or "Other sign".)  
        - Calculate precise GPS coordinates for each object  

        Upon completion:

        - You will receive an interactive map with marked objects  
        - You can view a table with data  
        - Individual objects include associated detection images  
        - You can download a CSV file containing information such as:  
            - Geographic latitude and longitude  
            - Object classification  
            - Detection confidence  
            - Image filename  
        - You can download the map as an HTML file
        """)

with st.sidebar:
    about = st.text(
        """
        Aplikacija omogućava detekciju prometnih 
        znakova te stupova javne rasvjete na 
        danom videu.

        Nakon što se izvrši detekcija detektirane 
        prometne znakove je moguće prikazati na 
        mapi te kao tablicu.

        Podatke detektiranih prometnih znakova i 
        stupova javne rasvjete  je također moguće 
        preuzeti u .csv formatu.
        """
    )

st.header("File upload")
video=st.file_uploader("File upload", type=["mp4"])

placeholder=st.empty()

if video is None:
    st.session_state.vid=True
    st.info("Please choose a file to upload!")
else:
    st.session_state.vid=False
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video.read())
        tmp_filepath = tmp_file.name    
    st.success("File uploaded successfully!", icon="✅")

placeholder=st.empty()

def click_button():
    with placeholder.container():
        with st.spinner("Detection in progress..."):
            create_dataset(tmp_filepath)
            telemetry(video)
            run_detection()
            triangulate_objects()
            time.sleep(2)
            st.session_state.df = get_data()
        st.success("Detection completed!", icon="✅")
        time.sleep(3)
    st.session_state.data=False

@st.cache_data
def get_data():
    df = pd.read_csv("detections/detections_geo.csv")
    return df

col1, col2=st.columns([3,1])
with col1:
    st.button('Start detection', on_click=click_button, disabled=st.session_state.vid)


