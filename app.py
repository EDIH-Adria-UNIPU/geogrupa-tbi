import tempfile
import time
from pathlib import Path

import pandas as pd

import streamlit as st
from main import (
    create_dataset,
    extract_telemetry,
    extract_video_id,
    run_detection,
    triangulate_objects,
)

st.set_page_config(
    page_title="Home",
    page_icon="🚦",
)

# st.image("logo.svg")
st.title("🛰️ GeoGrupa – Traffic Signs and Streetlight Poles Detection System")

st.markdown(
    """
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
        """
)

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
video = st.file_uploader("File upload", type=["mp4"])

placeholder = st.empty()

if video is None:
    st.session_state.vid = True
    st.info("Please choose a file to upload!")
else:
    st.session_state.vid = False
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video.read())
        tmp_filepath = tmp_file.name
    st.success("File uploaded successfully!", icon="✅")

placeholder = st.empty()


def click_button():
    with placeholder.container():
        with st.spinner("Detection in progress..."):
            # Extract dataset ID from the uploaded video filename
            video_path = Path(tmp_filepath)
            dataset_id = extract_video_id(video_path)

            # Step 1: Create dataset
            dataset_dir = create_dataset(video_path, dataset_id)

            # Step 2: Extract telemetry
            telemetry_file = extract_telemetry(video_path, dataset_id)

            # Step 3: Run detection
            detection_dir = run_detection(dataset_dir, telemetry_file, dataset_id)

            # Step 4: Triangulate objects
            triangulate_objects(detection_dir, telemetry_file, dataset_id)

            # Store the dataset_id for later use
            st.session_state.dataset_id = dataset_id
            st.session_state.df = get_data(dataset_id)
        st.success("Detection completed!", icon="✅")
        time.sleep(3)
    st.session_state.data = False


@st.cache_data
def get_data(dataset_id):
    df = pd.read_csv(f"./detections_{dataset_id}/detections_geo.csv")
    return df


col1, col2 = st.columns([3, 1])
with col1:
    st.button("Start detection", on_click=click_button, disabled=st.session_state.vid)

# Display results if processing is complete
if hasattr(st.session_state, "df") and st.session_state.df is not None:
    st.header("🎯 Detection Results")

    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", len(st.session_state.df))
    with col2:
        if "class" in st.session_state.df.columns:
            st.metric("Unique Object Types", st.session_state.df["class"].nunique())
    with col3:
        if "conf" in st.session_state.df.columns:
            st.metric("Avg Confidence", f"{st.session_state.df['conf'].mean():.2f}")

    # Display interactive map if available
    if hasattr(st.session_state, "dataset_id"):
        map_file = (
            Path(f"detections_{st.session_state.dataset_id}") / "objects_map.html"
        )
        if map_file.exists():
            st.subheader("🗺️ Interactive Map")
            with open(map_file, "r", encoding="utf-8") as f:
                map_html = f.read()
            st.components.v1.html(map_html, height=500)

    # Display data table
    st.subheader("📊 Detection Data")
    st.dataframe(st.session_state.df, use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv_data = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"detections_{st.session_state.dataset_id}.csv",
            mime="text/csv",
        )

    with col2:
        if hasattr(st.session_state, "dataset_id"):
            map_file = (
                Path(f"detections_{st.session_state.dataset_id}") / "objects_map.html"
            )
            if map_file.exists():
                with open(map_file, "r", encoding="utf-8") as f:
                    map_html = f.read()
                st.download_button(
                    label="🗺️ Download Map (HTML)",
                    data=map_html,
                    file_name=f"objects_map_{st.session_state.dataset_id}.html",
                    mime="text/html",
                )
