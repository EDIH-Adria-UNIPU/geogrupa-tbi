import base64
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def embed_images(html):
    def replacer(match):
        filename = match.group(1)
        img_path = Path(f"detections_{st.session_state.dataset_id}") / filename  # primjer: thumbnails/something.jpg
        if not img_path.exists():
            return match.group(0)  # ne mijenjaj ako ne postoji

        ext = img_path.suffix.lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        b64_data = image_to_base64(img_path)
        return f"src='data:{mime_type};base64,{b64_data}'"

    # zamjena za sve <img src='thumbnails/ime.jpg'> (i slične)
    updated_html = re.sub(r"src=['\"](thumbnails/[^'\"]+)['\"]", replacer, html)
    return updated_html

def sidebar():
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
def map():
    if "data" not in st.session_state:
        st.session_state.data=True

    if not st.session_state.data:
        st.header("🗺️ Map")
        st.write("")
        html_path = Path(f"detections_{st.session_state.dataset_id}/objects_map.html")
        html_content = html_path.read_text(encoding="utf-8")
        html_embedded = embed_images(html_content)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", len(st.session_state.df))
        with col2:
            if "class" in st.session_state.df.columns:
                st.metric("Unique Object Types", st.session_state.df["class"].nunique())
        with col3:
            if "conf" in st.session_state.df.columns:
                st.metric("Avg Confidence", f"{st.session_state.df['conf'].mean():.2f}")
        components.html(html_embedded, height=600)
        st.download_button(
            label="📥 Download Map (.html)",
            data=html_embedded,
            file_name="objects_map_embedded.html",
            mime="text/html",
        )
    
    else:
        st.warning("Map will be shown after detection is processed", icon="⚠️")

if __name__ == "__main__":
    sidebar()
    map()