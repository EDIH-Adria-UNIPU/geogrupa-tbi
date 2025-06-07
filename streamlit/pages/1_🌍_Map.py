import streamlit as st
import base64
from pathlib import Path
import re

import streamlit.components.v1 as components


def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")



html_path = Path("./detections/objects_map.html")
html_content = html_path.read_text(encoding="utf-8")

def embed_images(html):
    def replacer(match):
        filename = match.group(1)
        img_path = Path("detections") / filename  # primjer: thumbnails/something.jpg
        if not img_path.exists():
            return match.group(0)  # ne mijenjaj ako ne postoji

        ext = img_path.suffix.lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"
        b64_data = image_to_base64(img_path)
        return f"src='data:{mime_type};base64,{b64_data}'"

    # zamjena za sve <img src='thumbnails/ime.jpg'> (i slične)
    updated_html = re.sub(r"src=['\"](thumbnails/[^'\"]+)['\"]", replacer, html)
    return updated_html



html_embedded = embed_images(html_content)

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

if "data" not in st.session_state:
    st.session_state.data=True


if not st.session_state.data:
    df=st.session_state.df
    components.html(html_embedded, height=600)
    st.download_button(
            label="📥 Download Map (.html)",
            data=html_embedded,
            file_name="objects_map_embedded.html",
            mime="text/html"
        )
else:
    st.warning("Map will be shown after detection is processed", icon="⚠️")
