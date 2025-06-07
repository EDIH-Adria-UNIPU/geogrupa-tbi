import streamlit as st

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
    col1, col2, col3= st.columns([1,9,1])
    with col2:
        df = st.session_state.df
        df["lat"]=df["lat"].round(6)
        df["lon"]=df["lon"].round(6)
        df_unique = df.drop_duplicates(subset=['lat', 'lon'])
        df_renamed = df_unique.rename(columns={
            "lat": "Latitude",
            "lon": "Longitude",
            "bearing": "Bearing",
            "class": "Class",
            "conf": "Confidence",
            "frame": "Frame",
            "thumb": "Thumbnail"
        })
        st.dataframe(df_renamed)
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download (.csv)",
            data=csv,
            file_name="detected_objects.csv",
            mime="text/csv"
        )
else:
    st.warning("Table data will be shown after detection is processed", icon="⚠️")