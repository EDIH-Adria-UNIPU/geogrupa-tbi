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
    st.header("📊 Table data")
    st.write("")
    col1, col2, col3= st.columns([1,9,1])
    with col2:
        df = st.session_state.df
        st.dataframe(df)
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download (.csv)",
            data=csv,
            file_name="detected_objects.csv",
            mime="text/csv"
        )
else:
    st.warning("Table data will be shown after detection is processed", icon="⚠️")