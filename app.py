# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:13:01 2022

@author: bokey

Run command : streamlit run app.py [-- script args]

"""

import streamlit as st

st.title('Power IgmA Pipe:')

st.subheader('Interactive image pre-processing and automated pipeline creation')


# step 0: Accept image input

uploaded_file = st.file_uploader("Upload image file")
    
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)    