# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:13:01 2022

@author: bokey

Run command : streamlit run app.py [-- script args]

"""

import streamlit as st
from pre_processors.read_im import read_im
import numpy as np
from PIL import Image
import cv2
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('Power IgmA Pipe: Poweful Image Analysis Pipeline')
st.header('Interactive image pre-processing and automated pipeline creation')


# step 0: Accept image input

uploaded_file = st.file_uploader("Upload image file")
    
if uploaded_file is not None:
    # To read file as bytes:
    #step 0 : read the file 
    imput_im = Image.open(uploaded_file)
    imput_im_np = np.array(imput_im.convert('RGB'))
    im = imput_im_np
    
    st.subheader("map input image to different color spaces")
    
    #step 1: map the colore spce
    color_space = st.radio('chage to following color space:', ['Gray scale','hsv', 'lab', 'brg', 'ch_one',
                                                                        'ch_two',
                                                                        'ch_three',
                                                                        'merge_first_two_ch',
                                                                        'merge_last_two_ch', 'merge_last_first_ch'], horizontal=True,)
    col1, col2 = st.columns( [0.5, 0.5])

    with col1:
        st.image(imput_im)
        #width, height = imput_im.size
        st.write("resolution of input image: ", im.shape)
    with col2:
        if color_space == "Gray scale":
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            st.image(im)
            st.write(im.shape)
        elif color_space == "hsv":
            im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            st.image(im)
        elif color_space == "lab":
            im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)
            st.image(im)
        elif color_space == "brg":
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            st.image(im)
        elif color_space == "ch_one":
            im, _, _ = cv2.split(im)
            st.image(im)
        elif color_space == "ch_two":
            _, im, _ = cv2.split(im)
            st.image(im)
        elif color_space == "ch_three":
            _, _, im = cv2.split(im)
            st.image(im)
        elif color_space == "merge_first_two_ch":
            im[:, :, 2] = np.zeros((im.shape[0], im.shape[1]))
            #ch3 = np.zeros(ch3.shape)
            st.write(im.shape)
            #im = cv2.merge([ch1, ch2, ch3])
            st.image(im)
        elif color_space == "merge_last_two_ch":
            im[:, :, 0] = np.zeros((im.shape[0], im.shape[1]))
            #im = cv2.merge([ch2, ch3])
            st.image(im)
        elif color_space == "merge_last_first_ch":
            im[:, :, 1] = np.zeros((im.shape[0], im.shape[1]))
            #im = cv2.merge([ch1, ch3])
            st.image(im)


    st.subheader("Brightness and contrast")
    st.sidebar.subheader("Controls for Brightness and contrast")

    st.subheader("smooting")


    blur_method = st.radio('chage to following color space:', ['None','Averaging', 'Gaussian', 
                                                                'Median', 'Bilateral'], horizontal=True,)

    col3, col4 = st.columns( [0.5, 0.5])

    with col3:
        st.image(im)
        #width, height = imput_im.size
        st.write("current state of the image")
    with col4:
        if color_space == "None":
            pass
        elif blur_method == "Averaging":
            st.sidebar.subheader("Controls for Averaging smooting")
            filter_ = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=5, step=2)
            im = cv2.blur(im,(filter_,filter_))
            st.image(im)
        elif blur_method == "Gaussian":
            st.sidebar.subheader("Controls for Gaussian smooting")
            filter_ = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=7, step=2)
            im = cv2.GaussianBlur(im,(filter_,filter_),0)
            st.image(im)
        elif blur_method == "Median":
            st.sidebar.subheader("Controls for Median smooting")
            filter_ = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=5, step=2)
            im = cv2.medianBlur(im,filter_)
            st.image(im)
        elif blur_method == "Bilateral":
            st.sidebar.subheader("Controls for Bilateral smooting")
            sigma_color = st.sidebar.slider('Adjust para 1 (sigma color)', min_value=1, max_value=11, value=9, step=1)
            sigma_space = st.sidebar.slider('Adjust para 2 (sigma space)', min_value=1, max_value=150, value=75, step=1)
            im = cv2.bilateralFilter(im,sigma_color,sigma_space,sigma_space)
            st.image(im)

    st.subheader("Histogram")
    
    hist_radio = st.radio('Compute Histogram:', ['None','Histogram', 'Histigram_equilisation'], horizontal=True,)

    col11, col12 = st.columns( [0.5, 0.5])

    with col11:
        st.image(im)
        #width, height = imput_im.size
        st.write("current state of the image")
    with col12:
        if hist_radio == "None":
            pass
        elif hist_radio == "Histogram":
            st.sidebar.subheader("Controls for Histigram")
            if len(im.shape) == 2:
                histogram= cv2.calcHist([im], [0], None, [256], [0, 256])
                plt.figure(1)
                plt.title("Histogram of input One chaneel image") 
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                plt.xlim([0, 256])
                plt.plot(histogram)
                st.pyplot(plt.figure(1))
            else:
                plt.figure(1)
                plt.title("Histogram of input 3 channel image") 
                plt.xlabel('Bins')
                plt.ylabel('Number of pixels')
                colors = ("b", "g", "r")
                
                for i, col in enumerate(colors):
                        hist = cv2.calcHist([im], [i], None, [256], [0, 256])
                        plt.plot(hist, color=col)
                        plt.xlim([0, 256])

                st.pyplot(plt.figure(1))


        elif hist_radio == "Histigram_equilisation":
            st.sidebar.subheader("Controls for Histigram_equilisation")
            filter_ = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=5, step=2)
            im = cv2.blur(im,(filter_,filter_))
            st.image(im)

    st.subheader("thresholding")
    st.sidebar.subheader("Controls for thresholding")
    thresh_method = st.radio('chage to following color space:', ['None','Thresholding', 'Adaptive thresholding', 
                                                                'Otsu thresholding', ''], horizontal=True,)

    col13, col14 = st.columns( [0.5, 0.5])

    with col13:
        st.image(im)
        #width, height = imput_im.size
        st.write("current state of the image")
    with col14:
        if thresh_method == "None":
            pass
        elif thresh_method == "Thresholding":
            st.sidebar.subheader("Controls for Averaging smooting")
            filter_ = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=5, step=2)
            im = cv2.blur(im,(filter_,filter_))
            st.image(im)
        elif thresh_method == "Adaptive thresholding":
            st.sidebar.subheader("Controls for Gaussian smooting")
            filter_ = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=7, step=2)
            im = cv2.GaussianBlur(im,(filter_,filter_),0)
            st.image(im)
        elif thresh_method == "Otsu thresholding":
            pass


    st.subheader("canny")

    edge_option = st.radio('Edge detection:', ['None', 'Sobel','Lanlasian', 'Canny'], horizontal=True,)
    col5, col6 = st.columns( [0.5, 0.5])

    with col5:
        st.image(im)
        #width, height = imput_im.size
        st.write("current state of the image")
    with col6:
        if edge_option == "None":
            pass
        elif edge_option == "Sobel":
            slider3 = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=5, step=2)
            im = cv2.blur(im,(slider3,slider3))
            st.image(im)
        elif edge_option == "Lanlasian":
            slider3 = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=7, step=2)
            im = cv2.GaussianBlur(im,(slider3,slider3),0)
            st.image(im)
        elif edge_option == "Canny":
            st.sidebar.subheader("Controls for canny")
            slider3 = st.sidebar.slider('Adjust minVal', min_value=0, max_value=255, value=150, step=1)
            slider4 = st.sidebar.slider('Adjust maxVal', min_value=0, max_value=255, value=255, step=1)
            im = cv2.Canny(im, slider3, slider4)
            st.image(im)
    
    st.subheader("Dialate/Erode")
    dia_ero_option = st.radio('Operations o0n the detected edges:', ['None', 'Dialate','Erode'], horizontal=True,)

    col7, col8 = st.columns( [0.5, 0.5])

    with col7:
        st.image(im)
        #width, height = imput_im.size
        st.write("current state of the image")
    with col8:
        if dia_ero_option == "None":
            pass
        elif dia_ero_option == "Dialate":
            st.sidebar.subheader("Controls for Dialate")
            kernel_side = st.sidebar.slider('kernel size', min_value=1, max_value=11, value=5, step=2)
            iterations = st.sidebar.slider('iterations', min_value=1, max_value=11, value=1, step=1)
            kernel = np.ones((kernel_side, kernel_side))
            im = cv2.dilate(im, kernel, iterations)
            st.image(im)
        elif dia_ero_option == "Erode":
            st.sidebar.subheader("Controls for Erode")
            slider5 = st.sidebar.slider('Adjust the filter size', min_value=1, max_value=11, value=7, step=2)
            im = cv2.GaussianBlur(im,(slider,slider),0)

    st.subheader("find countours")

    contour_option = st.radio('Operations o0n the detected edges:', ['None','Detect contours'], horizontal=True,)

    col9, col10 = st.columns( [0.5, 0.5])

    with col9:
        st.image(im)
        #width, height = imput_im.size
        st.write("current state of the image")
    with col10:
        if contour_option == "None":
            pass
        elif contour_option == "Detect contours":
            st.sidebar.subheader("Controls for contour detection")
            img_cont = imput_im_np.copy()
            ret_method_op = st.sidebar.radio('Option retrival method', ['RETR_EXTERNAL','RETR_TREE', 'RETR_LIST', 'RETR_CCOMP',], horizontal=True,)
            approx_method_op = st.sidebar.radio('Option approximation method', ['CHAIN_APPROX_NONE', 'CHAIN_APPROX_SIMPLE'], horizontal=True,)
            filter_area = st.sidebar.radio('Filter contours based on area', ['Yes', 'No'], horizontal=True,)
            filter_peri = st.sidebar.radio('Filter contours based on peri', ['Yes', 'No'], horizontal=True,)
            if filter_area == "Yes":
                slider6 = st.sidebar.slider('Display contours within this area:', min_value=1, max_value=20000, value=(1000, 2000), step=100)
            if filter_peri == "Yes":   
                slider7 = st.sidebar.slider('Display Conours within this perimeter:', min_value=1, max_value=10000, value=(1000, 2000), step=100)
            
            if ret_method_op == "RETR_EXTERNAL":
                ret_method = cv2.RETR_EXTERNAL
            elif ret_method_op =="RETR_TREE":
                ret_method = cv2.RETR_TREE
            elif ret_method_op =="RETR_LIST":
                ret_method = cv2.RETR_LIST
            elif ret_method_op =="RETR_CCOMP":
                ret_method = cv2.RETR_CCOMP


            if approx_method_op == "CHAIN_APPROX_NONE":
                approx_method = cv2.CHAIN_APPROX_NONE
            elif approx_method_op == "CHAIN_APPROX_SIMPLE":
                approx_method = cv2.CHAIN_APPROX_SIMPLE
            contours, hierarchy = cv2.findContours(im, ret_method, approx_method)
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.2 * peri, True)
                if filter_area == "Yes":
                    if slider6[0] <=area <= slider6[1]:
                        cv2.drawContours(img_cont, cnt, -1, (255, 0, 255), 7)
                    
                elif filter_peri == "Yes":
                    if slider7[0] <= peri <= slider7[1]:
                        cv2.drawContours(img_cont, cnt, -1, (255, 0, 255), 7)
                else:
                    cv2.drawContours(img_cont, cnt, -1, (255, 0, 255), 7)
                st.sidebar.write("*************************contour detected", i, "******************************")
                st.sidebar.write("Number for vetices in this contour is:", len(approx))
                st.sidebar.write("area is:", area)
                st.sidebar.write("perimeter is:", peri)

                
            st.image(img_cont)





