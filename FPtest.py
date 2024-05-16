# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:00:30 2024

@author: user
"""

import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=load_model('best_FPmodel.h5')
  return model
model=load_model()
st.write("""
# Dog Image Recognition System"""
)
file=st.file_uploader("Choose any photo from computer",type=["jpg"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.Resampling.LANCZOS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Dog','Other']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)