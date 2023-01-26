import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
from utils import *

labels = gen_labels()

html_temp = '''
    <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Garbage Classification</h1></center>
    <center><h4>by candidate: Leon Yang</h4></center>
    <center><h3>Video Demo:</h3></center>
    </div>
    '''
video_file = open('Demo.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)

st.markdown(html_temp, unsafe_allow_html=True)
html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Try yourself: </h3></center>
    <center><h4>Please upload garbage image to find its Category</h4></center>
    </div>
    '''
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        image = Image.open(file)

elif opt == 'Upload image via link':

  try:
    img = st.text_input('Enter Image Address')
    image = Image.open(urllib.request.urlopen(img))
    
  except:
    if st.button('Submit'):
      show = st.error("Please enter a valid image address!")
      time.sleep(4)
      show.empty()

try:
  if image is not None:
    st.image(image, width = 300, caption = 'Uploaded Image')
    if st.button('Predict'):
        img = preprocess(image)

        model = model_arc()
        model.load_weights("model.h5")

        prediction = model.predict(img[np.newaxis, ...])
        st.info('The image has been classified as " {} waste " '.format(labels[np.argmax(prediction[0], axis=-1)]))
except Exception as e:
  #st.info(e)
  pass

