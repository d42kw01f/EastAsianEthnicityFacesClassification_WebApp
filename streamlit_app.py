from genericpath import isfile
import streamlit as slit
import time
import os.path
import gdown
import zipfile
from PIL import Image
import subprocess
from datetime import datetime as dt
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

@slit.cache
def loadModel(model_path):
    
    classifier = VisionClassifierInference(
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k'),
        model = ViTForImageClassification.from_pretrained(model_path),
    )
    return classifier

def theModel(path):
    print('theModel Function!!')
    slit.markdown('***')
    slit.header('Real Time Prediction')
    uploaded_files = slit.file_uploader("Upload the photo here: ", accept_multiple_files=False)
    try:
        with slit.container():
            if uploaded_files is not None:
                image = Image.open(uploaded_files)
                slit.markdown('***')
                with slit.expander("To see the uploaded photo:"):
                    slit.image(image)
                if slit.button('Predict'):
                    slit.markdown('***')
                    with slit.spinner('Wait for it...'):
                        start = dt.now()
                        classifier = loadModel(path)
                        label = classifier.predict(img_path=uploaded_files)
                        running_secs = (dt.now() - start).microseconds
                    slit.header('Results :speak_no_evil:')
                    slit.success(f'Done!, predicted as a: {label}')
                    slit.write('Time: ', running_secs, 'μs')
    except:
        slit.warning('Something went wrong :exclamation:')
    print('theModel Function is done!\n\n')

def download_weights(
    url='https://drive.google.com/uc?id=1foV-oDiGMBn2t0c3sPr0nK6e-H82HbI5',
    out="asian_faces_model.zip",
):
    print('The weights downloading function!!')
    weights = "model/asian_faces_detection_model_v1/"
    if os.path.isfile('asian_faces_model.zip') is False:
        if os.path.isdir(weights):
            print('the weight downloading function is done!')
            return weights
        with slit.spinner('downloading the model...'):
            print('Downloading the model from the Google Drive')
            gdown.download(url, out, use_cookies=False)

    if os.path.isdir(weights) is False:
        with slit.spinner('Unzing the model...'):
            with zipfile.ZipFile('asian_faces_model.zip', 'r') as zip_ref:
                zip_ref.extractall('./')
                print('unzing is done!!!')
            print('the weight downloading function is done!\n\n')
            return weights
    else:
        print('the weight downloading function is done!\n\n')
        return weights

def firstPage():
    slit.title('Asian Faces Detection')
    slit.markdown('***')

    
def sourceCode():
    tab1, tab2, tab3 = stlit.tabs(["Source Code", "Hugging Face", "About Me"])

    with tab1:
       slit.markdown(":facepunch: [GitHub](https://github.com/d42kw01f/EastAsianEthnicityFacesClassification)")

    with tab2:
       slit.markdown("🤗 [HuggingFace](https://huggingface.co/d42kw01f/EastAsianEthnicityClassification)")

    with tab3:
        slit.markdown('### Dakshitha Perera')
        slit.markdown(":earth_asia: [https://d42kw01f.github.io/](https://d42kw01f.github.io/)")
        slit.markdown(":computer: [d42kw01f](https://github.com/d42kw01f)")
    
def aboutMe():
    slit.markdown('## About Me :point_down:')
    with slit.expander(slit.markdown("More Details: ")):
        slit.markdown('### Dakshitha Perera')
        slit.markdown(":earth_asia: [https://d42kw01f.github.io/](https://d42kw01f.github.io/)")
        slit.markdown(":computer: [d42kw01f](https://github.com/d42kw01f)")
    

def main(model_path):
    firstPage()
    theModel(model_path)
    slit.markdown('***')


if __name__=='__main__':
    Modelpath = download_weights()
    main(Modelpath)
    sourceCode()
    aboutMe()
