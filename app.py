import streamlit as st
import PIL
#from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from tensorflow import keras
#import sys
#import base64
#import cv2
from tensorflow.keras.models import model_from_json
import pickle

model_config = pickle.load(open("model_arch.json", "rb"))
model = model_from_json(model_config)
model.load_weights('w.h5')

# model = keras.models.load_model('steel_defect_detector')

def process_img(image_path):
    im=Image.open(image_path)
    im=im.resize((120, 120))
    im_arr=np.asarray(im)
    im_arr=im_arr/255.0
    im_arr=im_arr.reshape(1,120,120,3)
    im_arr=im_arr.astype('float16')
    #print(im_arr.size*im_arr.itemsize)
    return im_arr


# def predict_json(project, region, model, instances, version=None):
#     """Send json data to a deployed model for prediction.
#     Args:
#         project (str): project where the Cloud ML Engine Model is deployed.
#         model (str): model name.
#         instances ([Mapping[str: Any]]): Keys should be the names of Tensors
#             your deployed model expects as inputs. Values should be datatypes
#             convertible to Tensors, or (potentially nested) lists of datatypes
#             convertible to Tensors.
#         version (str): version of the model to target.
#     Returns:
#         Mapping[str: any]: dictionary of prediction results defined by the 
#             model.
#     """
#     # Create the ML Engine service object
#     prefix = "{}-ml".format(region) if region else "ml"
#     api_endpoint = "https://{}.googleapis.com".format(prefix)
#     client_options = ClientOptions(api_endpoint=api_endpoint)

#     # Setup model path
#     model_path = "projects/{}/models/{}".format(project, model)
#     if version is not None:
#         model_path += "/versions/{}".format(version)

#     # Create ML engine resource endpoint and input data
#     ml_resource = googleapiclient.discovery.build(
#         "ml", "v1", cache_discovery=False, client_options=client_options).projects()
#     instances_list = instances.tolist()# turn input into list (ML Engine wants JSON)
#     #print(sys.getsizeof(instances_list))
    
#     input_data_json = {"signature_name": "serving_default",
#                        "instances": instances_list} 

#     request = ml_resource.predict(name=model_path, body=input_data_json)
#     response = request.execute()
    
#     # # ALT: Create model api
#     # model_api = api_endpoint + model_path + ":predict"
#     # headers = {"Authorization": "Bearer " + token}
#     # response = requests.post(model_api, json=input_data_json, headers=headers)

#     if "error" in response:
#         raise RuntimeError(response["error"])

#     return response["predictions"]
        

st.title('steel Detection App')
st.write(
    """
    ### A Simple WebApp to demonstrate Transfer Learning Predictions on  steel Dataset
    """
)

def find_pred(im):
    res = model.predict(im)
    return np.argmax(res) + 1

image_file=st.sidebar.file_uploader('Upload an image',type=['jpg','png'])

if(image_file):
    with st.beta_expander('Selected Image',expanded=True):
        st.image(image_file,use_column_width='auto')

if image_file and st.sidebar.button('Predict'):
    image=process_img(image_file)
    pred=find_pred(image)
    st.write(pred)

