
#$DELETE_BEGIN
from datetime import datetime
import pytz

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2


# @app.get("/")
# def index():
#     return dict(greeting="hello")


# @app.get("/predict")
# def predict(pickup_datetime,        # 2013-07-06 17:18:00
#             pickup_longitude,       # -73.950655
#             pickup_latitude,        # 40.783282
#             dropoff_longitude,      # -73.984365
#             dropoff_latitude,       # 40.769802
#             passenger_count):       # 1

#     # create datetime object from user provided date
#     pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

#     # localize the user provided datetime with the NYC timezone
#     eastern = pytz.timezone("US/Eastern")
#     localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

#     # convert the user datetime to UTC
#     utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

#     # format the datetime as expected by the pipeline
#     formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

#     # fixing a value for the key, unused by the model
#     # in the future the key might be removed from the pipeline input
#     # eventhough it is used as a parameter for the Kaggle submission
#     key = "2013-07-06 17:18:00.000000119"

#     # build X ⚠️ beware to the order of the parameters ⚠️
#     X = pd.DataFrame(dict(
#         key=[key],
#         pickup_datetime=[formatted_pickup_datetime],
#         pickup_longitude=[float(pickup_longitude)],
#         pickup_latitude=[float(pickup_latitude)],
#         dropoff_longitude=[float(dropoff_longitude)],
#         dropoff_latitude=[float(dropoff_latitude)],
#         passenger_count=[int(passenger_count)]))

#     # ⚠️ TODO: get model from GCP

#     # pipeline = get_model_from_gcp()
#     pipeline = joblib.load('model.joblib')

#     # make prediction
#     results = pipeline.predict(X)

#     # convert response from numpy to python type
#     pred = float(results[0])

#     return dict(fare=pred)
# $DELETE_END
"""
### segmantation_patched ###

import cv2
import numpy as np
import os

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
# import segmentation_models as sm

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

from keras.models import load_model
# model_trained = "models/satellite_standard_unet_100epochs_8Sep2022.hdf5"
# model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
#                             model_trained)
# model = load_model(model_path, compile=False)
model = load_model("models/satellite_standard_unet_100epochs_8Sep2022.hdf5", compile=False)

def segment(image):
    # img = cv2.imread("../raw_data/single_image/sg_random.jpg", 1)
    img = cv2.imread(image, 1)

    # size of patches
    patch_size = 256

    #################################################################################
    #Predict patch by patch with no smooth blending
    ###########################################

    SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
    SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
    large_img = Image.fromarray(img)
    large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
    #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    large_img = np.array(large_img)


    patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
    patches_img = patches_img[:,:,0,:,:,:]

    patched_prediction = []
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i,j,:,:,:]

            #Use minmaxscaler instead of just dividing by 255.
            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
            single_patch_img = np.expand_dims(single_patch_img, axis=0)
            pred = model.predict(single_patch_img)
            pred = np.argmax(pred, axis=3)
            pred = pred[0, :,:]

            patched_prediction.append(pred)

    patched_prediction = np.array(patched_prediction)
    patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1],
                                                patches_img.shape[2], patches_img.shape[3]])

    unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

    # plt.imshow(unpatched_prediction)
    # plt.axis('off')
    # image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    #                           'raw_data/single_image/sg_random_processed.png')
    # # plt.savefig('../raw_data/single_image/sg_random_processed.png')
    # plt.savefig(image_path)
    return unpatched_prediction
"""





# ###Uploading an image###
# from fastapi import File, UploadFile

# @app.post("/upload")
# def upload(file: UploadFile = File(...)):
#     try:
#         contents = file.file.read()
#         with open(file.filename, 'wb') as f:
#             f.write(contents)
#     except Exception:
#         return {"message": "There was an error uploading the file"}
#     finally:
#         file.file.close()

#     return {"message": f"Successfully uploaded {file.filename}"}

# ###Displaying an image###
# from fastapi.responses import FileResponse

# some_file_path = "data/sg_random.jpg"

# @app.get("/image_from_folder")
# async def main():
#     return FileResponse(some_file_path)

# ###Uploading and displaying the same image###

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import Response

# app = FastAPI()

# @app.post("/display_uploaded")
# async def create_upload_file(file: UploadFile = File(...)):

#     contents = await file.read()  # <-- Important!

#     response = Response(contents)
#     return response

### Upload then get image ####
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, FileResponse
from api.segmantation_patched import segment
import matplotlib.pyplot as plt
import os

app = FastAPI()
IMAGEDIR = "data/"

@app.post("/upload_image")
async def create_upload_file(file: UploadFile = File(...)):

    contents = await file.read()  # <-- Important!

    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    unpatched_prediction = segment(f'{IMAGEDIR}{file.filename}')
    # image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    #                           f'data/transformed_{file.filename}')
    image_path = f'{IMAGEDIR}transformed_{file.filename}'
    plt.imshow(unpatched_prediction)
    plt.axis('off')
    plt.savefig(image_path)


    return {"message": f"Successfully uploaded {file.filename}"}

@app.get("/get_image")
async def read_file(filename):

    # return a response object directly as FileResponse expects a file-like object
    # and StreamingResponse expects an iterator/generator
    # files = os.listdir(IMAGEDIR)
    # path = f"{IMAGEDIR}{files[1]}"
    path = f"{IMAGEDIR}{filename}"

    return FileResponse(path)
    # return path
