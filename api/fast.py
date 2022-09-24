
#$DELETE_BEGIN

# import pytz

import pandas as pd
# import joblib

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

### Upload then get image ####
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, FileResponse
from api.segmantation_patched import segment
import matplotlib.pyplot as plt
import os
import segmentation_models as sm
from api import jasper_predict

###Loading models###
from keras.models import load_model
model_trained1 = "models/satellite_standard_unet_100epochs_8Sep2022.hdf5"
model_trained2 = "models/satellite_standard_unet_mex_300epochs_17Sep2022.hdf5"
model_path1 = os.path.join(os.path.dirname(os.path.dirname(__file__)),model_trained1)
model_path2 = os.path.join(os.path.dirname(os.path.dirname(__file__)),model_trained2)

model1 = load_model(model_path1, compile=False)
model2 = load_model(model_path2, compile=False)
model3 = load_model("models/spacenet_gt_model_filtered_model_10_epoch.hdf5",
                        compile=False,
                        custom_objects={"focal_loss_plus_jaccard_loss": sm.losses.categorical_focal_jaccard_loss,
                                    "iou_score": sm.metrics.iou_score})


app = FastAPI()
IMAGEDIR = "data/"

@app.post("/upload_image")
async def create_upload_file(file: UploadFile = File(...)):

    contents = await file.read()  # <-- Important!

    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    unpatched_prediction1 = segment(f'{IMAGEDIR}{file.filename}',model1)

    # image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    #                           f'data/transformed_{file.filename}')
    image_path = f'{IMAGEDIR}t1_{file.filename}'
    plt.imshow(unpatched_prediction1)
    plt.axis('off')
    plt.savefig(image_path)

    unpatched_prediction2 = segment(f'{IMAGEDIR}{file.filename}',model2)
    image_path2 = f'{IMAGEDIR}t2_{file.filename}'
    plt.imshow(unpatched_prediction2, cmap="gray_r")
    plt.axis('off')
    plt.savefig(image_path2)

    sm.set_framework('tf.keras')
    sm.framework()


    unpatched_prediction3 = jasper_predict.predict(f'{IMAGEDIR}{file.filename}',model3)
    image_path3 = f'{IMAGEDIR}t3_{file.filename}'
    plt.imshow(unpatched_prediction3, cmap="gray")
    plt.axis('off')
    plt.savefig(image_path3)


    return {"message": f"Successfully uploaded {file.filename}"}

@app.get("/get_image")
async def read_file(filename):

    path = f"{IMAGEDIR}{filename}"

    return FileResponse(path)
    # return path



###testing###
# from segmantation_patched import segment
# model1 = load_model("../models/satellite_standard_unet_100epochs_8Sep2022.hdf5", compile=False)
# model2 = load_model("../models/satellite_standard_unet_mex_300epochs_17Sep2022.hdf5", compile=False)
# unpatched_prediction1 = segment("../data/sg_random.jpg",model1)

# # image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
# #                           f'data/transformed_{file.filename}')
# image_path = f'../data/t1_sg_random.jpg'
# plt.imshow(unpatched_prediction1)
# plt.axis('off')
# plt.savefig(image_path)

# unpatched_prediction2 = segment("../data/sg_random.jpg",model2)
# image_path2 = f'../data/t2_sg_random.jpg'
# plt.imshow(unpatched_prediction2)
# plt.axis('off')
# plt.savefig(image_path2)

###testing end###
