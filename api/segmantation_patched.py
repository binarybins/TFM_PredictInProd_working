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
model_trained = "models/satellite_standard_unet_100epochs_8Sep2022.hdf5"
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            model_trained)
model = load_model(model_path, compile=False)
# model = load_model("../models/satellite_standard_unet_100epochs_8Sep2022.hdf5", compile=False)

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

if __name__ == "__main__":
    image = "raw_data/single_image/sg_random.jpg"
    img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "raw_data/single_image/sg_random.jpg")
    unpatched_prediction = segment(img_path)
    plt.imshow(unpatched_prediction)
    plt.axis('off')
    image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'raw_data/single_image/sg_random_processed.png')
    # plt.savefig('../raw_data/single_image/sg_random_processed.png')
    plt.savefig(image_path)
