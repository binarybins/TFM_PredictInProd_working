import cv2
import numpy as np

import segmentation_models as sm
sm.set_framework('tf.keras')


def preprocess_data(image):
    # Image must be of size (1024, 1024, 3)
    # Preprocess based on the pretrained backbone...
    img = cv2.imread(image, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    processed_img = preprocess_input(img)
    final_img = processed_img.reshape((1, 1024, 1024, 3))
    return final_img

def predict(image, model):
    img = preprocess_data(image)
    pred = model.predict(img)
    pred = np.argmax(pred[0], axis=2)
    return pred # return a np.array of shape (1024, 1024)
