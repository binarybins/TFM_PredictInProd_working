import jasper_predict
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
model = load_model("../models/spacenet_gt_model_filtered_model_10_epoch.hdf5",
                       compile=False,
                       custom_objects={"focal_loss_plus_jaccard_loss": sm.losses.categorical_focal_jaccard_loss,
                                  "iou_score": sm.metrics.iou_score})

img = jasper_predict.predict("../data/socal-fire_00000629_pre_disaster.png",model)
plt.imshow(img, cmap="gray")
plt.axis('off')
