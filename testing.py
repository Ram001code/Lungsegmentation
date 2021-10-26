import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
#from glob import glob
#from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
#from call import load_data, tf_dataset

H = 512
W = 512

""" Load the test images """
test_images = r'D:\Albot\Medical Imaging\Code\segmentation\images\img1.png'
Left_mask = r'D:\Albot\Medical Imaging\Code\segmentation\images\left1.png'
Right_mask= r'D:\Albot\Medical Imaging\Code\segmentation\images\right1.png'


""" Loading model """
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model("files/model.h5")

#Reading image
x = cv2.imread(test_images)
x = cv2.resize(x, (W, H))
cv2.imshow('input image', x)
cv2.waitKey(0)
#cv2.destroyAllWindows()
print("shape", x.shape)

x1 = x/ 255.0
x1 = x1.astype(np.float32)
print("shape2", x1.shape)
x1 = np.expand_dims(x1, axis=0)
print("Shape of input image", x1.shape)  # dimensions (batch_size, channels, height, width))
# print(x)


""" Reading the mask """
y1 = cv2.imread(Right_mask, cv2.IMREAD_GRAYSCALE)  #Reading right mask
y1 = cv2.resize(y1, (W, H))
cv2.imshow('Left mask', y1)
cv2.waitKey(0)
#cv2.destroyAllWindows()

y2 = cv2.imread(Left_mask, cv2.IMREAD_GRAYSCALE) #Reading left mask
y2 = cv2.resize(y2, (W, H))
cv2.imshow('Right mask', y2)
cv2.waitKey(0)
#cv2.destroyAllWindows()

# Combining/concatinating the masks

y = y1 + y2
y = cv2.resize(y, (W, H))
cv2.imshow('Combined mask', y)
cv2.waitKey(0)
#cv2.destroyAllWindows()
print(y.shape)
y = np.expand_dims(y, axis=-1)
print("Shape after expand", y.shape)
y = np.concatenate([y, y, y], axis=-1)
print("Shape after Concatenate", y.shape)


""" Predicting the mask. """
y_pred = model.predict(x1)[0]
print("prediction shape", y_pred.shape)
#print(y_pred)
cv2.imshow('Predicted mask', y_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()
#y_pred = y_pred.astype(np.int32)
#Predicting Mask
y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
print("predicted mask shape", y_pred.shape)

sep_line = np.ones((H, 10, 3)) * 255

cat_image = np.concatenate([x, sep_line, y, sep_line, y_pred * 255], axis=1)
print("Shape after concatination of Ori, Grnd_trth and prediction", cat_image.shape)

save_image_path = f"results/Prediction.png"
cv2.imwrite(save_image_path, cat_image)

#cv2.imshow('Final image', y_pred)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

