



import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from call import load_data, tf_dataset
#from PIL import Image
#import PIL

H = 512
W = 512
'''
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)
'''
def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

""" Directory for storing files """
#create_dir("results")
#save_image_path = r'D:\Albot\Medical Imaging\Code\segmentation\results'

""" Load the test images """
#test_images = glob("images/*")

""" Loading model """
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model("files/model.h5")

""" Dataset """
dataset_path = r"D:\Albot\Medical Imaging\Code\segmentation\datasets"
(train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2), (test_x, test_y1, test_y2) = load_data(dataset_path)

""" Predicting the mask """
for x, y1, y2 in tqdm(zip(test_x, test_y1, test_y2), total=len(test_x)):
    """ Extracing the image name. """
    image_name = x.split("/")[-1]
    print(image_name)

'''
    """ Reading the image """
    ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
    ori_x = cv2.resize(ori_x, (W, H))
    #cv2.imshow('Lung Image', ori_x)
    #cv2.waitKey(0)
    x = ori_x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    print("Shape of input image", x.shape) #dimensions (batch_size, channels, height, width))
    #print(x)


    """ Reading the mask """
    ori_y1 = cv2.imread(y1, cv2.IMREAD_GRAYSCALE)
    ori_y1 = cv2.resize(ori_y1, (W, H))
    #cv2.imshow('Ri8 mask', ori_y1)
    #cv2.waitKey(0)
    ori_y2 = cv2.imread(y2, cv2.IMREAD_GRAYSCALE)
    ori_y2 = cv2.resize(ori_y2, (W, H))
    #cv2.imshow('Left mask', ori_y2)
    #cv2.waitKey(0)



       #Combining the masks

    ori_y = ori_y1 + ori_y2
    ori_y = cv2.resize(ori_y, (W, H))
    #cv2.imshow('Combined mask', ori_y)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print(ori_y.shape)
    ori_y = np.expand_dims(ori_y, axis=-1)
    print("Shape after expand", ori_y.shape)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)
    print("Shape after Concatenate", ori_y.shape)



    """ Predicting the mask. """
    y_pred = model.predict(x)[0]
    print("prediction shape", y_pred.shape)
    #print(y_pred)
    #cv2.imshow('Predicted mask', y_pred)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    y_pred = y_pred.astype(np.int32)
    #Predicting Mask
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    print("predicted mask shape", y_pred.shape)




    """ Saving the predicted mask along with the image and GT """
    #save_image_path = f"results/{image_name}"
    #y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    #print("predicted mask shape", y_pred.shape)


    sep_line = np.ones((H, 10, 3)) * 255

    cat_image = np.concatenate([ori_x, sep_line, ori_y, sep_line, y_pred * 255], axis=1)
    print("Shape after concatination of Ori, Grnd_trth and prediction", cat_image.shape)

    #save_image_path = r'D:\Albot\Medical Imaging\Code\segmentation\results'
    #file_path= os.path.join(save_image_path, cat_image)

    save_image_path = f"results/{image_name}"
    cv2.imwrite(save_image_path,cat_image)

    
    folder = 'results'
    os.chdir(folder)
    filename= '.png'
    cv2.imwrite(filename, cat_image)
  
'''
