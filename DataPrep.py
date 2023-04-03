# Images in "MaskImages" folder have not the same shape
# So we should reshape them and devide the reshaped images into 2 folders
# "Train" folder with 2 subfolders "MaskOn" and "MaskOff"
# and "Valid" folder with 2 subfolders "MaskOn" and "MaskOff"

import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

def reshape_100(img,path):
    image = cv2.imread("MaskImages/"+img)
    original_shape = image.shape
    new_shape = (100, 100)
    reshaped_image = cv2.resize(image, new_shape)
    cv2.imwrite(path+img, reshaped_image)

labels=pd.read_csv("train_labels.csv")
X_train, X_valid = train_test_split(labels, test_size=0.2)

for i in range(X_valid.shape[0]):
    if X_valid.target.iloc[i] == 0 :
        reshape_100(X_valid.image.iloc[i],"Valid/MaskOff/")
    else :
        reshape_100(X_valid.image.iloc[i],"Valid/MaskOn/")
    
for i in range(X_train.shape[0]):
    if X_train.target.iloc[i] == 0 :
        reshape_100(X_train.image.iloc[i],"Train/MaskOff/")
    else :
        reshape_100(X_train.image.iloc[i],"Train/MaskOn/")