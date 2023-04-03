# initially all the images are in the "imagesMask" folder 
# so we need to devide them into 2 part 
# "Train" folder with 2 subfolders "MaskOn" and "MaskOff"
# and "Valid" folder with 2 subfolders "MaskOn" and "MaskOff"

import pandas as pd
import shutil
import os

labels = pd.read_csv("train_labels.csv")
from sklearn.model_selection import train_test_split
X_train, X_valid = train_test_split(labels, test_size=0.2)

for i in range(X_valid.shape[0]):
    if X_valid.target.iloc[i] == 0 :
        src_path = "imagesMask/"+X_valid.image.iloc[i]
        dst_path = "Valid/MaskOff/"+X_valid.image.iloc[i]
        shutil.move(src_path, dst_path)
    else :
        src_path = "imagesMask/"+X_valid.image.iloc[i]
        dst_path = "Valid/MaskOn/"+X_valid.image.iloc[i]
        shutil.move(src_path, dst_path)
        
for i in range(X_train.shape[0]):
    if X_train.target.iloc[i] == 0 :
        src_path = "imagesMask/"+X_train.image.iloc[i]
        dst_path = "Valid/MaskOff/"+X_train.image.iloc[i]
        shutil.move(src_path, dst_path)
    else :
        src_path = "imagesMask/"+X_train.image.iloc[i]
        dst_path = "Valid/MaskOn/"+X_train.image.iloc[i]
        shutil.move(src_path, dst_path)
        
        
# There is no need to re-run this file because the "imagesMask" folder is empty