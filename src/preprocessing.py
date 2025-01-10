import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
import cv2

data_dir='../data/' 

processed_images=[]
labels=[]

def preprocess_image(image_path):
    img=cv2.imread(image_path)
    img_cvt=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #By default, cv2 reads channels in the order BGR. Convert the read order to RGB

    img_resized=cv2.resize(img_cvt, (128, 128), interpolation=cv2.INTER_CUBIC) #Resize to 128x128 with interpolation flag set to INTER_CUBIC 
    img_normalized=(img_resized/255.0).astype(np.float32)

    return img_normalized

for student_folder in os.listdir(data_dir):
    student_folder_path=os.path.join(data_dir, student_folder)

    if os.path.isdir(student_folder_path):
        for filename in os.listdir(student_folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path=os.path.join(student_folder_path, filename)

                img_preprocessed=preprocess_image(img_path)

                processed_images.append(img_preprocessed)
                labels.append(student_folder)

processed_images=np.array(processed_images) #Convert to np array
labels=np.array(labels)

X_train, X_val, y_train, y_val=train_test_split(processed_images, labels, test_size=0.2)

preprocessed_data_save_path='../src/preprocessed_data'
np.savez_compressed(preprocessed_data_save_path, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)