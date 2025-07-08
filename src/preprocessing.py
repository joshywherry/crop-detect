import os
import cv2
import numpy as np

def load_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)

    for label in class_names:
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (128, 128))  # adjust as needed
                    images.append(img)
                    labels.append(label)

    return np.array(images), np.array(labels), class_names
