import numpy as np
import cv2
import requests
from matplotlib import pyplot as plt
import os
from pathlib import Path
from typing import Callable, Optional, Tuple
from PIL import Image

base_path = "/Users/pedron/Desktop/Polytech/PROJ942/Base/IMG_1.JPG"
output_dir = "/Users/pedron/Desktop/Polytech/PROJ942/Base_Visages"
os.makedirs(output_dir, exist_ok=True)




# -----------------------------------------------------------------------------------
#                                    Crop image
# -----------------------------------------------------------------------------------

def crop_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image.shape
    cv2.imshow('grey',gray_image)
    cv2.waitKey(0)
    print("Image Grise")

    resize_1 = cv2.resize(gray_image, (600, 800)) 
    cv2.imshow('resize_1',resize_1)
    cv2.waitKey(0)

    #load the pre-trained classifier
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    #perform the classifier
    face = face_classifier.detectMultiScale(
        resize_1, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    x1 = face[0][0] - round(face[0][2] / 48)
    y1 = face[0][1] - round(face[0][2] / 6)
    x2 = face[0][0] + face[0][2] + round(face[0][2] / 48)
    y2 = face[0][1] + face[0][3] + round(face[0][2] / 6)

    print(round(face[0][2] / 48))
    print(round(face[0][2] / 6))

    cropped_image = resize_1[y1:y2, x1:x2]
    final = cv2.resize(cropped_image, (92, 112)) 
    
    return final



def read_image(path):
    img = cv2.imread(path)
    height, width, depth = img.shape
    print("Image Lu")
    return img


def main():
    # La constante data_root : “Chemin racine des données (à adapter si la base est déplacée).”
    data_root = Path("/Users/pedron/Desktop/Polytech/PROJ942")
    dataset_path = data_root / "Base_Visages"
    test_path = data_root / "Images_de_test"

    base = read_image(base_path)
    cropped = crop_image(base)
    
    
    
    # ICI
    
    
    output_path = os.path.join(output_dir, "1.pgm")
    
    cv2.imwrite(output_path, cropped)
    print("Image sauvegardée dans :", output_path)
    


    
if __name__ == "__main__":
    main()