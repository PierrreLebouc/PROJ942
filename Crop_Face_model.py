import numpy as np
import cv2
import requests
from matplotlib import pyplot as plt
import os


output_dir = "test"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "02.pgm")
path = "/Users/pedron/Desktop/Polytech/PROJ942/PROJ942/IMG_6434.JPG"
img = cv2.imread(path)

height, width, depth = img.shape
print(height, width, depth)
cv2.imshow('original',img)
cv2.waitKey(0)
print("Image de base")

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
cv2.imshow('cropped',cropped_image)
cv2.waitKey(0)

final = cv2.resize(cropped_image, (92, 112)) 
print("Image redimensionnée à :", final.shape[:2])
cv2.imshow('resized',final)
cv2.waitKey(0)

cv2.imwrite(output_path, final)

print("Image sauvegardée dans :", output_path)

cv2.destroyAllWindows()