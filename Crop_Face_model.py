import numpy as np
import cv2
import requests
from matplotlib import pyplot as plt


# reading an image
url = "https://raw.githubusercontent.com/PierrreLebouc/PROJ942/main/Bill_Gates.jpg"
response = requests.get(url)
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

#im = cv2.imread(img)

height, width, depth = img.shape
print(height, width, depth)
cv2.imshow('original',img)
cv2.waitKey(0)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image.shape
cv2.imshow('grey',gray_image)
cv2.waitKey(0)

#load the pre-trained classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#perform the classifier
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

x1 = face[0][0] - 10
y1 = face[0][1] - 10
x2 = face[0][0] + face[0][2] + 10
y2 = face[0][1] + face[0][3] + 10

cropped_image = gray_image[y1:y2, x1:x2]
cv2.imshow('cropped',cropped_image)
cv2.waitKey(0)



for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('final',img_rgb)
cv2.waitKey(0)

cv2.destroyAllWindows()