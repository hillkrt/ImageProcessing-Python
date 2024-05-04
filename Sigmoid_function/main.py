import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "koyu.tif"
image = cv2.imread(image_path)

#Görüntüyü Gri Tonlama Dönüşümü
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Histogram Hesaplama
histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

def sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-a*(x-b)))

#Sigmoid Fonksiyonu
processed_image = sigmoid(gray_image, a=0.0005, b=-20)

#Sonuç
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Processed Image")
plt.imshow(processed_image, cmap='gray')
plt.axis('off')

plt.show()
