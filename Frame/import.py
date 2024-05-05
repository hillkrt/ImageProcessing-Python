import cv2
import numpy as np

image = cv2.imread('tarla.png')

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Yeşil renge dayalı olarak bir eşikleme yap
lower_green = np.array([0, 0, 0], dtype="uint8")
upper_green = np.array([70, 255, 70], dtype="uint8")
mask = cv2.inRange(rgb_image, lower_green, upper_green)

# Konturları bul
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

table = []

for i, contour in enumerate(contours):

    if cv2.contourArea(contour) != 0:
        # Konturun merkezini ve boyutlarını hesapla
        moments = cv2.moments(contour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        x, y, w, h = cv2.boundingRect(contour)
        length = w
        width = h
        diagonal = int(np.sqrt(w ** 2 + h ** 2))

        # Gri tonlamalı alanları elde et
        roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Gri tonlamalı alanın enerji ve entropisini hesapla
        energy = np.sum(gray_roi)
        entropy = -np.sum(np.multiply(gray_roi / 255, np.log2(gray_roi / 255 + 1)))

        # Gri tonlamalı alanın ortalama ve medyanını hesapla
        mean = np.mean(gray_roi)
        median = np.median(gray_roi)

        # Tabloya ekle
        table.append([i+1, center, length, width, diagonal, energy, entropy, mean, median])

print("No   Center      Length    Width    Diagonal    Energy   Entropy    Mean      Median")
for row in table:
    center_x, center_y = row[1] # Merkez koordinatlarını ayır
    print("{:<5} {:<12} {:<9} {:<9} {:<12} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(row[0], f"({center_x}, {center_y})", row[2], row[3], row[4], row[5], row[6], row[7], row[8]))
