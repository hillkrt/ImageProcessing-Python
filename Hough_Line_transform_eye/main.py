import cv2
import numpy as np

# Önceden eğitilmiş göz sınıflandırıcısı
goz_siniflandirici = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread("images.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gözleri tespit et
gozler = goz_siniflandirici.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=50, minSize=(10, 10))

for (x, y, w, h) in gozler:
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x + w // 2, y + h // 2), min(w, h) // 7, (255), -1)

    iris = cv2.bitwise_and(gray, mask)

    _, iris_thresh = cv2.threshold(iris, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(iris_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        center = (int(cx), int(cy))
        radius = int(radius)
        cv2.circle(img, center, radius, (0, 255, 0), 1)

cv2.imshow('Gozun Iris Kismi', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
