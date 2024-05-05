import cv2
import numpy as np 

img = cv2.imread("yol.png")
gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Beyaz renk sınırları
lower_white = np.array([200, 200, 200])
upper_white = np.array([255, 255, 255])
mask_white = cv2.inRange(img, lower_white, upper_white)

# Kenarları bul
kenar = cv2.Canny(mask_white, 100, 250)
cizgi = cv2.HoughLinesP(kenar, 1, np.pi/180, -10)

for i in cizgi:
    x1, y1, x2, y2 = i[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imshow("yol", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
