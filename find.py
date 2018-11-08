import numpy as np
import cv2

# https://stackoverflow.com/questions/30331944/finding-red-color-using-python-opencv

img = cv2.imread("Where.jpg")
img_hsv=  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
mask = mask0 + mask1
cv2.imshow('Red Mask', mask)

img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
mask3 = img_hls[:,:,1]
mask3 = cv2.inRange(mask3, 230, 255)
cv2.imshow('Light Mask', mask3)

final_mask = cv2.bitwise_or(mask, mask3)
cv2.imshow('Red + Light mask', final_mask)
imgmasked = cv2.bitwise_and(img, img, mask=final_mask)
cv2.imshow('Final image', imgmasked)

cv2.waitKey(0)