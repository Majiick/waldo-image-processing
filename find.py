import numpy as np
import cv2
from operator import itemgetter

# https://stackoverflow.com/questions/30331944/finding-red-color-using-python-opencv

img = cv2.imread("Where.jpg")
# img = cv2.imread("test.png")
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
red_mask = mask0 + mask1
cv2.imshow('Red Mask', red_mask)

img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
white_mask = img_hls[:,:,1]
white_mask = cv2.inRange(white_mask, 230, 255)
cv2.imshow('Light Mask', white_mask)

final_mask = cv2.bitwise_or(red_mask, white_mask)
cv2.imshow('Red + Light mask', final_mask)
imgmasked = cv2.bitwise_and(img, img, mask=final_mask)


height, width, _ = imgmasked.shape

region_to_redwhite = dict()
#13x 32y
# for x in range(0, width-13, 13):
#     for y in range(0, height-32, 32):  
#         red_pixel_count = np.count_nonzero(red_mask[x:x+13, y:y+32])
#         white_pixel_count = np.count_nonzero(white_mask[x:x+13, y:y+32])
#         region_to_redwhite[(x, y)] = (red_pixel_count, white_pixel_count)
#         print(white_mask[x:x+13, y:y+32])
#         cv2.rectangle(imgmasked,(x,y),(x,y),(0,255,0),3)

for x in range(13 / 2, width-13, 13):
    for y in range(32 / 2, height-32, 32):  
        red_pixel_count = np.count_nonzero(red_mask[y:y+32, x:x+13])
        white_pixel_count = np.count_nonzero(white_mask[y:y+32, x:x+13])
        region_to_redwhite[(x, y)] = (red_pixel_count, white_pixel_count)
        # print(white_mask[x:x+13, y:y+32])
        # cv2.rectangle(imgmasked,(x,y),(x+13,y+32),(0,255,0),3)
        # cv2.imshow('Final image', imgmasked)
        # cv2.waitKey(0)

region_to_rating = dict()
for k, v in region_to_redwhite.items():
    # The rating is as follows.
    # Say there are 6 pixels in total, the perfect red and white pixel ratio would be 3 red and 3 white.
    # But if there are only 6 red pixels we don't want the rating to be 6.
    # So the rating formula is like this: (3 + 3 - (3 - 3)) / 6 = (red + white - abs(red - white)) / total_pixels
    region_to_rating[k] = (v[0] + v[1] - abs(v[0] - v[1])) / float(13 * 32)

for k, v in sorted(region_to_rating.items(), key=itemgetter(1)):
    print('{}: {}'.format(k, v))
    print(region_to_redwhite[k])
    if v > 0.1:
        cv2.rectangle(imgmasked,k,(k[0]+13, k[1]+32),(0,255,0),3)


cv2.imshow('Final image', imgmasked)
# print(region_to_rating)
cv2.waitKey(0)