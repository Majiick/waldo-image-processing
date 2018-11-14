import numpy as np
import cv2
from operator import itemgetter


def flood_fill(mask, seed_x, seed_y):
    assert(isinstance(mask, np.ndarray))
    if mask[seed_y, seed_x] != 255:
            return []
    height, width = mask.shape

    filled_pixels = set()
    queue = [(seed_x, seed_y)]

    while queue:
        cur_pixel = queue.pop()
        filled_pixels.add(cur_pixel)
        
        try:
            if (cur_pixel[0], cur_pixel[1] + 1) not in filled_pixels and mask[cur_pixel[1] +  1, cur_pixel[0]] == 255:
                queue.append((cur_pixel[0], cur_pixel[1] + 1))
        except IndexError:
            pass

        try:
            if (cur_pixel[0], cur_pixel[1] - 1) not in filled_pixels and mask[cur_pixel[1] - 1, cur_pixel[0]] == 255:
                queue.append((cur_pixel[0], cur_pixel[1] - 1))
        except IndexError:
            pass

        try:
            if (cur_pixel[0] + 1, cur_pixel[1]) not in filled_pixels and mask[cur_pixel[1], cur_pixel[0] + 1] == 255:
                queue.append((cur_pixel[0] + 1, cur_pixel[1]))
        except IndexError:
            pass

        try:
            if (cur_pixel[0] - 1, cur_pixel[1]) not in filled_pixels and mask[cur_pixel[1], cur_pixel[0] - 1] == 255:
                queue.append((cur_pixel[0] - 1, cur_pixel[1]))
        except IndexError:
            pass
    
    return filled_pixels


def detect_waldo_stripes(white_mask, red_mask):
        for y in range(len(white_mask)):
                print(y)
                for x in range(len(white_mask[y])):
                        if white_mask[y, x] == 255:
                                lines = 0
                                first_iter = True
                                for ydown in range(y, len(white_mask)):
                                        if first_iter:
                                                first_iter = False
                                                continue
                                        # white_mask[ydown, x] = 128
                                        # cv2.imshow('Testing', white_mask)
                                        # cv2.waitKey(0)
                                        if white_mask[ydown, x] == 255:
                                                if ydown-1 >= 0:
                                                        if white_mask[ydown-1, x] == 255:
                                                                continue 
                                                lines = lines + 1
                                                
                                                if white_mask[ydown, x] == 0:
                                                        if red_mask[ydown, x] == 0:
                                                                break

                                if lines > 1:
                                        return True

        return False

img = cv2.imread("Where.jpg")
# img = cv2.imread("test.png")
img_hsv=  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0,50,50]) # 0 
upper_red = np.array([5,255,255]) # 10
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([175,50,50]) # 170
upper_red = np.array([180,255,255]) # 180
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join my masks
red_mask = mask0 + mask1
cv2.imshow('Red Mask', red_mask)

img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
white_mask = img_hls[:,:,1]
white_mask = cv2.inRange(white_mask, 235, 255)

kernel = np.ones((1,3),np.uint8)
white_mask_eroded = cv2.erode(white_mask, kernel,iterations = 1)
white_mask_eroded = cv2.morphologyEx(white_mask_eroded, cv2.MORPH_CLOSE, kernel)
done_pixels = set()
for y in range(len(white_mask_eroded)):
        print(y)
        for x in range(len(white_mask_eroded[y])):
                if white_mask_eroded[y][x] == 0 or (x, y) in done_pixels:
                        continue
                pixels = flood_fill(white_mask_eroded, x, y)
                for p in pixels:
                        done_pixels.add(p)

                if len(pixels) > 3 or len(pixels) == 1:
                        for p in pixels:
                                white_mask_eroded[p[1], p[0]] = 0
#white_mask = cv2.erode(white_mask, kernel,iterations = 1)
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

imgmasked_rectangles = imgmasked.copy()
region_to_rating = dict()
for k, v in region_to_redwhite.items():
    # The rating is as follows.
    # Say there are 6 pixels in total, the perfect red and white pixel ratio would be 3 red and 3 white.
    # But if there are only 6 red pixels we don't want the rating to be 6.
    # So the rating formula is like this: (3 + 3 - (3 - 3)) / 6 = (red + white - abs(red - white)) / total_pixels
    region_to_rating[k] = (v[0] + v[1] - abs(v[0] - v[1])) / float(13 * 32)

white_mask_eroded_clean = white_mask_eroded.copy()
for k, v in sorted(region_to_rating.items(), key=itemgetter(1)):
    print('{}: {}'.format(k, v))
    print(region_to_redwhite[k])
    if v > 0.1:
        cv2.rectangle(imgmasked_rectangles,k,(k[0]+13, k[1]+32),(128,128,128),1)

        cv2.rectangle(white_mask_eroded,k,(k[0]+13, k[1]+32),(128,128,128),1)
        if detect_waldo_stripes(white_mask_eroded_clean[k[1]:k[1]+32, k[0]:k[0]+13], red_mask[k[1]:k[1]+32, k[0]:k[0]+13]):
                cv2.rectangle(white_mask_eroded,k,(k[0]+13, k[1]+32),(255,0,0),3)
        for x in range(k[0], k[0] + 13):
                for y in range(k[1], k[1]+32):
                        pass
                        #white_mask_eroded[y, x] = 255


cv2.imshow('Eroded white mask', white_mask_eroded)
cv2.imshow('Final image', imgmasked)
cv2.imshow('Final image rect', imgmasked_rectangles)

# print(region_to_rating)
cv2.waitKey(0)