from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
from matplotlib import pyplot as plt

# Load image
parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='image/PVAZnO_q003.tif')
args = parser.parse_args()

src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Convert to grayscale
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


# Apply Histogram Equalization
dst = cv.equalizeHist(src)

# Get dimensions of image
dimensions = src.shape

# height, width, number of channels in image
height = src.shape[0]
width = src.shape[1]
mask_y = int(height / 15)

# Crop image
y = 0
x = 0
h = height - mask_y
w = width
crop = dst[y:y + h, x:x + w]

# Replace Elements
c_crop = crop.copy()
v_color = 160
c_crop[c_crop < v_color] = 0
c_crop[c_crop >= v_color] = 255

# Resize image
r_src = cv.resize(src, (320, 240))
r_dst = cv.resize(dst, (320, 240))
r_crop = cv.resize(crop, (320, 224))
rc_crop = cv.resize(c_crop, (320, 224))

# Histogram calculation
hist = np.histogram(crop, 256, [0, 256])
plt.hist(crop.ravel(), 256, [0, 256])

# Color calculation
white = 0
black = 0
for i in range(0, 50):
    white += hist[0][i]
for i in range(200, 256):
    black += hist[0][i]


# Display results
# cv.imshow('Source image', r_src)
# cv.imshow('Equalized Image', r_dst)
# cv.imshow('Crop Image', r_crop)
def showInMovedWindow(winname, img, x, y):
    cv.namedWindow(winname)  # Create a named window
    cv.moveWindow(winname, x, y)  # Move it to (x,y)
    cv.imshow(winname, img)


showInMovedWindow('Source image', r_src, 50, 200)
showInMovedWindow('Equalized Image', r_dst, 400, 200)
showInMovedWindow('Crop Image', r_crop, 800, 205)
showInMovedWindow('Crop Replace Elements Image', rc_crop, 1200, 205)


# plt.title('Histogram for gray scale picture')
# plt.show()

# print(type(crop))
print(white)
print(black)
# print(round((white / black) * 100, 3), " %")
print(round((white / (w*h)) * 100, 3), " %")

# Wait until user exits the program
cv.waitKey()
