from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
from matplotlib import pyplot as plt

# Load image
parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='image/Si-PVA_q004.tif')
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
mask_y = src.shape[0]/15

# Crop image
y = 0
x = 0
h = height - mask_y
w = width
crop = dst[y:y + h, x:x + w]

# Histogram calculation
hist = np.histogram(crop, 256, [0, 256])
plt.hist(crop.ravel(), 256, [0, 256])

# Display results
cv.imshow('Source image', src)
cv.imshow('Equalized Image', dst)
cv.imshow('Crop Image', crop)

plt.title('Histogram for gray scale picture')
plt.show()

print(hist[0])

sum_white = 0
sum_black = 0
for i in range(20):
    sum_white += hist[0][i]
for i in range(225, 250):
    sum_black += hist[0][i]
print(sum_white)
print((sum_white / sum_black) * 100, " %")
# Wait until user exits the program
cv.waitKey()
