#! /usr/bin/env python
import os
import sys
import csv
import cv2
import glob
import numpy as np
import numpy.matlib
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 100
high_threshold = 110

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 1.2  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.38  # ditto for top edge of trapezoid
trap_height = 0.97  # height of the trapezoid expressed as percentage of image height

# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 #minimum number of pixels making up a line
max_line_gap = 20	# maximum gap in pixels between connectable line segments

alpha = 0.8
beta = 1.
gamma = 0.

def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
   
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
	"""
	NOTE: this is the function you might want to use as a starting point once you want to 
	average/extrapolate the line segments you detect to map out the full
	extent of the lane (going from the result shown in raw-lines-example.mp4
	to that shown in P1_example.mp4).  
	
	Think about things like separating line segments by their 
	slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	line vs. the right line.  Then, you can average the position of each of 
	the lines and extrapolate to the top and bottom of the lane.
	
	This function draws `lines` with `color` and `thickness`.	
	Lines are drawn on the image inplace (mutates the image).
	If you want to make the lines semi-transparent, think about combining
	this function with the weighted_img() function below
	"""
	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		
		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		# Filter lines based on slope
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)
		
	lines = new_lines
	
	# Split lines into right_lines and left_lines, representing the right and left lane lines
	# Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)
			
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
	
	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False
		
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False
	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - trap_height)
	
	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m
	
	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m
	
	# Convert calculated end points from float to int
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)
	
	# Draw the right and left lines on image
	if draw_right:
		cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
	if draw_left:
		cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

	
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

	
    (x1, x2) = img.shape
    dt = np.dtype(np.uint8)
    line_img = np.zeros((x1, x2, 3), dt)

    draw_lines(line_img, lines)
    return line_img

if __name__ == "__main__":

	#img2 = cv2.imread("test5.jpg", 1)
	vc = cv2.VideoCapture(1)

	while vc.isOpened():
		
		rval, frame = vc.read()

		grey = grayscale(frame)
		blur_grey = gaussian_blur(grey, kernel_size)

		edges = canny(blur_grey, low_threshold, high_threshold)
		
		imshape = frame.shape
		vertices = np.array([[\
			((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),\
			((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
			(imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
			(imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]]\
			, dtype=np.int32)

		masked_image = region_of_interest(edges, vertices)

		line_image = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)

		final_image = frame.astype('uint8')

		cv2.imshow('astype', final_image)	

		cv2.addWeighted(frame, alpha, line_image, beta, gamma, final_image)

		final_image2 = final_image.astype('uint8')

		cv2.imshow('test2', masked_image)	
		cv2.imshow('test', final_image2)	

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	vc.release()
	cv2.destroyAllWindows()
