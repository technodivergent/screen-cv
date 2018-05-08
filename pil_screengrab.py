import pyscreenshot as ImageGrab
from PIL import ImageGrab
import numpy as np
import cv2

def imgFromClipboard():
	im = ImageGrab.grabclipboard()
	im.save('temp.png')

def screenshot():
	im=ImageGrab.grab(bbox=(460,117,1860,988))
	im.save('temp.png')
	
def opencv():
	img = cv2.imread('temp.png')
	
	draw_img = detect_color(img)
	
	cv2.imshow('img', img)
	cv2.imshow('draw_img', draw_img)
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
def detect_color(img):
	img2 = img.copy()
	
	lower_yellow = np.array([0,254,254])
	upper_yellow = np.array([0,255,255])
	
	mask = cv2.inRange(img, lower_yellow, upper_yellow)
	res = cv2.bitwise_and(img, img, mask=mask)
	
	lsd = cv2.createLineSegmentDetector(0)
	lines = lsd.detect(mask)[0]
	
	dst = detect_corner(res)
	cv2.imshow('dst',dst)
	
	draw_img = lsd.drawSegments(img2, lines)
	
	return draw_img
	
def detect_corner(img):
	img2 = img.copy()
	gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)
	
	dst = cv2.dilate(dst,None)
	
	img2[dst>0.01*dst.max()] = [0,0,255]
	
	return img2

	
if __name__ == '__main__':
	#screenshot()
	opencv()
	detect_color()