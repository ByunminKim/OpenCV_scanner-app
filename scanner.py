# 스캐너 앱 만들기
# OCR 엔진을 활용한 글자 인식

import numpy as np
import cv2 as cv

##### step1
def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")

	s= np.sum(pts, axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	
	d = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(d)]
	rect[3] = pts[np.argmax(d)]
	
	return rect

def transform(img, pts):
	
	rect = order_points(pts)
	(tl, tr, br, bl) = rect	
	
	width_1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))	
	width_2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(width_1), int(width_2))
	
	height_1 = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))	
	height_2 = np.sqrt(((bl[0] - tl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(height_1), int(height_2))
	
	dst = np.array([
	         [0, 0],
	         [maxWidth - 1, 0],
	         [maxWidth - 1, maxHeight - 1],
	         [0, maxHeight - 1]], dtype = "float32")
	         
	M = cv.getPerspectiveTransform(rect, dst) 
	warped = cv.warpPerspective(img, M, (maxWidth, maxHeight))      
	 
	return warped	    

## image load
img = cv.imread('receiptF.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (511,668), interpolation=cv.INTER_AREA)
# blur = cv.GaussianBlur(img, (5, 5), 0)
edged = cv.Canny(img, 75, 200)

# 영상에서 영수증만 뽑아내기
# contours 방법
cnts, _ = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv.contourArea, reverse= True)[:5]

for c in cnts:
    per1 = cv.arcLength(c, True)					# 곡선의 길이
    approx = cv.approxPolyDP(c, 0.02*per1, True)	# 외곽선 근사화

    if len(approx) == 4:
        screenCnt = approx
        break

cv.drawContours(img, [screenCnt], -1, (255,255,0), 2)
# cv.imshow('contours', img)
# print(screenCnt)

## 직사각형으로 변환
warped = transform(img, screenCnt.reshape(4, 2))
# cv.imshow('warped', warped)

##### step2
## binarization
thresh = 145
maxValue = 255

_, dst1 = cv.threshold(warped, thresh, maxValue, cv.THRESH_BINARY)
cv.imshow('binary', dst1)

## bluring
blur = cv.GaussianBlur(dst1, (5, 5), 0)
cv.imshow('blur', blur)

## denoise & morphology
# 열기 연산(erosion 후 dilation) 수행하면 노이즈 제거하는 효과가 있다.
# 검은색이 객체기 때문에 연산 순서를 반대로 함
kernel1 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((3, 3), np.uint8)

opn = cv.erode(dst1, kernel2, iterations = 1)
opn = cv.dilate(opn, kernel1, iterations = 1)
cv.imshow('opn', opn)

cls = cv.dilate(dst1, kernel1, iterations = 1)
cls = cv.erode(cls, kernel2, iterations = 1)
cv.imshow('cls', cls)

##### step3
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

d = pytesseract.image_to_data(dst1, output_type=Output.DICT)
print(d.keys())

n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        fin = cv.rectangle(dst1, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imshow('fin', fin)
cv.imwrite('receiptF_output.jpg', fin)

print(pytesseract.image_to_string(fin))

cv.waitKey()
cv.destroyAllWindows()