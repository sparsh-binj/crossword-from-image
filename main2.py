import numpy as np
import cv2

import reporter as RP
import generator as GEN

taskName = 'crucigrama06'

img = cv2.imread("Manual cropped images/crossword_0.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
thresh2 = cv2.bitwise_not(thresh)

rows = 15
cols = 15

contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 1)

max_area = -1

# find contours with maximum area
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        if cv2.contourArea(cnt) > max_area:
            max_area = cv2.contourArea(cnt)
            max_cnt = cnt
            max_approx = approx

# cut the crossword region, and resize it to a standard size of 130x130
x,y,w,h = cv2.boundingRect(max_cnt)
cross_rect = thresh2[y:y+h, x:x+w]

# cv2.imshow("Thresh", thresh)
# cv2.imshow("THresh2", thresh2)
# cv2.imshow("Cross", cross_rect)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#no resizing
#cross_rect = cv2.resize(cross_rect,(rows*10, cols*10))

values = RP.report(cross_rect, rows, cols, w, h)

GEN.generate(cross_rect, rows, cols, w, h, values, taskName)