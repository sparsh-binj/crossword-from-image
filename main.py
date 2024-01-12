import cv2
from pytesseract import pytesseract
import numpy as np
from roboflow import Roboflow



rf = Roboflow(api_key="sBKzAHQRCL6ucVQ8DuCw")
project = rf.workspace("random-ideas").project("crossword-image-segmentation")
dataset = project.version(1).download("yolov5-obb")

# yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=5 imgsz=640

def process_image(original_image):
    img = original_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply Laplacian filter for image sharpening
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Convert the Laplacian result back to uint8 (8-bit image)
    sharpened_img = np.uint8(np.clip(gray - 1.3*laplacian, 0, 255))

    processed_img = cv2.GaussianBlur(gray, (3,3), 1)

    (T, processed_img) = cv2.threshold(processed_img, 175, 255, cv2.THRESH_BINARY)

    return sharpened_img, processed_img

# Prepare images
img = cv2.imread("Crossword Images/crossword_30 copy.jpeg")
img = cv2.resize(img, (700, 700))
sharp_img, processed_img = process_image(img)

contour_img = img.copy()
contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [ctr for ctr in contours if 1000 > cv2.contourArea(ctr) > 50]
cv2.drawContours(contour_img, valid_contours, -1, (0, 255, 0), 3)

hough_img = img.copy()
hough_img = cv2.Canny(hough_img, 30, 120, None, 3)

# Copy edges to the images that will display the results in BGR
cdstP = cv2.cvtColor(hough_img, cv2.COLOR_GRAY2BGR)
linesP = cv2.HoughLinesP(hough_img, 1, np.pi / 180, 50, None, 50, 10)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 2)



# Extract and recognize numbers using Tesseract OCR
iter = 0
for contour in valid_contours:
    iter += 1
    print(iter)

    x, y, w, h = cv2.boundingRect(contour)

    # Extract the region of interest (ROI) containing the number
    roi = sharp_img[y - 3:y + h+3, x-3:x + w+3]

    # Perform OCR on the ROI
    number_text = pytesseract.image_to_string(roi, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

    # Print the recognized number and its position
    print(f"Number: {number_text.strip()}, Position: ({x}, {y})")


cv2.imshow('Original Image', img)
cv2.imshow('Sharp Image', sharp_img)
cv2.imshow('Processed Image', processed_img)
cv2.imshow('Contour Image', contour_img)
# cv2.imshow('Hough Image', hough_img)
# cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
cv2.waitKey(0)
cv2.destroyAllWindows()






