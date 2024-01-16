import cv2
from pytesseract import pytesseract
import numpy as np
from ultralytics import YOLO
import torch
import re
from roboflow import Roboflow

"""
1. Relies on images being upright and of the same format as training samples
2. Clues list must have vertical lines separating columns of text -- start with across and then down
3. 

"""


"""
1. Run YOLO model to segment image into crossword grid and clues.
2. Use boxed segmentation to then run OCR and grid extraction logic
3. For grid extraction
    (a) Process image and then run edge detection (Canny) with Hough lines
    (b) Detect corners of the grid? Might already be done by the boxed segmentation
    (c) Filter white squares and rest are black (or filter black squares and rest are white)
    (d) For numbers on the grid see 4 (a)
4. OCR Text extraction (keras-ocr or pytesseract)
    (a) Use contours to segregate numbers and individually identify them. Will need to club together nearby contours
        for double digit recognition OR try to parse it in one shot
    (b) Clues recognition done through pytesseract automode
    

5. UI to play crossword
6. Autosolver

"""


def find_grid_and_clues(image_path):
    rf = Roboflow(api_key="sBKzAHQRCL6ucVQ8DuCw")
    project = rf.workspace("random-ideas").project("crossword-image-segmentation")
    model = project.version(2).model

    # model = YOLO("yolov8n.yaml")
    # results = model.train(
    #     data="../datasets/Crossword Image Segmentation.v1i.yolov8/data.yaml",
    #     imgsz=900,
    #     epochs=5,
    #     batch=1,
    #     name="train_result",
    #     device="mps"
    # )

    # infer on a local image
    response = model.predict(image_path, confidence=40, overlap=30)
    predictions = response.json()['predictions']
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

    flag = 0
    grid, clues = {}, {}
    for pred in predictions:
        if flag == 2:
            break
        if pred['class'] == "Clues":
            clues = pred
            flag += 1
        if pred['class'] == "Grid":
            grid = pred
            flag += 1

    img = cv2.imread(image_path)
    y1, y2, x1, x2 = int(grid['y']-grid['height']/2), int(grid['y']+grid['height']/2), \
        int(grid['x']-grid['width']/2), int(grid['x']+grid['width']/2)
    grid_img = cv2.resize(img[y1:y2, x1:x2], (650, 650))
    y1, y2, x1, x2 = int(clues['y'] - clues['height'] / 2), int(clues['y'] + clues['height'] / 2), \
        int(clues['x'] - clues['width'] / 2), int(clues['x'] + clues['width'] / 2)
    clues_img = img[y1:y2, x1:x2]
    return grid_img, clues_img


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bnw = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    # Apply Laplacian filter for image sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    processed = cv2.GaussianBlur(gray, (3,3), 1)
    _, processed = cv2.threshold(processed, 165, 255, cv2.THRESH_BINARY)

    return bnw, sharp, processed


def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_number_contours(contours, img):
    new_img = img.copy()
    valid_contours = [ctr for ctr in contours if 1000 > cv2.contourArea(ctr) > 35]
    cv2.drawContours(new_img, valid_contours, -1, (0, 255, 0), 3)
    pos_list = []
    for i in range(len(valid_contours)):
        (x, y, w, h) = cv2.boundingRect(valid_contours[i])
        pos_list.append(((x, y, w, h), ""))

    for i in range(len(pos_list) - 1):
        for j in range(i + 1, len(pos_list)):
            (x1, y1, w1, h1), _ = pos_list[i]
            (x2, y2, w2, h2), _ = pos_list[j]
            if abs(x1 - x2) <= 15 and abs(y1 - y2) <= 5:
                pos_list[i] = (min(x1, x2), min(y1, y2), w1 + w2 + max(0, x2 - x1 - w1), max(h1, h2)), ""
                del pos_list[j]
                break

    return pos_list, new_img


def show_black_patches(contours, img):
    new_img = img.copy()
    cv2.drawContours(new_img, contours, -1, (0, 255, 0), thickness=3)

    # Display the original image and the mask highlighting black patches
    cv2.imshow('Original Image', img)
    cv2.imshow('Contours', new_img)


def ocr_grid_numbers(position_number_list, img):
    for i in range(len(position_number_list)):
        (x, y, w, h), _ = position_number_list[i]

        # Extract the region of interest (ROI) containing the number
        cushion = 2
        x1 = max(0, x-cushion)
        x2 = min(650, x+w+cushion)
        y1 = max(0, y-cushion)
        y2 = min(650, y+h+cushion)
        roi = img[y1:y2, x1:x2]

        # Perform OCR on the ROI
        number_text = pytesseract.image_to_string(roi, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

        # Print the recognized number and its position
        position_number_list[i] = (x, y), number_text.strip()
        # print(f"Number: {number_text.strip()}, Position: ({x}, {y})")

    return sorted(position_number_list, key=lambda x: [x[0][1], x[0][0]])


def ocr_clues(img):
    text = pytesseract.image_to_string(img, config='--psm 3 --oem 3 -c tessedit_char_whitelist=" \'0123456789abcdefghijlkmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.!;()"')
    clues_list = text.split("\n")
    clues_list = list(filter(lambda x: x != '', clues_list))
    a_index = clues_list.index('ACROSS')
    d_index = clues_list.index('DOWN')
    across_clues = clean_clues(clues_list[a_index+1:d_index])
    down_clues = clean_clues(clues_list[d_index+1:-1])
    return across_clues, down_clues


def clean_clues(clues_list):
    for i in range(len(clues_list)-1):
        if i >= len(clues_list):
            break
        if re.search(r'\(\d{1,2}\)', clues_list[i]) is None:
            clues_list[i] = clues_list[i].strip() + " " + clues_list[i+1].strip()
            del clues_list[i+1]
    return clues_list


def grid_representation_from_image(img):
    sq_side = 50
    rows = 13
    grid_representation = np.zeros((rows, rows))
    for i in range(rows):
        for j in range(rows):
            grid_sq = img[i*sq_side:(i+1)*sq_side, j*sq_side:(j+1)*sq_side]
            white_count = np.count_nonzero(grid_sq)
            print(f"#: {i},{j}\tWhite: {white_count}\tResult: {white_count >= sq_side*sq_side*0.5}")
            grid_representation[i, j] = 1 if white_count >= sq_side*sq_side*0.7 else 0 if white_count <= sq_side*sq_side*0.4 else 2
    print(grid_representation)


# def incorporate_numbers(grid)


# Create 3 versions of Grid image
img_path = "Original Images/Crossword-00015.jpeg"
grid_img, clues_img = find_grid_and_clues(img_path)
gray_grid_img, sharp_grid_img, processed_grid_img = process_image(grid_img)
gray_clues_img, sharp_clues_img, processed_clues_img = process_image(clues_img)

# Find grid numbers
grid_contour_list = find_contours(processed_grid_img)
contour_img = cv2.drawContours(grid_img, grid_contour_list, -1, (0,255,0), 3)
grid_num_pos_list, grid_num_img = filter_number_contours(grid_contour_list, grid_img)
grid_num_prediction_list = ocr_grid_numbers(grid_num_pos_list, sharp_grid_img)

# Find clues list
across_list, down_list = ocr_clues(clues_img)
# clues_text2 = ocr_clues(sharp_clues_img)

grid_representation_from_image(gray_grid_img)



cv2.imshow('Grid Image', processed_grid_img)
# cv2.imshow('Clues Image', gray_clues_img)
# cv2.imshow('Sharp Image', sharp_grid_img)
# cv2.imshow('Processed Image', processed_clues_img)
cv2.imshow('Contour Image', contour_img)

cv2.waitKey(0)
cv2.destroyAllWindows()







