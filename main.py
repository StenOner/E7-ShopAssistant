import numpy as np
import cv2

import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

IMAGE_BASE_PATH: str = 'images/'
DSIZE: tuple = (1280, 720)
CURRENT_GOLD: int = 0
CURRENT_SKYSTONES: int = 0

def detect_currencies(img_name: str):
    from paddleocr import PaddleOCR

    img = cv2.imread(f"{IMAGE_BASE_PATH}{img_name}")
    resized = cv2.resize(src=img, dsize=DSIZE, interpolation=cv2.INTER_AREA)
    cropped = resized[:60,700:960]
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        )

    result = ocr.predict(input=cropped)
    if not result or not result[0]:
        raise Exception('Could not find currencies')
    
    global CURRENT_GOLD, CURRENT_SKYSTONES
    [CURRENT_GOLD, CURRENT_SKYSTONES] = [int(currency.replace(',','')) for currency in result[0]['rec_texts']]
 
def detect_shop_items(img_name: str):
    img = cv2.imread(f"{IMAGE_BASE_PATH}{img_name}")
    resized = cv2.resize(src=img, dsize=DSIZE, interpolation=cv2.INTER_AREA)
    
    # Detect "Buy" button regions using color-based detection
    hsv = cv2.cvtColor(src=resized, code=cv2.COLOR_BGR2HSV)
    
    # Green color range for the Buy button
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(src=hsv, lowerb=lower_green, upperb=upper_green)

    # Blue mask to exclude when inside of button
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])
    mask_blue = cv2.inRange(src=hsv, lowerb=blue_lower, upperb=blue_upper)
    
    # Find contours
    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    
    buy_buttons = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000: # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)

            roi_blue = mask_blue[y:y+h, x:x+w]
            blue_pixels = cv2.countNonZero(roi_blue)
            aspect_ratio = w / float(h)

            if 1.5 < aspect_ratio < 4.0 and blue_pixels == 0:
                buy_buttons.append((x, y, w, h))
    
    print(f"Found {len(buy_buttons)} items")
    
    # Draw rectangles around detected items
    for (x, y, w, h) in buy_buttons:
        # Draw the buy button
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Estimate the full item region (extending left from the button)
        item_x = max(0, x - 650)
        item_y = max(0, y - 60)
        item_w = x - item_x + w
        item_h = h + 80
        cv2.rectangle(resized, (item_x, item_y), (item_x + item_w, item_y + item_h), (255, 0, 0), 2)
    
    cv2.imshow('Detected Items', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return buy_buttons

def main():
    images = [
        "test1_fail.jpeg",
        "test2_fail.jpeg",
        # "test1_success.jpg",
        # "test2_success.jpg",
        # "test1_success_achieved.jpg",
        # "test2_success_achieved.jpg",
    ]

    # result = [detect_shop_items(image) for image in images]
    [detect_currencies(image) for image in images]

if __name__ == "__main__":
    main()