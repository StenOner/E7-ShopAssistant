import numpy as np
import cv2

BASE_SCREENSHOT_PATH: str = 'screenshots'
BASE_TEMPLATE_PATH: str = 'templates'
DSIZE: tuple = (1920, 1080)

class ShopItemDetector:
    def __init__(self):
        self.green_lower = np.array([35, 40, 40])
        self.green_upper = np.array([85, 255, 255])
        self.blue_lower = np.array([85, 50, 50])
        self.blue_upper = np.array([130, 255, 255])
        self.min_area = 1_000
        self.aspect_ratio_range = (1.5, 6.0)
        self.templates = {
            'tc': cv2.imread(f'{BASE_TEMPLATE_PATH}/template_covenant.png')[:,:480],
            'tm': cv2.imread(f'{BASE_TEMPLATE_PATH}/template_mystic.png')[:,:480],
        }
        self.scales = np.arange(0.7, 1.2, 0.05).tolist()
        
    def detect_refresh_button(self, img: np.ndarray) -> list[tuple[int, int, int, int]]:
        # resized = cv2.resize(src=img, dsize=DSIZE, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(src=hsv, lowerb=self.green_lower, upperb=self.green_upper)
        blue_mask = cv2.inRange(src=hsv, lowerb=self.blue_lower, upperb=self.blue_upper)

        contours, _ = cv2.findContours(image=green_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        refresh_buttons = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if not (self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]):
                continue

            roi_blue_mask = blue_mask[y:y+h, x:x+w]
            blue_pixels = cv2.countNonZero(roi_blue_mask)
            blue_percentage = (blue_pixels / (w * h)) * 100

            if blue_percentage <= 1.0:
                continue

            refresh_buttons.append((x, y, w, h))
            
        #testing: draw matches
        # for x, y, w, h in refresh_buttons:
        #     cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
        # cv2.imshow('Detected Refresh Button', resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #end testing

        return refresh_buttons
    
    def match_template(self, img: np.ndarray, templates: list[np.ndarray], threshold: float = 0.7) -> list[tuple[int, int, int, int, float]]:
        """
        Perform template matching to find occurrences of template in image.
        
        Args:
            image: Input BGR image
            templates: List of templates to search for
            threshold: Matching threshold (0-1)
            
        Returns:
            List of (x, y, width, height, confidence) for each match
        """
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_templates = [cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) for template in templates]
        
        all_matches = []
        for gray_template in gray_templates:
            for scale in self.scales:
                if scale != 1.0:
                    scaled_template = cv2.resize(gray_template, None, fx=scale, fy=scale)
                else:
                    scaled_template = gray_template
                    
                if scaled_template.shape[0] > gray_image.shape[0] or scaled_template.shape[1] > gray_image.shape[1]:
                    continue
                
                result = cv2.matchTemplate(gray_image, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                locations = np.where(result >= threshold)
                
                h, w = scaled_template.shape
                
                for pt in zip(*locations[::-1]):
                    confidence = result[pt[1], pt[0]]
                    # original_w = int(w / scale)
                    # original_h = int(h / scale)
                    # all_matches.append((pt[0], pt[1], original_w, original_h, confidence))
                    all_matches.append((pt[0], pt[1], w, h, confidence))
                
        matches = self._non_max_suppression(all_matches, overlap_threshold=0.5)
        
        for match in matches:
            x, y, w, h, confidence = match
            img_region = img[int(y):int(y+h), int(x):int(x+w)]
            hsv_img = cv2.cvtColor(src=img_region, code=cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(src=hsv_img, lowerb=self.green_lower, upperb=self.green_upper)
            green_pixels = cv2.countNonZero(green_mask)
            green_percentage = (green_pixels / (w * h)) * 100
            
            if green_percentage > 3.0:
                matches.remove(match)        
        
        #testing: draw matches
        # for match in matches:
        #     x, y, w, h, _ = match
        #     cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 255), 3)
        
        # cv2.imshow('Detected Refresh Button', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #end testing
        
        return matches
    
    def match_with_green_button(self, detections: list[tuple[int, int, int, int, float]], buy_buttons: list[tuple[int, int, int, int]]) -> dict[str, list[tuple[int, int, int, int, float]]]:
        """
        Match detected items with nearby green Buy buttons.
        
        Args:
            detections: List of (x, y, width, height, confidence) for detected items
            buy_buttons: List of (x, y, width, height) for detected Buy buttons
            
        Returns:
            List of (x, y, width, height) for matched items
        """
        matched_items = []
        for x, y, w, h, confidence in detections:
            match_x = int(x)
            match_y = int(y)
            match_w = int(w)
            match_h = int(h)
            
            match_center_y = match_y + match_h // 2
            
            best_button = None
            min_distance = float('inf')
            
            for btn_x, btn_y, btn_w, btn_h in buy_buttons:
                btn_center_y = btn_y + btn_h // 2
                y_diff = abs(btn_center_y - match_center_y)
                
                if btn_x > (match_x + match_w // 2) and y_diff < 100:
                    distance = ((btn_x - match_x) ** 2 + y_diff ** 2) ** 0.5
                    if distance < min_distance:
                        best_button = (btn_x, btn_y, btn_w, btn_h)
                        min_distance = distance
            
            if best_button:
                btn_x, btn_y, btn_w, btn_h = best_button
                
                matched_items.append({
                    'item_bbox': (match_x, match_y, match_w, match_h),
                    'buy_button_bbox': (btn_x, btn_y, btn_w, btn_h),
                    'confidence': confidence
                })
        
        unique_detections = []
        seen_buttons = set()
        
        for match in sorted(matched_items, key=lambda x: x['confidence'], reverse=True):
            btn_key = match['buy_button_bbox'][:2]
            if btn_key not in seen_buttons:
                seen_buttons.add(btn_key)
                unique_detections.append(match)
                    
        return unique_detections
    
    def estimate_green_button_position(self, detections: list[tuple[int, int, int, int, float]], screen_size: tuple[int, int]) -> dict[str, list[tuple[int, int, int, int]]]:
        matched_items = []
        screen_size_x, _ = screen_size
        for x, y, w, h, _ in detections:
            match_x = int(x)
            match_y = int(y)
            match_w = int(w)
            match_h = int(h)
            
            button = (screen_size_x-200, match_y+60, 200, 60)
                
            matched_items.append({
                'item_bbox': (match_x, match_y, match_w, match_h),
                'buy_button_bbox': button,
            })
                    
        return matched_items
    
    def find_green_buy_buttons(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Find all green (available) Buy buttons in the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of (x, y, width, height) for each match
        """
        hsv = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(src=hsv, lowerb=self.green_lower, upperb=self.green_upper)
        
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
        green_mask = cv2.morphologyEx(src=green_mask, op=cv2.MORPH_CLOSE, kernel=kernel)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        buy_buttons = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            aspect_ratio = w / float(h)
            # if self.aspect_ratio_range[0] < aspect_ratio < self.aspect_ratio_range[1]:
                # buy_buttons.append((x, y, w, h))
            if 180 < w < 400 and 40 < h < 150:
                buy_buttons.append((x, y, w, h))
                
        #testing: draw matches
        # for x, y, w, h in buy_buttons:
        #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # cv2.imshow('Detected Buy Buttons', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #end testing
        
        return buy_buttons
    
    def _non_max_suppression(self, boxes: list[tuple[int, int, int, int, float]], 
                             overlap_threshold: float = 0.3) -> list[tuple[int, int, int, int, float]]:
        """Remove overlapping bounding boxes, keeping the one with highest confidence."""
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        scores = boxes[:, 4]
        
        areas = (x2 - x1) * (y2 - y1)
        indices = np.argsort(scores)[::-1]  # Sort by confidence, highest first
        
        selected = []
        while len(indices) > 0:
            i = indices[0]
            selected.append(i)
            
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / areas[indices[1:]]
            
            indices = indices[1:][overlap <= overlap_threshold]
        
        return [tuple(boxes[i]) for i in selected]

if __name__ == '__main__':
    shop = ShopItemDetector()
    match = shop.match_template(cv2.imread('images/a88cb4f6-ed10-47d5-9a3f-b4b50137db58.png'), [shop.templates['tc'], shop.templates['tm']])