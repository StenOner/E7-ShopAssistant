from cv2_lib import cv2, DSIZE
import numpy as np

CURRENT_GOLD: int = 0
CURRENT_SKYSTONES: int = 0

def detect_currencies(img: np.ndarray) -> dict[str, int]:
    from paddleocr import PaddleOCR
    
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
    
    return {'gold': CURRENT_GOLD, 'skystones': CURRENT_SKYSTONES}

if __name__ == '__main__':
    pass