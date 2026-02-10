from time import sleep
from cv2_lib import ShopItemDetector, cv2
import adb_lib
import ocr_lib

BASE_SCREENSHOT_PATH: str = 'temp'
SLOW_FACTOR: float = 1

def read_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Automate shop interactions in a mobile game.')
    parser.add_argument('--slow-factor', type=float, default=1, help='Factor to slow down interactions (default: 1)')
    args = parser.parse_args()
    
    global SLOW_FACTOR
    SLOW_FACTOR = args.slow_factor

def get_currencies_from_screenshot(screenshot: str) -> dict[str, int]:
    currencies = ocr_lib.detect_currencies(cv2.imread(f'{BASE_SCREENSHOT_PATH}/{screenshot}'))
    print('='*80)
    print('gold:', currencies['gold'])
    print('skystones:', currencies['skystones'])
    print('='*80)
    
    return { 'gold': currencies['gold'], 'skystones': currencies['skystones'] }

def press_center_of_button(button: tuple[int, int, int, int]):
    x, y, w, h = button
    center_x = x + w // 2
    center_y = y + h // 2
    adb_lib.tap((center_x, center_y))
    
def press_confirm_purchase_button(screen_dimensions: tuple[int, int]):
    screen_x, screen_y = screen_dimensions
    accept_x = int(screen_x * 0.6)
    accept_y = int(screen_y * 0.7)
    adb_lib.tap((accept_x, accept_y))
    
def press_confirm_refresh_button(screen_dimensions: tuple[int, int]):
    screen_x, screen_y = screen_dimensions
    accept_x = int(screen_x * 0.6)
    accept_y = int(screen_y * 0.65)
    adb_lib.tap((accept_x, accept_y))
    
def swipe_shop_items_to_the_bottom(screen_dimensions: tuple[int, int], duration_ms: int = 1_000):
    screen_x, screen_y = screen_dimensions
    start_x = int(screen_x * 0.7)
    start_y = int(screen_y * 0.9)
    end_x = start_x
    end_y = int(screen_y * 0.3)
    adb_lib.swipe(
        (start_x, start_y),
        (end_x, end_y),
        duration_ms
    )

def main():
    detector = ShopItemDetector()
    initial = adb_lib.take_screenshot(base_url=BASE_SCREENSHOT_PATH)
    # currencies = get_currencies_from_screenshot(screenshot)
    refresh_button = detector.detect_refresh_button(cv2.imread(f'{BASE_SCREENSHOT_PATH}/{initial}'))[0]
    adb_lib.delete_screenshot(f'{BASE_SCREENSHOT_PATH}/{initial}')
    print('\n'+'='*80)
    print('Refresh Button:', refresh_button)
    print('='*80)
    should_swipe_down = True
    while True:
        screenshot = adb_lib.take_screenshot(base_url=BASE_SCREENSHOT_PATH)
        detections = detector.match_template(
            cv2.imread(f'{BASE_SCREENSHOT_PATH}/{screenshot}'),
            [detector.templates['tc'], detector.templates['tm']],
        )
        buy_buttons = detector.find_green_buy_buttons(cv2.imread(f'{BASE_SCREENSHOT_PATH}/{screenshot}'))
        # matching_buy_buttons = detector.match_with_green_button(detections, buy_buttons)
        matching_buy_buttons = detector.estimate_green_button_position(detections, adb_lib.DEVICE_DIMENSIONS)
        
        for matches in matching_buy_buttons:
            # x, y, w, h = matches['item_bbox']
            button_x, button_y, button_w, button_h = matches['buy_button_bbox']
            sleep(0.5*SLOW_FACTOR)
            press_center_of_button((button_x, button_y, button_w, button_h))
            sleep(0.5*SLOW_FACTOR)
            press_confirm_purchase_button(adb_lib.DEVICE_DIMENSIONS)
            
        print('\n'+'='*80)
        print('Detected Items:', detections)
        print('='*80)
        # adb_lib.delete_screenshot(f'{BASE_SCREENSHOT_PATH}/{screenshot}')
        
        if should_swipe_down:
            should_swipe_down = not should_swipe_down
            sleep(0.5*SLOW_FACTOR)
            swipe_shop_items_to_the_bottom(adb_lib.DEVICE_DIMENSIONS, 500)
            sleep(0.5)
            continue
        
        should_swipe_down = not should_swipe_down
        sleep(0.5*SLOW_FACTOR)
        press_center_of_button(refresh_button)
        sleep(0.5*SLOW_FACTOR)
        press_confirm_refresh_button(adb_lib.DEVICE_DIMENSIONS)
        sleep(3)

if __name__ == '__main__':
    read_args()
    main()