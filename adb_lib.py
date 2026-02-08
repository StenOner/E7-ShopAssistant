from ppadb.client import Client as AdbClient
from ppadb.device import Device
import subprocess
import shutil
import socket

SELECTED_DEVICE: Device = None
DEFAULT_HOST: str = '127.0.0.1'
DEFAULT_ADB_PORT: int = 5037
DEFAULT_BS_PORT: int = 5555
DEVICE_DIMENSIONS: tuple[int, int] = (1920, 1080)

def is_adb_running(host: str = DEFAULT_HOST, port: int = DEFAULT_ADB_PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 1
    
def ensure_adb(host: str = DEFAULT_HOST, port: int = DEFAULT_ADB_PORT):
    adb = shutil.which('adb')
    
    if not adb:
        raise RuntimeError('ADB not found in PATH')
    
    if not is_adb_running(host, port):
        subprocess.run([adb, 'start-server'], check=True)

def select_device_bluestacks():
    ensure_adb()
    client = AdbClient(host=DEFAULT_HOST, port=DEFAULT_ADB_PORT)
    if not client.remote_connect(host=DEFAULT_HOST, port=DEFAULT_BS_PORT):
        raise RuntimeError('Failed to connect to Bluestacks ADB server.')
    
    global SELECTED_DEVICE
    SELECTED_DEVICE = client.device(f'{DEFAULT_HOST}:{DEFAULT_BS_PORT}')
    
    set_device_dimensions()

def select_device():
    ensure_adb()
    client = AdbClient(host=DEFAULT_HOST, port=DEFAULT_ADB_PORT)
    devices = client.devices()

    if not devices:
        raise RuntimeError('No ADB devices found')

    print(' ───────────\n','|','Devices','|','\n ───────────')
    for index, d in enumerate(devices):
        print(f'{index+1}.', d.serial)

    invalid = True
    while invalid:
        try:
            device_index = int(input('Select a device: '))

            global SELECTED_DEVICE
            SELECTED_DEVICE = devices[device_index-1]
            invalid = False
            
            set_device_dimensions()
        except:
            print('Invalid index or value.')
            
def device_not_selected(func):
    def wrapper(*args, **kwargs):
        if not SELECTED_DEVICE:
            select_device_bluestacks()
        return func(*args, **kwargs)
    return wrapper

@device_not_selected
def set_device_dimensions() -> None:
    output = SELECTED_DEVICE.shell('wm size')
    dimensions_str = output.strip().split(':')[-1].strip()
    width, height = map(int, dimensions_str.split('x'))
    
    global DEVICE_DIMENSIONS
    DEVICE_DIMENSIONS = (width, height)

@device_not_selected
def tap(coordinates: tuple = (0,0)):
    SELECTED_DEVICE.shell(cmd=f'input tap {coordinates[0]} {coordinates[1]}')

@device_not_selected
def long_tap(coordinates: tuple = (0,0), duration: int = 1_000):
    SELECTED_DEVICE.shell(cmd=f'input swipe {coordinates[0]} {coordinates[1]} {coordinates[0]} {coordinates[1]} {duration}')

@device_not_selected
def swipe(starting: tuple = (0,0), ending: tuple = (0,0), duration: int = 1_000):
    SELECTED_DEVICE.shell(cmd=f'input swipe {starting[0]} {starting[1]} {ending[0]} {ending[1]} {duration}')

@device_not_selected
def type(text: str):
    SELECTED_DEVICE.shell(cmd=f'input text {text}')

@device_not_selected
def take_screenshot(base_url: str = 'temp') -> str:
    screenshot_name = generate_random_name()
    with open(f'{base_url}/{screenshot_name}', 'wb') as f:
        f.write(SELECTED_DEVICE.screencap())

    return screenshot_name

def generate_random_name(extension: str = 'png') -> str:
    from uuid import uuid4
    
    file_name: str = str(uuid4())

    return f'{file_name}.{extension}'

def delete_screenshot(path: str):
    import os
    
    os.remove(path)

if __name__ == '__main__':
    pass