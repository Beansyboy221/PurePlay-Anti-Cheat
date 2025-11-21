import tkinter.filedialog
import threading
import keyboard
import win32gui
import XInput
import ctypes
import mouse
import time
import csv

# =============================================================================
# Global Variables
# =============================================================================
mouse_deltas = [0, 0]  # [delta_x, delta_y]
mouse_lock = threading.Lock()
user32_library = ctypes.windll.user32

# =============================================================================
# ctypes Structures for Raw Input
# =============================================================================
class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", ctypes.wintypes.DWORD),
        ("dwSize", ctypes.wintypes.DWORD),
        ("hDevice", ctypes.wintypes.HANDLE),
        ("wParam", ctypes.wintypes.WPARAM)
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", ctypes.wintypes.USHORT),
        ("ulButtons", ctypes.wintypes.ULONG),
        ("ulRawButtons", ctypes.wintypes.ULONG),
        ("lLastX", ctypes.c_long),
        ("lLastY", ctypes.c_long),
        ("ulExtraInformation", ctypes.wintypes.ULONG)
    ]

class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("mouse",  RAWMOUSE)
    ]

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", ctypes.wintypes.USHORT),
        ("usUsage", ctypes.wintypes.USHORT),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("hwndTarget", ctypes.wintypes.HWND)
    ]

# =============================================================================
# Raw Input Listener (mouse only)
# =============================================================================
def raw_input_window_procedure(window_handle: ctypes.wintypes.HWND, message: ctypes.wintypes.UINT, input_code: ctypes.wintypes.WPARAM, data_handle: ctypes.wintypes.LPARAM) -> ctypes.c_long:
    if message == 0x00FF:
        buffer_size = ctypes.wintypes.UINT(0)
        if not(user32_library.GetRawInputData(data_handle, 0x10000003, None, ctypes.byref(buffer_size), ctypes.sizeof(RAWINPUTHEADER)) == 0):
            return 0
        buffer = ctypes.create_string_buffer(buffer_size.value)
        if user32_library.GetRawInputData(data_handle, 0x10000003, buffer, ctypes.byref(buffer_size), ctypes.sizeof(RAWINPUTHEADER)) == buffer_size.value:
            raw_input_data = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
            if raw_input_data.header.dwType == 0:
                delta_x = raw_input_data.mouse.lLastX
                delta_y = raw_input_data.mouse.lLastY
                with mouse_lock:
                    mouse_deltas[0] += delta_x
                    mouse_deltas[1] += delta_y
    return win32gui.DefWindowProc(window_handle, message, input_code, data_handle)

def listen_for_mouse_movement() -> None:
    instance_handle = win32gui.GetModuleHandle(None)
    class_name = "RawInputWindow"
    window = win32gui.WNDCLASS()
    window.hInstance = instance_handle
    window.lpszClassName = class_name
    window.lpfnWndProc = raw_input_window_procedure
    win32gui.RegisterClass(window)
    window_handle = win32gui.CreateWindow(class_name, "Raw Input Hidden Window", 0, 0, 0, 0, 0, 0, 0, instance_handle, None)
    
    device = RAWINPUTDEVICE()
    device.usUsagePage = 0x01   # Generic Desktop Controls
    device.usUsage = 0x02       # Mouse
    device.dwFlags = 0x00000100 # RIDEV_INPUTSINK: receive input even when unfocused
    device.hwndTarget = window_handle
    if not user32_library.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
         raise ctypes.WinError()

    while True:
         win32gui.PumpWaitingMessages()
         time.sleep(0.001)

# =============================================================================
# Helper Function to Poll Keyboard
# =============================================================================
def poll_keyboard(keyboard_whitelist: list) -> list:
    row = [1 if keyboard.is_pressed(key) else 0 for key in keyboard_whitelist]
    return row

# =============================================================================
# Helper Function to Poll Mouse
# =============================================================================
def poll_mouse(mouse_whitelist: list) -> list:
    row = []
    for button in mouse_whitelist:
        if button in ['left', 'right', 'middle', 'x', 'x2']:
            row.append(1 if mouse.is_pressed(button) else 0)
    if any(key in mouse_whitelist for key in ('deltaX', 'deltaY')):
        with mouse_lock:
            if 'deltaX' in mouse_whitelist:
                row.append(mouse_deltas[0])
            if 'deltaY' in mouse_whitelist:
                row.append(mouse_deltas[1])
            mouse_deltas[0] = 0
            mouse_deltas[1] = 0
    return row

# =============================================================================
# Helper Function to Poll Gamepad
# =============================================================================
def poll_gamepad(gamepad_whitelist: list) -> list:
    row = []
    if not XInput.get_connected()[0]: 
        return [0] * len(gamepad_whitelist)
    gamepad_state = XInput.get_state(0)
    button_values = XInput.get_button_values(gamepad_state)
    for feature in gamepad_whitelist:
        if feature in button_values:
            row.append(1 if button_values[feature] else 0)
        else:
            if feature == 'LT':
                trigger_values = XInput.get_trigger_values(gamepad_state)
                row.append(trigger_values[0])
            elif feature == 'RT':
                trigger_values = XInput.get_trigger_values(gamepad_state)
                row.append(trigger_values[1])
            elif feature in ['LX', 'LY', 'RX', 'RY']:
                left_thumb, right_thumb = XInput.get_thumb_values(gamepad_state)
                if feature == 'LX':
                    row.append(left_thumb[0])
                elif feature == 'LY':
                    row.append(left_thumb[1])
                elif feature == 'RX':
                    row.append(right_thumb[0])
                elif feature == 'RY':
                    row.append(right_thumb[1])
            else:
                row.append(0)
    return row

# =============================================================================
# Collection Mode
# =============================================================================
def collect_input_data(config: dict) -> None:
    kill_key = config['kill_key']
    capture_bind = config['capture_bind']
    polling_rate = config['polling_rate']
    keyboard_whitelist = config['keyboard_whitelist']
    mouse_whitelist = config['mouse_whitelist']
    gamepad_whitelist = config['gamepad_whitelist']

    save_directory = tkinter.filedialog.askdirectory(title='Select data save folder')
    file_name = f"{save_directory}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    if any(key in mouse_whitelist for key in ('deltaX', 'deltaY')):
        raw_input_thread = threading.Thread(target=listen_for_mouse_movement, daemon=True)
        raw_input_thread.start()

    with open(file_name, mode='w', newline='') as file_handle:
        csv_writer = csv.writer(file_handle)
        header = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
        csv_writer.writerow(header)
        print(f'Polling devices for collection (press {kill_key} to stop)...')
        while True:
            if keyboard.is_pressed(kill_key):
                break
            should_capture = True
            if capture_bind:
                should_capture = False
                try:
                    if mouse.is_pressed(capture_bind):
                        should_capture = True
                except:
                    pass
                try:
                    if keyboard.is_pressed(capture_bind):
                        should_capture = True
                except:
                    pass
                try:
                    gamepad_state = XInput.get_state(0)
                    button_values = XInput.get_button_values(gamepad_state)
                    if button_values[capture_bind]:
                        should_capture = True
                except:
                    pass
                try:
                    gamepad_state = XInput.get_state(0)
                    trigger_values = XInput.get_trigger_values(gamepad_state)
                    if capture_bind == 'LT' and trigger_values[0] > 0:
                        should_capture = True
                    elif capture_bind == 'RT' and trigger_values[1] > 0:
                        should_capture = True
                except:
                    pass
            kb_row = poll_keyboard(keyboard_whitelist)
            m_row = poll_mouse(mouse_whitelist)
            gp_row = poll_gamepad(gamepad_whitelist)
            row = kb_row + m_row + gp_row
            if should_capture and not (row.count(0) == len(row)):
                csv_writer.writerow(row)
            time.sleep(1.0 / polling_rate)
    print('Data collection stopped. Inputs saved.')
