import threading
import ctypes
import keyboard
import win32gui
import mouse
import XInput
import time
import torch
import os

#region ctypes Structures for Raw Input
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
#endregion

#region Raw Input Listener (mouse only)
mouse_deltas = [0, 0]  # [delta_x, delta_y]
mouse_lock = threading.Lock()
user32_library = ctypes.windll.user32

def listen_for_mouse_movement(kill_event: threading.Event) -> None:
    """Listens for raw mouse input and updates global mouse delta values."""
    instance_handle = win32gui.GetModuleHandle(None)
    class_name = "RawInputWindow"
    window = win32gui.WNDCLASS()
    window.hInstance = instance_handle
    window.lpszClassName = class_name
    window.lpfnWndProc = _raw_input_window_procedure
    win32gui.RegisterClass(window)
    window_handle = win32gui.CreateWindow(class_name, "Raw Input Hidden Window", 0, 0, 0, 0, 0, 0, 0, instance_handle, None)
    
    device = RAWINPUTDEVICE()
    device.usUsagePage = 0x01   # Generic Desktop Controls
    device.usUsage = 0x02       # Mouse
    device.dwFlags = 0x00000100 # RIDEV_INPUTSINK: receive input even when unfocused
    device.hwndTarget = window_handle
    if not user32_library.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
        raise ctypes.WinError()

    while not kill_event.is_set():
        win32gui.PumpWaitingMessages()
        time.sleep(0.001)

def _raw_input_window_procedure(window_handle: ctypes.wintypes.HWND, message: ctypes.wintypes.UINT, input_code: ctypes.wintypes.WPARAM, data_handle: ctypes.wintypes.LPARAM) -> ctypes.c_long:
    """Window procedure to handle raw input messages."""
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
#endregion

#region Device Polling Functions
def poll_keyboard(keyboard_whitelist: list) -> list:
    """Returns a list of state values for all binds in the given whitelist."""
    return [1 if keyboard.is_pressed(key) else 0 for key in keyboard_whitelist]

def poll_mouse(mouse_whitelist: list) -> list:
    """Returns a list of state values for all binds in the given whitelist."""
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

def poll_gamepad(gamepad_whitelist: list) -> list:
    """Returns a list of state values for all binds in the given whitelist."""
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

def is_pressed(bind: str) -> bool:
    """Determines whether a one-dimensional bind is pressed. (works on all supported devices)"""
    if not bind:
        return True
    try:
        if mouse.is_pressed(bind):
            return True
    except:
        pass
    try:
        if keyboard.is_pressed(bind):
            return True
    except:
        pass
    try:
        gamepad_state = XInput.get_state(0)
        button_values = XInput.get_button_values(gamepad_state)
        if button_values[bind]:
            return True
    except:
        pass
    try:
        gamepad_state = XInput.get_state(0)
        trigger_values = XInput.get_trigger_values(gamepad_state)
        if bind == 'LT' and trigger_values[0] > 0:
            return True
        elif bind == 'RT' and trigger_values[1] > 0:
            return True
    except:
        pass
    return False
#endregion

#region Processor Optimizations
TORCH_DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DEVICE = torch.device(TORCH_DEVICE_TYPE)
CPU_WORKERS = max(os.cpu_count()//2, 2) if TORCH_DEVICE_TYPE == 'cuda' else os.cpu_count()//2

def optimize_cuda_for_hardware():
    """Optimizes CUDA settings based on detected hardware capabilities."""
    if not torch.cuda.is_available():
        print("CUDA not available - running on CPU.")
        return
    
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    processor = torch.cuda.get_device_name(device)
    
    has_tensor_cores = (major >= 7)
    
    print(f'Using CUDA device: {processor}')

    if has_tensor_cores:
        torch.set_float32_matmul_precision('medium')
        print("Tensor Cores detected - Using medium matmul precision.")
    else:
        torch.set_float32_matmul_precision('high')
        print("No Tensor Cores - Using high matmul precision.")

    torch.backends.cudnn.benchmark = True
    print("cuDNN benchmark mode enabled (faster convolution selection).")
#endregion