import sklearn.preprocessing
import tkinter.filedialog
import tkinter.ttk
import threading
import keyboard
import win32gui
import tkinter
import XInput
import pandas
import ctypes
import torch
import numpy
import mouse
import json
import time
import os
import models

# =============================================================================
# Data Scaler
# =============================================================================
SCALER = sklearn.preprocessing.StandardScaler()

def fit_global_scaler(files: list[str], whitelist: list[str]) -> None:
    """Fits the global scaler on the given files."""
    scalable_columns = [col for col in SCALABLE_FEATURES if col in whitelist]
    if not scalable_columns:
        return

    all_data = []
    for file in files:
        df = pandas.read_csv(file, usecols=lambda col: col in scalable_columns)
        all_data.append(df)

    if all_data:
        combined_df = pandas.concat(all_data, ignore_index=True)
        SCALER.fit(combined_df[scalable_columns])

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
mouse_deltas = [0, 0]  # [delta_x, delta_y]
mouse_lock = threading.Lock()
user32_library = ctypes.windll.user32

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
# Device Polling Functions
# =============================================================================
def poll_keyboard(keyboard_whitelist: list) -> list:
    return [1 if keyboard.is_pressed(key) else 0 for key in keyboard_whitelist]

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

def is_pressed(capture_bind: str) -> bool:
    """Determines whether data capture should occur based on the capture bind."""
    if not capture_bind:
        return True
    try:
        if mouse.is_pressed(capture_bind):
            return True
    except:
        pass
    try:
        if keyboard.is_pressed(capture_bind):
            return True
    except:
        pass
    try:
        gamepad_state = XInput.get_state(0)
        button_values = XInput.get_button_values(gamepad_state)
        if button_values[capture_bind]:
            return True
    except:
        pass
    try:
        gamepad_state = XInput.get_state(0)
        trigger_values = XInput.get_trigger_values(gamepad_state)
        if capture_bind == 'LT' and trigger_values[0] > 0:
            return True
        elif capture_bind == 'RT' and trigger_values[1] > 0:
            return True
    except:
        pass
    return False

def poll_if_capturing(config: dict) -> list:
    """Polls input devices if capture bind(s) are pressed."""
    capturing = True
    capture_binds = config.get('capture_bind_list', [])
    if len(capture_binds) > 1:
        pressed_capture_binds = [is_pressed(bind) for bind in capture_binds]
        if config.get('capture_bind_logic') == 'ANY':
            capturing = any(pressed_capture_binds)
        else:
            capturing = all(pressed_capture_binds)
    elif not is_pressed(capture_binds[0]):
        capturing = False
    if capturing:
        row = poll_keyboard(config.get('keyboard_whitelist')) + poll_mouse(config.get('mouse_whitelist')) + poll_gamepad(config.get('gamepad_whitelist'))
        if config.get('ignore_empty_polls') and not (row.count(0) == len(row)):
            return row
        elif not config.get('ignore_empty_polls'):
            return row
    return None

def should_kill(config: dict) -> bool:
    """Determines whether the program should be terminated based on kill binds."""
    kill_bind_list = config.get('kill_bind_list', [])
    if not kill_bind_list:
        return False
    pressed_kill_binds = [is_pressed(bind) for bind in kill_bind_list]
    if config.get('kill_bind_logic') == 'ANY':
        return any(pressed_kill_binds)
    else: # 'ALL'
        return all(pressed_kill_binds)

# =============================================================================
# Utility Classes
# =============================================================================
class InputDataset(torch.utils.data.Dataset):
    """Dataset for loading input data from CSV files."""
    def __init__(self: object, file_path: str, config: dict, label: int = 0):
        self.polls_per_sequence = config.get('polls_per_sequence')
        self.label = label
        whitelist = config.get('keyboard_whitelist') + config.get('mouse_whitelist') + config.get('gamepad_whitelist')
        data_frame = pandas.read_csv(file_path)
        self.feature_columns = [col for col in whitelist if col in data_frame.columns]
        data_frame = self.scale_features(data_frame)
        data_array = self.to_numpy_array(data_frame)
        data_array = self.trim_to_polls_per_sequence(data_array)
        if config.get('ignore_empty_polls'):
            data_array = self.filter_out_empty_polls(data_array)
        self.data_tensor = torch.from_numpy(data_array)

    def scale_features(self, data_frame: pandas.DataFrame) -> pandas.DataFrame:
        """Scales specified columns of the DataFrame."""
        df_copy = data_frame.copy()
        scalable_columns = [col for col in SCALABLE_FEATURES if col in self.feature_columns]
        if scalable_columns:
            df_copy.loc[:, scalable_columns] = SCALER.transform(df_copy[scalable_columns].astype(numpy.float32))
        return df_copy

    def to_numpy_array(self, data_frame: pandas.DataFrame) -> numpy.ndarray:
        """Converts the DataFrame to a NumPy array of float32."""
        return data_frame[self.feature_columns].values.astype(numpy.float32)

    def trim_to_polls_per_sequence(self, data_array: numpy.ndarray) -> numpy.ndarray:
        """Trims the array to be divisible by the sequence length."""
        remainder = len(data_array) % self.polls_per_sequence
        if remainder != 0:
            return data_array[:-remainder]
        return data_array
    
    def filter_out_empty_polls(self, data_array: numpy.ndarray) -> numpy.ndarray:
        """Filters out any empty rows (rows of all zeros)"""
        return data_array[data_array.sum(axis=1) != 0]

    def __len__(self):
        return len(self.data_tensor) // self.polls_per_sequence

    def __getitem__(self, idx):
        start_idx = idx * self.polls_per_sequence
        seq = self.data_tensor[start_idx : start_idx + self.polls_per_sequence]
        return seq, torch.tensor(self.label, dtype=torch.float32)

# =============================================================================
# Configuration
# =============================================================================
KEY_BINDS = (
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', 
        '/', '.', ',', '<', '>', '?', '!', '@', '#', '$', '%', '^', '&', 
        '*', '(', ')', '_', '=', '{', '}', '[', ']', '|', '\\', ':', ';', 
        ' ', '~', 'enter', 'esc', 'backspace', 'tab', 'space', 'caps lock', 
        'num lock', 'scroll lock', 'home', 'end', 'page up', 'page down', 
        'insert', 'delete', 'left', 'right', 'up', 'down', 'f1', 'f2', 'f3', 
        'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'print screen', 
        'pause', 'break', 'windows', 'menu', 'right alt', 'ctrl', 
        'left shift', 'right shift', 'left windows', 'left alt', 'right windows', 
        'alt gr', 'windows', 'alt', 'shift', 'right ctrl', 'left ctrl'
    )
MOUSE_BINDS = ('left', 'right', 'middle', 'x', 'x2', 'deltaX', 'deltaY')
GAMEPAD_BINDS = (
        'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'START', 'BACK', 
        'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
        'A', 'B', 'X', 'Y', 'LT', 'RT', 'LX', 'LY', 'RX', 'RY'
    )
SCALABLE_FEATURES = ('deltaX', 'deltaY', 'LX', 'LY', 'RX', 'RY')
BINDABLE_FEATURES = [feature for feature in (KEY_BINDS + MOUSE_BINDS + GAMEPAD_BINDS) if feature not in SCALABLE_FEATURES]
INPUT_GATES = ('ANY', 'ALL')

def get_config_from_gui() -> dict:
    return json.load(tkinter.filedialog.askopenfile(title='Select configuration file', filetypes=[('JSON Files', '*.json')]))

def get_model_class_from_gui(config: dict) -> None:
    print(models.AVAILABLE_MODELS)
    root = tkinter.Tk()
    frame = tkinter.ttk.Frame(padding=5)
    frame.pack()
    tkinter.ttk.Label(frame, text='Select model class:').pack()
    combo_box = tkinter.ttk.Combobox(frame, values=list(models.AVAILABLE_MODELS.keys()))
    combo_box.pack()
    def save_and_exit() -> None:
        config['model_class'] = models.AVAILABLE_MODELS[combo_box.get()]
        root.destroy()
    tkinter.ttk.Button(frame, text='OK', command=save_and_exit).pack()
    root.mainloop()

def get_model_file_from_gui(config: dict) -> None:
    config['model_file'] = tkinter.filedialog.askopenfilename(title='Select model file', filetypes=[('Checkpoint Files', '*.ckpt')])

def get_training_files_from_gui(config: dict) -> None:
    config['training_files'] = tkinter.filedialog.askopenfilenames(title='Select training files', filetypes=[('CSV Files', '*.csv')])
    if not config.get('model_class'):
        return
    if config.get('model_class').training_type == 'supervised':
        config['cheat_training_files'] = tkinter.filedialog.askopenfilenames(title='Select cheat training files', filetypes=[('CSV Files', '*.csv')])

def get_validation_files_from_gui(config: dict) -> None:
    config['validation_files'] = tkinter.filedialog.askopenfilenames(title='Select validation files', filetypes=[('CSV Files', '*.csv')])
    if not config.get('model_class'):
        return
    if config.get('model_class').training_type == 'supervised':
        config['cheat_validation_files'] = tkinter.filedialog.askopenfilenames(title='Select cheat validation files', filetypes=[('CSV Files', '*.csv')])

def get_testing_files_from_gui(config: dict) -> None:
    config['testing_files'] = tkinter.filedialog.askopenfilenames(title='Select testing files', filetypes=[('CSV Files', '*.csv')])

def get_save_dir_from_gui(config: dict) -> None:
    config['save_dir'] = tkinter.filedialog.askdirectory(title='Select save directory')

def validate_config(config: dict) -> bool:
    list_validations = {
        'keyboard_whitelist': KEY_BINDS,
        'mouse_whitelist': MOUSE_BINDS,
        'gamepad_whitelist': GAMEPAD_BINDS,
        'kill_bind_list': BINDABLE_FEATURES,
        'capture_bind_list': BINDABLE_FEATURES,
    }

    print(config.get('kill_bind_list'))
    print(config.get('capture_bind_list'))

    for key, allowed in list_validations.items():
        for bind in config.get(key, []):
            if bind not in allowed:
                print(f'Invalid {key[:-10] if key.endswith("_whitelist") else key} bind: {bind}')
                return False

    if config.get('kill_bind_logic') not in INPUT_GATES:
        print('Invalid kill bind logic config')
        return False

    if config.get('capture_bind_logic') not in INPUT_GATES:
        print('Invalid capture bind logic config')
        return False

    if config.get('model_class') not in models.get_available_models().values():
        print('Invalid model class config')
        return False

    file_keys = [
        'model_file',
        'training_files',
        'validation_files',
        'cheat_training_files',
        'cheat_validation_files',
        'testing_files'
    ]

    for key in file_keys:
        for path in config.get(key, []):
            if not os.path.isfile(path):
                print(f'Invalid file in {key}: {path}')
                return False
            
    if not config.get('deployment_window_type') not in ('sliding', 'tumbling'):
        print('Invalid deployment window type config')
        return False
            
    if not config.get('polls_per_sequence'):
        print('Missing sequence length config')
        return False
    
    if not config.get('sequences_per_batch'):
        print('Missing batch size config')
        return False

    return True

# =============================================================================
# Hardware Compatibility
# =============================================================================
def optimize_cuda_for_hardware():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        processor = torch.cuda.get_device_name(device)
        
        has_tensor_cores = (major >= 7)
        
        print(f"CUDA device: {processor}")

        if has_tensor_cores:
            torch.set_float32_matmul_precision('medium')
            print("Tensor Cores detected → Using MEDIUM matmul precision.")
        else:
            torch.set_float32_matmul_precision('high')
            print("No Tensor Cores → Using HIGH matmul precision.")

        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark mode enabled (faster convolution selection).")
    else:
        print("CUDA not available — running on CPU.")