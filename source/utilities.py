import sklearn.preprocessing
import tkinter.filedialog
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
import time
import models

# =============================================================================
# Data Scaler
# =============================================================================
SCALER = sklearn.preprocessing.StandardScaler()
SCALE_COLUMNS = ['deltaX', 'deltaY', 'LX', 'LY', 'RX', 'RY']

def fit_global_scaler(files: list[str], whitelist: list[str]) -> None:
    """Fits the global scaler on the given files."""
    scalable_columns = [col for col in SCALE_COLUMNS if col in whitelist]
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

def poll_all_devices(keyboard_whitelist: list, mouse_whitelist: list, gamepad_whitelist: list) -> list:
    return poll_keyboard(keyboard_whitelist) + poll_mouse(mouse_whitelist) + poll_gamepad(gamepad_whitelist)

# =============================================================================
# Common Functions
# =============================================================================
def prompt_for_csv_files(message: str) -> list[str]:
    """Prompts the user to select CSV files and returns the list of selected file paths."""
    files = tkinter.filedialog.askopenfilenames(
        title=message,
        filetypes=[('CSV Files', '*.csv')]
    )
    return files

def prompt_for_checkpoint_file() -> str:
    """Prompts the user to select a model checkpoint file and returns the file path."""
    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )
    return checkpoint

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

# =============================================================================
# Utility Classes
# =============================================================================
class InputDataset(torch.utils.data.Dataset):
    """Dataset for loading input data from CSV files."""
    def __init__(self: object, file_path: str, sequence_length: int, whitelist: list[str], label: int = 0):
        self.sequence_length = sequence_length
        self.label = label
        
        data_frame = self.load_data(file_path)
        self.feature_columns = self.get_feature_columns(data_frame, whitelist)
        data_frame = self.scale_features(data_frame)
        data_array = self.to_numpy_array(data_frame)
        data_array = self.trim_to_sequence_length(data_array)
        self.data_tensor = torch.from_numpy(data_array)

    def load_data(self, file_path: str) -> pandas.DataFrame:
        """Loads data from a CSV file into a pandas DataFrame."""
        return pandas.read_csv(file_path)

    def get_feature_columns(self, data_frame: pandas.DataFrame, whitelist: list[str]) -> list[str]:
        """Filters DataFrame columns based on a whitelist."""
        return [col for col in whitelist if col in data_frame.columns]

    def scale_features(self, data_frame: pandas.DataFrame) -> pandas.DataFrame:
        """Scales specified columns of the DataFrame."""
        scalable_columns = [col for col in SCALE_COLUMNS if col in self.feature_columns]
        if scalable_columns:
            data_frame.loc[:, scalable_columns] = SCALER.transform(data_frame[scalable_columns])
        return data_frame

    def to_numpy_array(self, data_frame: pandas.DataFrame) -> numpy.ndarray:
        """Converts the DataFrame to a NumPy array of float32."""
        return data_frame[self.feature_columns].values.astype(numpy.float32)

    def trim_to_sequence_length(self, data_array: numpy.ndarray) -> numpy.ndarray:
        """Trims the array to be divisible by the sequence length."""
        remainder = len(data_array) % self.sequence_length
        if remainder != 0:
            return data_array[:-remainder]
        return data_array

    def __len__(self):
        return len(self.data_tensor) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        seq = self.data_tensor[start_idx : start_idx + self.sequence_length]
        return seq, torch.tensor(self.label, dtype=torch.float32)
    
# =============================================================================
# Configuration GUI
# =============================================================================
def get_config_from_gui() -> dict | None:
    """Displays a GUI to get configuration from the user."""
    config = {}
    available_models = models.get_available_models()

    def browse_save_dir():
        directory = tkinter.filedialog.askdirectory()
        if directory:
            save_dir_var.set(directory)

    def on_submit():
        nonlocal config
        config['mode'] = mode_var.get()
        config['kill_bind'] = kill_bind_var.get()
        config['capture_bind'] = capture_bind_var.get()
        config['keyboard_whitelist'] = [item.strip() for item in keyboard_whitelist_var.get().split(',') if item.strip()]
        config['mouse_whitelist'] = [item.strip() for item in mouse_whitelist_var.get().split(',') if item.strip()]
        config['gamepad_whitelist'] = [item.strip() for item in gamepad_whitelist_var.get().split(',') if item.strip()]
        config['model_class'] = available_models.get(model_var.get())
        config['training_files'] = training_files_var.get()
        config['validation_files'] = validation_files_var.get()
        config['cheat_training_files'] = cheat_training_files_var.get()
        config['cheat_validation_files'] = cheat_validation_files_var.get()
        config['testing_files'] = testing_files_var.get()
        config['live_graphing'] = live_graphing_var.get()
        config['save_dir'] = save_dir_var.get()

        try:
            config['polling_rate'] = int(polling_rate_var.get())
            config['sequence_length'] = int(sequence_length_var.get())
            config['batch_size'] = int(batch_size_var.get())
        except ValueError:
            print("Polling rate, sequence length, and batch size must be integers.")
            config = {} # Invalidate config
            win.destroy()
            return

        win.destroy()

        with open(f"{config['save_dir']}/config-{time.strftime('%Y%m%d-%H%M%S')}.txt", 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Configuration saved to ${config['save_dir']}/config-{time.strftime('%Y%m%d-%H%M%S')}.txt")

    def on_mode_change(*args):
        match (mode_var.get()):
            case 'train':
                # Enable training and validation files selection
                train_files_button.config(state='normal')
                val_files_button.config(state='normal')
                test_files_button.config(state='disabled')
                model_menu.config(state='normal')
                if available_models:
                    model_var.set(list(available_models.keys())[0])
            case 'test':
                train_files_button.config(state='disabled')
                val_files_button.config(state='disabled')
                test_files_button.config(state='normal')
                model_menu.config(state='normal')
                if available_models:
                    model_var.set(list(available_models.keys())[0])
            case 'deploy':
                train_files_button.config(state='disabled')
                val_files_button.config(state='disabled')
                test_files_button.config(state='disabled')
                model_menu.config(state='normal')
                if available_models:
                    model_var.set(list(available_models.keys())[0])
            case _:
                train_files_button.config(state='disabled')
                val_files_button.config(state='disabled')
                test_files_button.config(state='disabled')
                model_menu.config(state='disabled')
    
    def on_model_change(*args):
        if model_var.get():
            model_class = available_models.get(model_var.get())
            if model_class.training_type == 'supervised':
                cheat_training_files_button.config(state='normal')
                cheat_validation_files_button.config(state='normal')
            else:
                cheat_training_files_button.config(state='disabled')
                cheat_validation_files_button.config(state='disabled')


    win = tkinter.Toplevel()
    win.title("Configuration")

    # --- Variables ---
    mode_var = tkinter.StringVar(value='collect')
    kill_bind_var = tkinter.StringVar(value='\\')
    capture_bind_var = tkinter.StringVar(value='RT')
    keyboard_whitelist_var = tkinter.StringVar(value='')
    mouse_whitelist_var = tkinter.StringVar(value='deltaX, deltaY')
    gamepad_whitelist_var = tkinter.StringVar(value='LT, LX, LY, RX, RY')
    polling_rate_var = tkinter.StringVar(value='60')
    sequence_length_var = tkinter.StringVar(value='30')
    batch_size_var = tkinter.StringVar(value='16')
    model_var = tkinter.StringVar(value=list(available_models.keys())[0] if available_models else '')
    training_files_var = tkinter.StringVar(value=())
    validation_files_var = tkinter.StringVar(value=())
    cheat_training_files_var = tkinter.StringVar(value=())
    cheat_validation_files_var = tkinter.StringVar(value=())
    testing_files_var = tkinter.StringVar(value=())
    live_graphing_var = tkinter.BooleanVar(value=False)
    save_dir_var = tkinter.StringVar(value='output/')

    # --- Widgets ---
    frame = tkinter.ttk.Frame(win, padding="10")
    frame.grid(row=0, column=0, sticky=(tkinter.W, tkinter.E, tkinter.N, tkinter.S))

    # Mode
    tkinter.ttk.Label(frame, text="Mode:").grid(column=0, row=0, sticky=tkinter.W)
    mode_menu = tkinter.ttk.OptionMenu(frame, variable=mode_var, default='collect', values=('collect', 'train', 'test', 'deploy'))
    mode_menu.grid(column=1, row=0, sticky=(tkinter.W, tkinter.E))
    mode_var.trace_add('write', on_mode_change)

    # Model
    tkinter.ttk.Label(frame, text="Model:").grid(column=0, row=1, sticky=tkinter.W)
    model_menu = tkinter.ttk.OptionMenu(frame, variable=model_var, default=model_var.get(), values=list(available_models.keys()))
    model_menu.config(*available_models.keys())
    model_menu.grid(column=1, row=1, sticky=(tkinter.W, tkinter.E))
    model_var.trace_add('write', on_model_change)

    # File Selections
    tkinter.ttk.Label(frame, text="Training Files:").grid(column=0, row=1, sticky=tkinter.W)
    train_files_button = tkinter.ttk.Button(frame, text="Browse", command=lambda: training_files_var.set(prompt_for_csv_files('Select Training Files'))).grid(column=1, row=1, sticky=tkinter.W)
    
    tkinter.ttk.Label(frame, text="Validation Files:").grid(column=0, row=2, sticky=tkinter.W)
    val_files_button = tkinter.ttk.Button(frame, text="Browse", command=lambda: validation_files_var.set(prompt_for_csv_files('Select Validation Files'))).grid(column=1, row=2, sticky=tkinter.W)

    tkinter.ttk.Label(frame, text="Cheat Training Files:").grid(column=0, row=3, sticky=tkinter.W)
    cheat_training_files_button = tkinter.ttk.Button(frame, text="Browse", command=lambda: cheat_training_files_var.set(prompt_for_csv_files('Select Cheat Training Files'))).grid(column=1, row=3, sticky=tkinter.W)

    tkinter.ttk.Label(frame, text="Cheat Validation Files:").grid(column=0, row=4, sticky=tkinter.W)
    cheat_validation_files_button = tkinter.ttk.Button(frame, text="Browse", command=lambda: cheat_validation_files_var.set(prompt_for_csv_files('Select Cheat Validation Files'))).grid(column=1, row=4, sticky=tkinter.W)

    tkinter.ttk.Label(frame, text="Testing Files:").grid(column=0, row=3, sticky=tkinter.W)
    test_files_button = tkinter.ttk.Button(frame, text="Browse", command=lambda: testing_files_var.set(prompt_for_csv_files('Select Testing Files'))).grid(column=1, row=3, sticky=tkinter.W)

    # Live Graphing
    tkinter.ttk.Label(frame, text="Live Graphing:").grid(column=0, row=2, sticky=tkinter.W)
    live_graphing_check = tkinter.ttk.Checkbutton(frame, variable=live_graphing_var)
    live_graphing_check.grid(column=1, row=2, sticky=tkinter.W)

    # Kill Key
    tkinter.ttk.Label(frame, text="Kill Bind:").grid(column=0, row=1, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=kill_bind_var).grid(column=1, row=1, sticky=(tkinter.W, tkinter.E))

    # Capture Bind
    tkinter.ttk.Label(frame, text="Capture Bind:").grid(column=0, row=2, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=capture_bind_var).grid(column=1, row=2, sticky=(tkinter.W, tkinter.E))

    # Whitelists (comma-separated)
    tkinter.ttk.Label(frame, text="Keyboard Whitelist (csv):").grid(column=0, row=3, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=keyboard_whitelist_var).grid(column=1, row=3, sticky=(tkinter.W, tkinter.E))

    tkinter.ttk.Label(frame, text="Mouse Whitelist (csv):").grid(column=0, row=4, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=mouse_whitelist_var).grid(column=1, row=4, sticky=(tkinter.W, tkinter.E))

    tkinter.ttk.Label(frame, text="Gamepad Whitelist (csv):").grid(column=0, row=5, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=gamepad_whitelist_var).grid(column=1, row=5, sticky=(tkinter.W, tkinter.E))

    # Numerical settings
    tkinter.ttk.Label(frame, text="Polling Rate (Hz):").grid(column=0, row=6, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=polling_rate_var).grid(column=1, row=6, sticky=(tkinter.W, tkinter.E))

    tkinter.ttk.Label(frame, text="Sequence Length:").grid(column=0, row=7, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=sequence_length_var).grid(column=1, row=7, sticky=(tkinter.W, tkinter.E))

    tkinter.ttk.Label(frame, text="Batch Size:").grid(column=0, row=8, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=batch_size_var).grid(column=1, row=8, sticky=(tkinter.W, tkinter.E))

    # Save Directory
    tkinter.ttk.Label(frame, text="Save Directory:").grid(column=0, row=9, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=save_dir_var).grid(column=1, row=9, sticky=(tkinter.W, tkinter.E))
    browse_button = tkinter.ttk.Button(frame, text="Browse", command=browse_save_dir)
    browse_button.grid(column=2, row=9, sticky=tkinter.W)

    # Submit Button
    submit_button = tkinter.ttk.Button(frame, text="Submit", command=on_submit)
    submit_button.grid(column=1, row=10, sticky=tkinter.E, pady=10)

    frame.columnconfigure(1, weight=1)

    on_mode_change()

    win.wait_window()

    return config if config else None