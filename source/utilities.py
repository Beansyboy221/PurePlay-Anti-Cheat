import sklearn.preprocessing
import pandas
import torch
import numpy
import constants, devices

# region Device Polling Utilities
def poll_if_capturing(config: dict) -> list:
    """Polls input devices if capture bind(s) are pressed."""
    capturing = True
    capture_binds = config.get('capture_bind_list', [])
    if len(capture_binds) > 1:
        pressed_capture_binds = [devices.is_pressed(bind) for bind in capture_binds]
        if config.get('capture_bind_logic') == 'ANY':
            capturing = any(pressed_capture_binds)
        else:
            capturing = all(pressed_capture_binds)
    elif not devices.is_pressed(capture_binds[0]):
        capturing = False
    if capturing:
        row = devices.poll_keyboard(config.get('keyboard_whitelist')) + devices.poll_mouse(config.get('mouse_whitelist')) + devices.poll_gamepad(config.get('gamepad_whitelist'))
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
    pressed_kill_binds = [devices.is_pressed(bind) for bind in kill_bind_list]
    if config.get('kill_bind_logic') == 'ANY':
        return any(pressed_kill_binds)
    else: # 'ALL'
        return all(pressed_kill_binds)
# endregion

# region Data Utilities
SCALER = sklearn.preprocessing.StandardScaler()

def fit_global_scaler(files: list[str], whitelist: list[str]) -> None:
    """Fits the global scaler on the given files."""
    scalable_columns = [col for col in constants.TWO_DIMENSIONAL_BINDS if col in whitelist]
    if not scalable_columns:
        return

    all_data = []
    for file in files:
        df = pandas.read_csv(file, usecols=lambda col: col in scalable_columns)
        all_data.append(df)

    if all_data:
        combined_df = pandas.concat(all_data, ignore_index=True)
        SCALER.fit(combined_df[scalable_columns])


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
        scalable_columns = [col for col in constants.TWO_DIMENSIONAL_BINDS if col in self.feature_columns]
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
# endregion

# region Hardware Compatibility
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
# endregion