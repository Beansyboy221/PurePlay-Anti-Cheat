import sklearn.preprocessing
import pandas
import torch
import numpy
import constants, devices

#region Device Polling Utilities
def poll_if_capturing(config: dict) -> list:
    """Polls input devices if capture bind(s) are pressed."""
    capturing = True
    capture_binds = config.capture_bind_list
    if len(capture_binds) > 1:
        pressed_capture_binds = [devices.is_pressed(bind) for bind in capture_binds]
        if config.capture_bind_logic == 'ANY':
            capturing = any(pressed_capture_binds)
        else:
            capturing = all(pressed_capture_binds)
    elif not devices.is_pressed(capture_binds[0]):
        capturing = False
    if capturing:
        row = devices.poll_keyboard(config.keyboard_whitelist) + devices.poll_mouse(config.mouse_whitelist) + devices.poll_gamepad(config.gamepad_whitelist)
        if config.ignore_empty_polls and not (row.count(0) == len(row)):
            return row
        elif not config.ignore_empty_polls:
            return row
    return None

def should_kill(config: dict) -> bool:
    """Determines whether the program should be terminated based on kill binds."""
    kill_bind_list = config.kill_bind_list
    if not kill_bind_list:
        return False
    pressed_kill_binds = [devices.is_pressed(bind) for bind in kill_bind_list]
    if config.kill_bind_logic == 'ANY':
        return any(pressed_kill_binds)
    else: # 'ALL'
        return all(pressed_kill_binds)
#endregion

#region Scaler and Scaler Fitting
SCALER = sklearn.preprocessing.StandardScaler() # Global scaler instance

def fit_global_scaler(file_paths: list[str], whitelist: list[str]) -> None:
    """Fits the global scaler on the given files."""
    scalable_columns = [col for col in constants.TWO_DIMENSIONAL_BINDS if col in whitelist]
    if not scalable_columns:
        print("No scalable columns found in the whitelist; skipping scaler fitting.")
        return

    all_data = []
    for file_path in file_paths:
        data_frame = pandas.read_hdf(file_path, columns=scalable_columns)
        all_data.append(data_frame)

    if all_data:
        combined_df = pandas.concat(all_data, ignore_index=True)
        SCALER.fit(combined_df[scalable_columns])

def get_scaler_state():
    """Returns the fitted parameters of the global SCALER."""
    if hasattr(SCALER, 'mean_'):
        return {
            'mean': SCALER.mean_,
            'var': SCALER.var_,
            'scale': SCALER.scale_,
            'n_samples_seen': SCALER.n_samples_seen_
        }
    return None
#endregion

#region Input Dataset
class InputDataset(torch.utils.data.Dataset):
    """Dataset for loading and filtering tensors from HDF5 files."""
    def __init__(self: object, file_path: str, config: dict, label: int = 0):
        self.polls_per_sequence = config.polls_per_sequence
        self.label = label
        whitelist = config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist

        data_frame = pandas.read_hdf(file_path)
        self.polling_rate = data_frame.attrs.get('polling_rate')
        self.feature_columns = [col for col in whitelist if col in data_frame.columns]
        
        # Scale 2D bind columns
        scalable_columns = [col for col in constants.TWO_DIMENSIONAL_BINDS if col in self.feature_columns]
        if scalable_columns:
            data_frame.loc[:, scalable_columns] = SCALER.transform(data_frame[scalable_columns].astype(numpy.float32))
        
        # Convert to numpy array of type float32
        data_array = data_frame[self.feature_columns].values.astype(numpy.float32)
        
        # Optionally filter out rows with all zeros
        if config.ignore_empty_polls:
            data_array = data_array[data_array.sum(axis=1) != 0] 

        # Trim to make divisible by polls_per_sequence
        excess_rows = len(data_array) % self.polls_per_sequence
        if excess_rows != 0:
            data_array = data_array[:-excess_rows]
        
        self.data_tensor = torch.from_numpy(data_array)

    def __len__(self):
        return len(self.data_tensor) // self.polls_per_sequence

    def __getitem__(self, idx):
        start_idx = idx * self.polls_per_sequence
        seq = self.data_tensor[start_idx : start_idx + self.polls_per_sequence]
        return seq, torch.tensor(self.label, dtype=torch.float32)
#endregion