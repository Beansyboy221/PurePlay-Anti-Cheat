import sklearn.preprocessing
import tkinter.filedialog
import pandas
import torch
import numpy

scaler = sklearn.preprocessing.MinMaxScaler()

def _prompt_for_csv_files(message: str) -> list[str]:
    files = tkinter.filedialog.askopenfilenames(
        title=message,
        filetypes=[('CSV Files', '*.csv')]
    )
    return files

class InputDataset(torch.utils.data.Dataset):
    def __init__(self: object, file_path: str, sequence_length: int, whitelist: list[str], label: int = 0):
        self.sequence_length = sequence_length
        self.label = label
        
        data_frame = self._load_data(file_path)
        self.feature_columns = self._get_feature_columns(data_frame, whitelist)
        data_frame = self._scale_features(data_frame)
        data_array = self._to_numpy_array(data_frame)
        data_array = self._trim_to_sequence_length(data_array)
        self.data_tensor = torch.from_numpy(data_array)

    def _load_data(self, file_path: str) -> pandas.DataFrame:
        """Loads data from a CSV file into a pandas DataFrame."""
        return pandas.read_csv(file_path)

    def _get_feature_columns(self, data_frame: pandas.DataFrame, whitelist: list[str]) -> list[str]:
        """Filters DataFrame columns based on a whitelist."""
        return [col for col in whitelist if col in data_frame.columns]

    def _scale_features(self, data_frame: pandas.DataFrame) -> pandas.DataFrame:
        """Scales specified columns of the DataFrame."""
        scale_columns = ['deltaX', 'deltaY', 'LX', 'LY', 'RX', 'RY']
        for column in scale_columns:
            if column in self.feature_columns:
                data_frame[column] = scaler.fit_transform(data_frame[column].values.reshape(-1, 1)).flatten()
        return data_frame

    def _to_numpy_array(self, data_frame: pandas.DataFrame) -> numpy.ndarray:
        """Converts the DataFrame to a NumPy array of float32.""";
        return data_frame[self.feature_columns].values.astype(numpy.float32)

    def _trim_to_sequence_length(self, data_array: numpy.ndarray) -> numpy.ndarray:
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