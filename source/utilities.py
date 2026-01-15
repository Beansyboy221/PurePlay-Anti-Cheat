import pyarrow.parquet
import threading
import pyarrow
import polars
import queue
import torch
import numpy
import time
import constants, devices, scaler

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

class InputDataset(torch.utils.data.Dataset):
    """Dataset for input sequences stored in Parquet files."""
    def __init__(self, file_path: str, polls_per_sequence: int, whitelist: list[str], ignore_empty_polls: bool = True, label: int = 0):
        self.polls_per_sequence = polls_per_sequence
        self.label = label

        lazy_frame = polars.scan_parquet(file_path)
        self.polling_rate = lazy_frame.schema.metadata.get(b'polling_rate').decode('utf-8')

        # Verify all requested whitelist columns exist in the parquet file.
        missing_columns = [column for column in whitelist if column not in lazy_frame.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in {file_path}: {', '.join(missing_columns)}")
        
        feature_columns = [column for column in whitelist if column in lazy_frame.columns]
        
        # Remove empty polls if configured
        if ignore_empty_polls:
            lazy_frame = lazy_frame.filter(polars.sum_horizontal(feature_columns) != 0)
        
        data_frame = lazy_frame.collect()

        # Apply scaling if the global SCALER is fitted
        scalable_columns = [column for column in constants.TWO_DIMENSIONAL_BINDS if column in feature_columns]
        if scalable_columns:
            raw_values = data_frame.select(scalable_columns).to_numpy().astype(numpy.float32)
            scaled_values = scaler.SCALER.transform(raw_values)
            
            data_frame = data_frame.with_columns([
                polars.Series(name=column, values=scaled_values[:, index]) 
                for index, column in enumerate(scalable_columns)
            ])

        # Convert to array for slicing
        data_array = data_frame.select(feature_columns).to_numpy(writable=True).astype(numpy.float32)
        excess_rows = len(data_array) % self.polls_per_sequence
        if excess_rows != 0:
            data_array = data_array[:-excess_rows]
        
        # Convert to tensor
        self.data_tensor = torch.from_numpy(data_array)

    def __len__(self):
        return len(self.data_tensor) // self.polls_per_sequence

    def __getitem__(self, index):
        start_index = index * self.polls_per_sequence
        sequence = self.data_tensor[start_index : start_index + self.polls_per_sequence]
        return sequence, torch.tensor(self.label, dtype=torch.float32)
    
def parquet_writer_worker(file_name: str, schema: pyarrow.schema, data_queue: queue, kill_event: threading.Event):
    """Worker function to write polled data to a Parquet file."""
    with pyarrow.parquet.ParquetWriter(file_name, schema) as writer:
        while not kill_event.is_set():
            batch = []
            while not data_queue.empty():
                try:
                    batch.append(data_queue.get_nowait())
                except queue.Empty:
                    break
            
            if batch:
                arrays = [pyarrow.array(column, type=pyarrow.float32()) for column in zip(*batch)]
                table = pyarrow.Table.from_arrays(arrays, schema=schema)
                writer.write_table(table)
            
            if not kill_event.is_set():
                time.sleep(1.0)