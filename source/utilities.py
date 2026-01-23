import pyarrow.parquet
import threading
import pyarrow
import queue
import time
import devices

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