import threading
import pyarrow
import queue
import time
import utilities

def collect_input_data(config: object) -> None:
    """Collects input data and saves it to a Parquet file."""
    file_name = f"{config.save_dir}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.parquet"
    header = config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist
    fields = [(name, pyarrow.float32()) for name in header]
    fields.append(('polling_rate', pyarrow.int32()))
    schema = pyarrow.schema(fields)
    
    # Parquet Writer Thread
    data_queue = queue.Queue()
    kill_event = threading.Event()
    writer_thread = threading.Thread(
        target=utilities.parquet_writer_worker,
        args=(file_name, schema, data_queue, kill_event),
        daemon=True
    )
    writer_thread.start()
    
    if any(key in config.mouse_whitelist for key in ('deltaX', 'deltaY')):
        threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True).start()
    
    poll_interval = 1.0 / config.polling_rate
    print(f'Polling at {config.polling_rate}Hz (press {", ".join(config.kill_bind_list)} to stop)...')

    try:
        while True:
            if utilities.should_kill(config):
                print("Kill bind(s) detected. Stopping...")
                break
            
            row = utilities.poll_if_capturing(config)
            if row:
                data_queue.put(row)
            
            time.sleep(poll_interval)
    finally:
        kill_event.set()
        writer_thread.join()
        print(f"Data saved to {file_name}")
