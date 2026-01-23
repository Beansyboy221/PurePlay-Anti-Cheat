import threading
import pyarrow
import queue
import time
import utilities, devices

def collect_input_data(config: object) -> None:
    """Collects input data and saves it to a Parquet file."""
    file_name = f"{config.save_dir}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.parquet"
    whitelist = config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist
    header_fields = [(name, pyarrow.float32()) for name in whitelist]
    metadata = {b'polling_rate': str(config.polling_rate).encode('utf-8')}
    schema = pyarrow.schema(header_fields, metadata=metadata)

    data_queue = queue.Queue()
    kill_event = threading.Event()
    writer_thread = threading.Thread(
        target=utilities.parquet_writer_worker,
        args=(file_name, schema, data_queue, kill_event),
        daemon=True
    )
    writer_thread.start()
    
    if any(key in config.mouse_whitelist for key in ('deltaX', 'deltaY')):
        threading.Thread(target=devices.listen_for_mouse_movement, args=(kill_event,), daemon=True).start()
    
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
