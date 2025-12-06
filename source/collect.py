import threading
import time
import csv
import utilities

# =============================================================================
# Data Collection Mode
# =============================================================================
def collect_input_data(config: dict) -> None:
    file_name = f"{config['save_dir']}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    if any(key in config['mouse_whitelist'] for key in ('deltaX', 'deltaY')):
        threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True).start()

    with open(file_name, mode='w', newline='') as file_handle:
        csv_writer = csv.writer(file_handle)
        header = config['keyboard_whitelist'] + config['mouse_whitelist'] + config['gamepad_whitelist']
        csv_writer.writerow(header)
        print(f'Polling devices for collection (press {", ".join(config["kill_bind_list"])} to stop)...')
        while True:
            row = utilities.poll_if_capturing(config)
            if row:
                csv_writer.writerow(row)
            time.sleep(1.0 / config['polling_rate'])