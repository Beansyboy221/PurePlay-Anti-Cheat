import threading
import time
import csv
import utilities

# =============================================================================
# Data Collection Mode
# =============================================================================
def collect_input_data(config: dict) -> None:
    keyboard_whitelist = config['keyboard_whitelist']
    mouse_whitelist = config['mouse_whitelist']
    gamepad_whitelist = config['gamepad_whitelist']

    file_name = f"{config['save_dir']}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    if any(key in mouse_whitelist for key in ('deltaX', 'deltaY')):
        raw_input_thread = threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True)
        raw_input_thread.start()

    with open(file_name, mode='w', newline='') as file_handle:
        csv_writer = csv.writer(file_handle)
        header = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
        csv_writer.writerow(header)
        print(f'Polling devices for collection (press {config['kill_bind']} to stop)...')
        while True:
            if utilities.is_pressed(config['kill_bind']):
                break

            row = utilities.poll_all_devices(keyboard_whitelist, mouse_whitelist, gamepad_whitelist)

            if utilities.is_pressed(config['capture_bind']):
                if not row.count(0) == len(row): # Ignore empty inputs
                    csv_writer.writerow(row)
            time.sleep(1.0 / config['polling_rate'])
    print('Data collection stopped. Inputs saved.')
