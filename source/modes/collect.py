import threading
import time
import csv
import utilities

def collect_input_data(config: dict) -> None:
    file_name = f"{config.get('save_dir')}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    if any(key in config.get('mouse_whitelist') for key in ('deltaX', 'deltaY')):
        threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True).start()
    
    with open(file_name, mode='w', newline='') as file_handle:
        csv_writer = csv.writer(file_handle)
        header = config.get('keyboard_whitelist') + config.get('mouse_whitelist') + config.get('gamepad_whitelist')
        csv_writer.writerow(header)
        print(f'Polling devices for collection (press {", ".join(config.get("kill_bind_list"))} to stop)...')

        poll_interval = 1.0 / config.get('polling_rate')
        while True:
            if utilities.should_kill(config):
                print("Kill bind(s) detected. Stopping data collection.")
                break

            row = utilities.poll_if_capturing(config)
            if row:
                csv_writer.writerow(row)
                
            time.sleep(poll_interval)