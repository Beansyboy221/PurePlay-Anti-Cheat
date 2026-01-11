import threading
import time
import h5py
import utilities

def collect_input_data(config: dict) -> None:
    file_name = f"{config.get('save_dir')}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.h5"
    header = config.get('keyboard_whitelist') + config.get('mouse_whitelist') + config.get('gamepad_whitelist')
    polling_rate = config.get('polling_rate')
    poll_interval = 1.0 / polling_rate
    
    if any(key in config.get('mouse_whitelist') for key in ('deltaX', 'deltaY')):
        threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True).start()
    
    with h5py.File(file_name, 'w') as h5_file:
        h5_file.attrs['polling_rate'] = polling_rate
        h5_file.attrs['column_names'] = header
        dataset = h5_file.create_dataset(
            'input_data', 
            shape=(0, len(header)), 
            maxshape=(None, len(header)), 
            dtype='float32', 
            compression="gzip"
        )
        
        print(f'Polling devices for collection (press {", ".join(config.get("kill_bind_list"))} to stop)...')
        while True:
            if utilities.should_kill(config):
                print("Kill bind(s) detected. Stopping data collection.")
                break
            row = utilities.poll_if_capturing(config)
            if row:
                dataset.resize((dataset.shape[0] + 1, dataset.shape[1]))
                dataset[-1, :] = row
            time.sleep(poll_interval)