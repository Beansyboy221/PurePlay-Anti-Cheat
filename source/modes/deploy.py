import collections
import threading
import torch
import numpy
import time
import utilities, devices

def run_live_analysis(config: object) -> None:
    model = config.model_class.load_from_checkpoint(config.model_file)
    model.to(devices.TORCH_DEVICE_TYPE)
    model.eval()

    whitelist = config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist
    polls_per_sequence = model.hparams.polls_per_sequence

    scalable_columns = [col for col in utilities.SCALABLE_FEATURES if col in whitelist]
    scalable_indices = [whitelist.index(col) for col in scalable_columns]

    buffer_lock = threading.Lock()
    sequence_buffer = collections.deque(maxlen=2*polls_per_sequence)
    kill_flag = threading.Event()

    @torch.no_grad()
    def analysis_worker():
        while not kill_flag.is_set():
            sequence = None

            with buffer_lock:
                if len(sequence_buffer) >= polls_per_sequence:
                    match config.deployment_window_type:
                        case 'tumbling':
                            sequence = list(sequence_buffer)[:polls_per_sequence]
                            sequence_buffer.clear()
                        case 'sliding':
                            sequence = list(sequence_buffer)[-polls_per_sequence:]
                            sequence_buffer.popleft()

            if sequence:
                seq_array = numpy.array(sequence, dtype=numpy.float32)
                if scalable_indices:
                    seq_array[:, scalable_indices] = utilities.SCALER.transform(seq_array[:, scalable_indices])
                input_sequence = torch.from_numpy(seq_array).to(devices.TORCH_DEVICE_TYPE).unsqueeze(0)
                output = model(input_sequence)
                if model.training_type == 'supervised':
                    confidence = torch.sigmoid(output).mean().item()
                    print(f'Confidence: {confidence:.4f}')
                else:
                    recon_error = model.loss_function(output, input_sequence).item()
                    print(f'Reconstruction Error: {recon_error:.6f}')

            time.sleep(0.001)

    if any(key in config.mouse_whitelist for key in ('deltaX', 'deltaY')):
        threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True).start()

    threading.Thread(target=analysis_worker, daemon=True).start()

    print(f'Polling devices for live analysis (press {", ".join(config.kill_bind_list)} to stop)...')
    poll_interval = 1.0 / config.polling_rate

    while True:
        if utilities.should_kill(config):
            print('Kill bind(s) detected. Stopping live analysis.')
            kill_flag.set()
            break

        row = utilities.poll_if_capturing(config)
        if row:
            with buffer_lock:
                sequence_buffer.append(row)

        time.sleep(poll_interval)