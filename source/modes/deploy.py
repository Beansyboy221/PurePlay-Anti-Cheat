import collections
import threading
import pyarrow
import queue
import torch
import numpy
import time
import utilities, constants, devices, scaler

def run_live_analysis(config: object) -> None:
    """Performs live analysis on input data using a pre-trained model."""
    model = config.model_class.load_from_checkpoint(config.model_file)
    model.to(devices.TORCH_DEVICE_TYPE)
    model.eval()
    
    kill_event = threading.Event()
    
    # Analysis Thread
    buffer_lock = threading.Lock()
    sequence_buffer = collections.deque(maxlen=2*model.hparams.polls_per_sequence)
    scalable_columns = [col for col in utilities.SCALABLE_FEATURES if col in model.hparams.whitelist]
    scalable_indices = [model.hparams.whitelist.index(col) for col in scalable_columns]
    analysis_thread = threading.Thread(
        target=analysis_worker,
        args=(model, config.deployment_window_type, kill_event, buffer_lock, sequence_buffer, scalable_indices),
        daemon=True
    )
    analysis_thread.start()

    # Parquet Writer Thread
    if config.write_to_file:
        file_name = f"{config.save_dir}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.parquet"
        fields = [(name, pyarrow.float32()) for name in model.hparams.whitelist]
        fields.append(('polling_rate', pyarrow.int32()))
        schema = pyarrow.schema(fields)
        
        data_queue = queue.Queue()
        writer_thread = threading.Thread(
            target=utilities.parquet_writer_worker,
            args=(file_name, schema, data_queue, kill_event),
            daemon=True
        )
        writer_thread.start()

    # Mouse Listener Thread
    if any(key in config.mouse_whitelist for key in ('deltaX', 'deltaY')):
        mouse_listener_thread = threading.Thread(
            target=utilities.listen_for_mouse_movement,
            args=(kill_event),
            daemon=True
        )
        mouse_listener_thread.start()

    poll_interval = 1.0 / config.polling_rate
    print(f'Polling at {config.polling_rate}Hz (press {", ".join(config.kill_bind_list)} to stop)...')

    try:
        while True:
            if utilities.should_kill(config):
                print('Kill bind(s) detected. Stopping live analysis.')
                break

            row = utilities.poll_if_capturing(config)
            if row:
                if config.write_to_file:
                    data_queue.put(row)
                with buffer_lock:
                    sequence_buffer.append(row)

            time.sleep(poll_interval)
    finally:
        kill_event.set()
        analysis_thread.join()
        if mouse_listener_thread:
            mouse_listener_thread.join()
        if config.write_to_file:
            writer_thread.join()
            print(f"Data saved to {file_name}")

@torch.no_grad()
def analysis_worker(model, 
                    window_type: constants.WindowType, 
                    kill_event: threading.Event, 
                    buffer_lock: threading.Lock, 
                    sequence_buffer: collections.deque, 
                    scalable_indices: list[int]):
    """Worker function to perform live analysis on input sequences."""
    
    while not kill_event.is_set():
        sequence = None
        with buffer_lock:
            if len(sequence_buffer) >= model.hparams.polls_per_sequence:
                if window_type == constants.WindowType.TUMBLING:
                    sequence = list(sequence_buffer)[:model.hparams.polls_per_sequence]
                    sequence_buffer.clear()
                else: # sliding
                    sequence = list(sequence_buffer)[-model.hparams.polls_per_sequence:]
                    sequence_buffer.popleft()

        if sequence:
            seq_array = numpy.array(sequence, dtype=numpy.float32)
            if scalable_indices:
                seq_array[:, scalable_indices] = scaler.SCALER.transform(seq_array[:, scalable_indices])
            
            input_seq = torch.from_numpy(seq_array).to(devices.TORCH_DEVICE_TYPE).unsqueeze(0)
            output = model(input_seq)
            
            if model.training_type == constants.TrainingType.SUPERVISED:
                print(f'Confidence: {torch.sigmoid(output).mean().item():.4f}')
            else:
                print(f'Error: {model.loss_function(output, input_seq).item():.6f}')

        time.sleep(0.001)