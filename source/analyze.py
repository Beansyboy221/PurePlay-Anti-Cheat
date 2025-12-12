import threading
import lightning
import torch
import time
import collections
import utilities
import pandas

# =============================================================================
# Static Analysis Mode
# =============================================================================
def run_static_analysis(config: dict) -> None:
    """Performs static analysis on selected data files using a pre-trained model."""
    whitelist = config.get('keyboard_whitelist') + config.get('mouse_whitelist') + config.get('gamepad_whitelist')

    model = config.get('model_class').load_from_checkpoint(config.get('model_file'))
    config['polls_per_sequence'] = model.hparams.polls_per_sequence
    model.save_dir = config.get('save_dir')

    utilities.fit_global_scaler(config.get('testing_files'), whitelist)
    for file in config.get('testing_files'):
        test_dataset = utilities.InputDataset(file, config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
        trainer.test(model, dataloaders=test_loader, ckpt_path=None)

# =============================================================================
# Live Analysis Mode
# =============================================================================
def run_live_analysis(config: dict) -> None:
    """Performs live analysis using a pre-trained model without blocking polling."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config.get('model_class').load_from_checkpoint(config.get('model_file'))
    model.to(device)
    model.eval()

    if any(key in config.get('mouse_whitelist') for key in ('deltaX', 'deltaY')):
        threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True).start()

    whitelist = config.get('keyboard_whitelist') + config.get('mouse_whitelist') + config.get('gamepad_whitelist')
    polls_per_sequence = model.hparams.polls_per_sequence

    buffer_lock = threading.Lock()
    sequence_buffer = collections.deque(maxlen=2*polls_per_sequence) # Chose an arbitrary size larger than polls_per_sequence
    kill_flag = threading.Event()

    def analysis_worker():
        while not kill_flag.is_set():
            with buffer_lock:
                if len(sequence_buffer) < polls_per_sequence:
                    pass
                else:
                    match config.get('deployment_window_type'):
                        case 'tumbling':
                            sequence = list(sequence_buffer)[:polls_per_sequence]
                            sequence_buffer.clear()
                        case 'sliding':
                            sequence = list(sequence_buffer)[-polls_per_sequence:]
                            sequence_buffer.popleft()
                        case _:
                            print(f'Invalid deployment window type: {config.get("deployment_window_type")}')
                            kill_flag.set()
                            return

            if len(sequence) == polls_per_sequence:
                dataframe = pandas.DataFrame(sequence, columns=whitelist)
                scalable_columns = [column for column in utilities.SCALABLE_FEATURES if column in whitelist]
                if scalable_columns:
                    dataframe.loc[:, scalable_columns] = utilities.SCALER.transform(dataframe[scalable_columns])
                input_sequence = torch.tensor(dataframe.values, dtype=torch.float32, device=device).unsqueeze(0)
                
                output = model(input_sequence)
                match (model.training_type):
                    case 'supervised':
                        print(f'Confidence: {torch.sigmoid(output).mean().item()}')
                    case 'unsupervised':
                        print(f'Reconstruction Error: {model.loss_function(output, input_sequence).item()}')

            time.sleep(0.001)  # Sleep so worker doesn't consume full CPU
    threading.Thread(target=analysis_worker, daemon=True).start()

    print(f'Polling devices for live analysis (press {", ".join(config.get("kill_bind_list"))} to stop)...')
    poll_interval = 1.0 / config.get("polling_rate")

    while True:
        if utilities.should_kill(config):
            print('Kill bind(s) detected. Stopping live analysis.')
            kill_flag.set()
            break

        row = utilities.poll_if_capturing(config)
        if row:
            with buffer_lock: # Prevents race condition with analysis worker
                sequence_buffer.append(row)

        time.sleep(poll_interval)