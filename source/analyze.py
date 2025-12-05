import threading
import lightning
import torch
import time
import utilities
import pandas

# =============================================================================
# Static Analysis Mode
# =============================================================================
def run_static_analysis(config: dict) -> None:
    """Performs static analysis on selected data files using a pre-trained model."""
    whitelist = config['keyboard_whitelist'] + config['mouse_whitelist'] + config['gamepad_whitelist']

    model = config['model_class'].load_from_checkpoint(config['model_file'])
    sequence_length = model.hparams.sequence_length
    model.save_dir = config['save_dir']

    utilities.fit_global_scaler(config['testing_files'], whitelist)
    for file in config['testing_files']:
        test_dataset = utilities.InputDataset(file, sequence_length, whitelist)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=False)
        trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
        trainer.test(model, dataloaders=test_loader, ckpt_path=None)

# =============================================================================
# Live Analysis Mode
# =============================================================================
def run_live_analysis(config: dict) -> None:
    """Performs live analysis using a pre-trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config['model_class'].load_from_checkpoint(config['model_file'])
    model.to(device)
    model.eval()

    if any(key in config['mouse_whitelist'] for key in ('deltaX', 'deltaY')):
        threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True).start()
    sequence = []

    whitelist = config['keyboard_whitelist'] + config['mouse_whitelist'] + config['gamepad_whitelist']
    sequence_length = model.hparams.sequence_length
    
    print(f'Polling devices for live analysis (press {", ".join(config["kill_bind_list"])} to stop)...')
    while True:
        row = utilities.poll_all_devices(config['keyboard_whitelist'], config['mouse_whitelist'], config['gamepad_whitelist'])
        should_write = False
        if config['capture_bind_list']:
            pressed_capture_binds = [utilities.is_pressed(bind) for bind in config['capture_bind_list']]
            if config['capture_bind_logic'] == 'ANY':
                should_write = any(pressed_capture_binds)
            else:
                should_write = all(pressed_capture_binds)
        else:
            should_write = not (config['ignore_empty_polls'] and utilities.row_is_empty(row))
        if should_write:
            sequence.append(row)

        if len(sequence) == sequence_length:
            current_sequence = sequence[-sequence_length:]
            dataframe = pandas.DataFrame(current_sequence, columns=whitelist)
            
            scalable_columns = [col for col in utilities.SCALABLE_FEATURES if col in whitelist]
            if scalable_columns:
                dataframe.loc[:, scalable_columns] = utilities.SCALER.transform(dataframe[scalable_columns])

            input_sequence = torch.tensor(dataframe.values, dtype=torch.float32, device=device).unsqueeze(0)
            output = model(input_sequence)

            match (model.training_type): # Ensure models save their training type
                case 'supervised':
                    confidence = torch.sigmoid(output).mean()
                    print(f'Confidence: {confidence.item()}')
                case 'unsupervised':
                    reconstruction_error = model.loss_function(output, input_sequence)
                    print(f'Reconstruction Error: {reconstruction_error.item()}')
            sequence.pop(0)
        time.sleep(1.0 / config['polling_rate'])