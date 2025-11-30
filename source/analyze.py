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
    
    files = utilities.prompt_for_csv_files('Select data files for analysis')
    if not files:
        return print('No data files selected. Exiting.')
    
    checkpoint = utilities.prompt_for_checkpoint_file()
    if not checkpoint:
        return print('No model checkpoint selected. Exiting.')

    utilities.fit_global_scaler(files, whitelist)

    model = config['model_class'].load_from_checkpoint(checkpoint)
    model.save_dir = config['save_dir']

    for file in files:
        test_dataset = utilities.InputDataset(file, config['sequence_length'], whitelist)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=False)

        trainer = lightning.Trainer(
            logger=False,
            enable_checkpointing=False,
        )
        trainer.test(model, dataloaders=test_loader, ckpt_path=None)

# =============================================================================
# Live Analysis Mode
# =============================================================================
def run_live_analysis(config: dict) -> None:
    """Performs live analysis using a pre-trained model."""
    keyboard_whitelist = config['keyboard_whitelist']
    mouse_whitelist = config['mouse_whitelist']
    gamepad_whitelist = config['gamepad_whitelist']

    checkpoint = utilities.prompt_for_checkpoint_file()
    if not checkpoint:
        return print('No model checkpoint selected. Exiting.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = config['model_class'].load_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    if any(key in mouse_whitelist for key in ('deltaX', 'deltaY')):
        raw_input_thread = threading.Thread(target=utilities.listen_for_mouse_movement, daemon=True)
        raw_input_thread.start()
    sequence = []

    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
    
    print(f'Polling devices for live analysis (press {config['kill_bind']} to stop)...')
    while True:
        if utilities.is_pressed(config['kill_bind']):
            break

        if utilities.is_pressed(config['capture_bind']):
            row = utilities.poll_all_devices(keyboard_whitelist, mouse_whitelist, gamepad_whitelist)
            if not (row.count(0) == len(row)): # Ignore empty inputs
                sequence.append(row)

        if len(sequence) == config['sequence_length']:
            current_sequence = sequence[-config['sequence_length']:]
            dataframe = pandas.DataFrame(current_sequence, columns=whitelist)
            
            columns_to_scale = [col for col in utilities.SCALE_COLUMNS if col in whitelist]
            if columns_to_scale:
                dataframe.loc[:, columns_to_scale] = utilities.SCALER.transform(dataframe[columns_to_scale])

            input_sequence = torch.tensor(dataframe.values, dtype=torch.float32, device=device).unsqueeze(0)
            output = model(input_sequence)

            match (model.training_type): # Model may not have training_type saved
                case 'supervised':
                    confidence = torch.sigmoid(output).mean()
                    print(f'Confidence: {confidence.item()}')
                case 'unsupervised':
                    reconstruction_error = model.loss_function(output, input_sequence)
                    print(f'Reconstruction Error: {reconstruction_error.item()}')
            sequence.pop(0)
        time.sleep(1.0 / config['polling_rate'])