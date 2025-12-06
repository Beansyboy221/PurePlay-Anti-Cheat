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
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
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
    
    whitelist = config['keyboard_whitelist'] + config['mouse_whitelist'] + config['gamepad_whitelist']
    sequence_length = model.hparams.sequence_length
    sequence_buffer = []
    
    print(f'Polling devices for live analysis (press {", ".join(config["kill_bind_list"])} to stop)...')
    while True:
        row = utilities.poll_if_capturing(config)
        if row:
            sequence_buffer.append(row)
        if len(sequence_buffer) == sequence_length:
            sequence = sequence_buffer[-sequence_length:]
            dataframe = pandas.DataFrame(sequence, columns=whitelist)
            
            scalable_columns = [column for column in utilities.SCALABLE_FEATURES if column in whitelist]
            if scalable_columns:
                dataframe.loc[:, scalable_columns] = utilities.SCALER.transform(dataframe[scalable_columns])

            input_sequence = torch.tensor(dataframe.values, dtype=torch.float32, device=device).unsqueeze(0)
            output = model(input_sequence)

            match (model.training_type): # Ensure models save their training type
                case 'supervised':
                    print(f'Confidence: {torch.sigmoid(output).mean().item()}')
                case 'unsupervised':
                    print(f'Reconstruction Error: {model.loss_function(output, input_sequence).item()}')
            sequence_buffer.pop(0)
        time.sleep(1.0 / config['polling_rate'])