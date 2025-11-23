import tkinter.filedialog
import threading
import keyboard
import lightning
import torch
import models
import utilities
import collect
import tkinter
import numpy
import sklearn.preprocessing
import mouse
import time

scaler = sklearn.preprocessing.MinMaxScaler()

# =============================================================================
# Static Analysis Process
# =============================================================================
def run_static_analysis(config: dict) -> None:
    model_class = config['model_class']
    keyboard_whitelist = config['keyboard_whitelist']
    mouse_whitelist = config['mouse_whitelist']
    gamepad_whitelist = config['gamepad_whitelist']
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
    
    files = utilities._prompt_for_csv_files('Select data files for analysis')
    if not files:
        print('No data files selected. Exiting.')
        return
    
    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )
    if not checkpoint:
        print('No model checkpoint selected. Exiting.')
        return

    report_dir = tkinter.filedialog.askdirectory(title='Select report save location')
    
    model = model_class.load_from_checkpoint(checkpoint)
    model.save_dir = report_dir

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
    model_class = config['model_class']
    kill_key = config['kill_key']
    capture_bind = config['capture_bind']
    polling_rate = config['polling_rate']
    sequence_length = config['sequence_length']
    keyboard_whitelist = config['keyboard_whitelist']
    mouse_whitelist = config['mouse_whitelist']
    gamepad_whitelist = config['gamepad_whitelist']
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist

    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )
    if not checkpoint:
        print('No model checkpoint selected. Exiting.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class.load_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    if any(key in mouse_whitelist for key in ('deltaX', 'deltaY', 'angle')):
        raw_input_thread = threading.Thread(target=collect.listen_for_mouse_movement, daemon=True)
        raw_input_thread.start()
    sequence = []

    scale_columns = ['deltaX', 'deltaY', 'LX', 'LY', 'RX', 'RY']
    
    print(f'Polling devices for live analysis (press {kill_key} to stop)...')
    while True:
        if keyboard.is_pressed(kill_key):
            break

        should_capture = True
        if capture_bind:
            should_capture = False
            try:
                if mouse.is_pressed(capture_bind):
                    should_capture = True
            except:
                pass
            try:
                if keyboard.is_pressed(capture_bind):
                    should_capture = True
            except:
                pass

        if should_capture:
            kb_row = collect.poll_keyboard(keyboard_whitelist)
            m_row = collect.poll_mouse(mouse_whitelist)
            gp_row = collect.poll_gamepad(gamepad_whitelist)
            row = kb_row + m_row + gp_row
            if not (row.count(0) == len(row)):
                sequence.append(row)

        if len(sequence) >= sequence_length:
            for i, column in enumerate(whitelist):
                if column in scale_columns:
                    row[i] = scaler.fit_transform(numpy.array(row[i]).reshape(-1, 1)).flatten()
            input_sequence = torch.tensor([sequence[-sequence_length:]], dtype=torch.float32, device=device)
            
            output = model(input_sequence)

            if isinstance(model, models.LSTMBinaryClassifier):
                confidence = torch.sigmoid(output).mean()
                print(f'Confidence: {confidence.item()}')
            elif isinstance(model, models.LSTMAutoencoder):
                reconstruction_error = model.loss_function(output, input_sequence)
                print(f'Reconstruction Error: {reconstruction_error.item()}')
        time.sleep(1.0 / polling_rate)