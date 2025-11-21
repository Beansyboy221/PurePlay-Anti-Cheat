import lightning.pytorch.callbacks
import sklearn.preprocessing
import tkinter.filedialog
import optuna.integration
import matplotlib.pyplot
import torch.utils.data
import tkinter.ttk
import lightning
import threading
import keyboard
import collect
import logging
import tkinter
import optuna
import pandas
import models
import mouse
import numpy
import json
import time

# =============================================================================
# Global Variables
# =============================================================================
scaler = sklearn.preprocessing.MinMaxScaler()

# =============================================================================
# Dataset
# =============================================================================
class InputDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_length, whitelist, label=0):
        self.sequence_length = sequence_length
        self.label = label
        data_frame = pandas.read_csv(file_path)
        self.feature_columns = [col for col in whitelist if col in data_frame.columns]
        scale_columns = ['deltaX', 'deltaY', 'LX', 'LY', 'RX', 'RY']
        for column in scale_columns:
            if column in self.feature_columns:
                data_frame[column] = scaler.fit_transform(data_frame[column].values.reshape(-1, 1)).flatten()
        data_array = data_frame[self.feature_columns].values.astype(numpy.float32)
        remainder = len(data_array) % sequence_length
        if remainder != 0:
            data_array = data_array[:-remainder]
        self.data_tensor = torch.from_numpy(data_array)

    def __len__(self):
        return len(self.data_tensor) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        seq = self.data_tensor[start_idx : start_idx + self.sequence_length]
        return seq, torch.tensor(self.label, dtype=torch.float32)

# =============================================================================
# Training Process
# =============================================================================
def train_model(config: dict) -> None:
    training_type = config['training_type']
    sequence_length = config['sequence_length']
    batch_size = config['batch_size']
    keyboard_whitelist = config['keyboard_whitelist']
    mouse_whitelist = config['mouse_whitelist']
    gamepad_whitelist = config['gamepad_whitelist']
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
    kill_key = config['kill_key']

    # Preprocessing
    if not whitelist:
        return print('No input features selected. Exiting...')
    
    train_files = _prompt_for_files('Select non-cheat training files')
    if not train_files:
        print('No training files selected. Exiting...')
        return
    val_files = _prompt_for_files('Select non-cheat validation files')
    if not val_files:
        print('No validation files selected. Exiting...')
        return

    train_datasets = [InputDataset(file, sequence_length, whitelist) for file in train_files]
    val_datasets = [InputDataset(file, sequence_length, whitelist) for file in val_files]

    if training_type == 'supervised':
        cheat_train_files = _prompt_for_files('Select cheat training files')
        if not cheat_train_files:
            print('No files selected. Exiting...')
            return
        cheat_val_files = _prompt_for_files('Select cheat validation files')
        if not cheat_val_files:
            print('No files selected. Exiting...')
            return
        train_datasets += [InputDataset(file, sequence_length, whitelist, label=1) for file in cheat_train_files]
        val_datasets += [InputDataset(file, sequence_length, whitelist, label=1) for file in cheat_val_files]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    save_dir = tkinter.filedialog.askdirectory(title='Select model/graph save location')

    # Tuning and Training
    class KillKeyCallback:
        def __init__(self, stop_key='q'):
            self.stop_key = stop_key
            self.stop_flag = False
            keyboard.add_hotkey(self.stop_key, self.stop)

        def stop(self):
            print(f"Stopping study. Finishing current trial...")
            self.stop_flag = True

        def __call__(self, study: optuna.Study, trial: optuna.Trial):
            if self.stop_flag:
                study.stop()

    def objective(trial: optuna.Trial) -> float:
        hyperparams = {
            'hidden_dim' : trial.suggest_int('hidden_dim', 1, 256, log=True),
            'num_layers' : trial.suggest_int('num_layers', 1, 6),
            'learning_rate' : trial.suggest_float('learning_rate', 0.00001, 0.01, log=True),
            'dropout' : trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        }
        if hyperparams['num_layers'] == 1:
            hyperparams['dropout'] = 0.0
        
        if training_type == 'unsupervised':
            model = models.UnsupervisedModel(len(whitelist), hyperparams, sequence_length, save_dir, trial.number)
        else:
            model = models.SupervisedModel(len(whitelist), hyperparams, sequence_length, save_dir, trial.number)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-8,
            patience=10,
            mode='min'
        )
        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=save_dir,
            filename=f"trial_{trial.number}_{param_count}_{time.strftime('%Y%m%d-%H%M%S')}",
            save_top_k=1,
            mode='min'
        )
        prune_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_loss')
        trainer = lightning.Trainer(
            max_epochs=1024,
            precision='16-mixed',
            callbacks=[early_stop_callback, checkpoint_callback, prune_callback],
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        try:
            trainer.fit(model, train_loader, val_loader)
        finally:
            matplotlib.pyplot.close(model.figure)

        print(f'[Early Stopping Triggered!] Trial {trial.number} stopped at epoch {trainer.current_epoch}.')
        best_checkpoint = checkpoint_callback.best_model_path
        if best_checkpoint:
            trial.set_user_attr('best_checkpoint', best_checkpoint)
        
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            raise ValueError('Validation loss not found!')
        return val_loss.item()

    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    study = optuna.create_study(direction='minimize')
    study.optimize(
        objective,
        n_trials=2048,
        callbacks=[KillKeyCallback(kill_key)],
        gc_after_trial=True
    )

    importances = optuna.importance.get_param_importances(study)
    print('\nTuning Importances:')
    for key, value in importances.items():
        print(f'{key}:{value}')
    best_trial = study.best_trial
    print(f'\nBest Trial: {best_trial.number} Loss: {best_trial.values[0]}')

def _prompt_for_files(message: str) -> list[str]:
    files = tkinter.filedialog.askopenfilenames(
        title=message,
        filetypes=[('CSV Files', '*.csv')]
    )
    return files

# =============================================================================
# Static Analysis Process
# =============================================================================
def run_static_analysis(config: dict) -> None:
    training_type = config['training_type']
    sequence_length = config['sequence_length']
    keyboard_whitelist = config['keyboard_whitelist']
    mouse_whitelist = config['mouse_whitelist']
    gamepad_whitelist = config['gamepad_whitelist']
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
    
    file = tkinter.filedialog.askopenfilename(
        title='Select data file to analyze',
        filetypes=[('CSV Files', '*.csv')]
    )
    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )
    report_dir = tkinter.filedialog.askdirectory(title='Select report save location')

    # Analyze data
    if not file or not checkpoint:
        return print('Data or model not selected. Exiting.')
    
    if training_type == 'unsupervised':
        model = models.UnsupervisedModel.load_from_checkpoint(checkpoint)
    else:
        model = models.SupervisedModel.load_from_checkpoint(checkpoint)
    model.save_dir = report_dir

    test_dataset = InputDataset(file, sequence_length, whitelist)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    trainer = lightning.Trainer(
        logger=False,
        enable_checkpointing=False,
    )
    trainer.test(model, dataloaders=test_loader, ckpt_path=None)

# =============================================================================
# Live Analysis Mode
# =============================================================================
def run_live_analysis(config: dict) -> None:
    kill_key = config['kill_key']
    capture_bind = config['capture_bind']
    training_type = config['training_type']
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if training_type == 'unsupervised':
        model = models.UnsupervisedModel.load_from_checkpoint(checkpoint)
    else:
        model = models.SupervisedModel.load_from_checkpoint(checkpoint)
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
            match training_type:
                case 'unsupervised':
                    reconstruction = model(input_sequence)
                    reconstruction_error = model.loss_function(reconstruction, input_sequence)
                    print(f'Reconstruction Error: {reconstruction_error.item()}')
                case 'supervised':
                    logits = model(input_sequence)
                    confidence = torch.sigmoid(logits).mean()
                    print(f'Confidence: {confidence.item()}')
                case _:
                    print(f'Error: Invalid model type specified: {training_type}')
        time.sleep(1.0 / polling_rate)

# =============================================================================
# Main Function
# =============================================================================
def main():
    if torch.cuda.is_available():
        processor = torch.cuda.get_device_name(torch.cuda.current_device())
        if 'RTX' in processor or 'Tesla' in processor:
            torch.set_float32_matmul_precision('medium')
            print(f'Tensor Cores detected on device: "{processor}". Using medium precision for matmul.')

    matplotlib.use("agg")
    root = tkinter.Tk()
    root.withdraw()
    with open('config.json', 'r') as file_handle:
        config = json.load(file_handle)
    mode = config['mode']

    match mode:
        case 'collect':
            collect.collect_input_data(config)
        case 'train':
            train_model(config)
        case 'test':
            run_static_analysis(config)
        case 'deploy':
            run_live_analysis(config)
        case _:
            print(f'Error: Invalid mode specified in config file: {mode}')

    root.destroy()

if __name__ == '__main__':
    main()