import lightning.pytorch.callbacks
import sklearn.preprocessing
import tkinter.filedialog
import optuna.integration
import matplotlib.pyplot
import torch.utils.data
import ctypes.wintypes
import lightning
import threading
import win32gui
import torch.nn
import keyboard
import logging
import tkinter
import optuna
import XInput
import pandas
import ctypes
import mouse
import numpy
import json
import time
import math
import csv

# =============================================================================
# Global Variables
# =============================================================================
mouse_deltas = [0, 0]  # [delta_x, delta_y]
mouse_lock = threading.Lock()
user32_library = ctypes.windll.user32
scaler = sklearn.preprocessing.MinMaxScaler()

# =============================================================================
# ctypes Structures for Raw Input
# =============================================================================
class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", ctypes.wintypes.DWORD),
        ("dwSize", ctypes.wintypes.DWORD),
        ("hDevice", ctypes.wintypes.HANDLE),
        ("wParam", ctypes.wintypes.WPARAM)
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", ctypes.wintypes.USHORT),
        ("ulButtons", ctypes.wintypes.ULONG),
        ("ulRawButtons", ctypes.wintypes.ULONG),
        ("lLastX", ctypes.c_long),
        ("lLastY", ctypes.c_long),
        ("ulExtraInformation", ctypes.wintypes.ULONG)
    ]

class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ("header", RAWINPUTHEADER),
        ("mouse",  RAWMOUSE)
    ]

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", ctypes.wintypes.USHORT),
        ("usUsage", ctypes.wintypes.USHORT),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("hwndTarget", ctypes.wintypes.HWND)
    ]

# =============================================================================
# Raw Input Listener (mouse only)
# =============================================================================
def raw_input_window_procedure(window_handle, message, input_code, data_handle):
    if message == 0x00FF:
        buffer_size = ctypes.wintypes.UINT(0)
        if not(user32_library.GetRawInputData(data_handle, 0x10000003, None, ctypes.byref(buffer_size), ctypes.sizeof(RAWINPUTHEADER)) == 0):
            return 0
        buffer = ctypes.create_string_buffer(buffer_size.value)
        if user32_library.GetRawInputData(data_handle, 0x10000003, buffer, ctypes.byref(buffer_size), ctypes.sizeof(RAWINPUTHEADER)) == buffer_size.value:
            raw_input_data = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
            if raw_input_data.header.dwType == 0:
                delta_x = raw_input_data.mouse.lLastX
                delta_y = raw_input_data.mouse.lLastY
                with mouse_lock:
                    mouse_deltas[0] += delta_x
                    mouse_deltas[1] += delta_y
    return win32gui.DefWindowProc(window_handle, message, input_code, data_handle)

def listen_for_mouse_movement():
    instance_handle = win32gui.GetModuleHandle(None)
    class_name = "RawInputWindow"
    window = win32gui.WNDCLASS()
    window.hInstance = instance_handle
    window.lpszClassName = class_name
    window.lpfnWndProc = raw_input_window_procedure
    win32gui.RegisterClass(window)
    window_handle = win32gui.CreateWindow(class_name, "Raw Input Hidden Window", 0, 0, 0, 0, 0, 0, 0, instance_handle, None)
    
    device = RAWINPUTDEVICE()
    device.usUsagePage = 0x01   # Generic Desktop Controls
    device.usUsage = 0x02       # Mouse
    device.dwFlags = 0x00000100 # RIDEV_INPUTSINK: receive input even when unfocused
    device.hwndTarget = window_handle
    if not user32_library.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
         raise ctypes.WinError()

    while True:
         win32gui.PumpWaitingMessages()
         time.sleep(0.001)

# =============================================================================
# Helper Function to Poll Keyboard
# =============================================================================
def poll_keyboard(keyboard_whitelist):
    row = [1 if keyboard.is_pressed(key) else 0 for key in keyboard_whitelist]
    return row

# =============================================================================
# Helper Function to Poll Mouse
# =============================================================================
def poll_mouse(mouse_whitelist, dpi):
    row = []
    for button in mouse_whitelist:
        if button in ['left', 'right', 'middle', 'x', 'x2']:
            row.append(1 if mouse.is_pressed(button) else 0)
    if any(key in mouse_whitelist for key in ('deltaX', 'deltaY')):
        with mouse_lock:
            if 'deltaX' in mouse_whitelist:
                inches = mouse_deltas[0] / dpi
                row.append(inches)
            if 'deltaY' in mouse_whitelist:
                inches = mouse_deltas[1] / dpi
                row.append(inches)
            mouse_deltas[0] = 0
            mouse_deltas[1] = 0
    return row

# =============================================================================
# Helper Function to Poll Gamepad
# =============================================================================
def poll_gamepad(gamepad_whitelist):
    row = []
    if XInput.get_connected()[0]:
        gamepad_state = XInput.get_state(0)
        button_values = XInput.get_button_values(gamepad_state)
        for feature in gamepad_whitelist:
            if feature in button_values:
                row.append(1 if button_values[feature] else 0)
            else:
                if feature == 'LT':
                    trigger_values = XInput.get_trigger_values(gamepad_state)
                    row.append(trigger_values[0])
                elif feature == 'RT':
                    trigger_values = XInput.get_trigger_values(gamepad_state)
                    row.append(trigger_values[1])
                elif feature in ['LX', 'LY', 'RX', 'RY']:
                    left_thumb, right_thumb = XInput.get_thumb_values(gamepad_state)
                    if feature == 'LX':
                        row.append(left_thumb[0])
                    elif feature == 'LY':
                        row.append(left_thumb[1])
                    elif feature == 'RX':
                        row.append(right_thumb[0])
                    elif feature == 'RY':
                        row.append(right_thumb[1])
                else:
                    row.append(0)
    else:
        row = [0] * len(gamepad_whitelist)
    return row

# =============================================================================
# Collection Mode
# =============================================================================
def collect_input_data(configuration):
    kill_key = configuration['kill_key']
    capture_bind = configuration['capture_bind']
    polling_rate = configuration['polling_rate']
    keyboard_whitelist = configuration['keyboard_whitelist']
    mouse_whitelist = configuration['mouse_whitelist']
    mouse_dpi = configuration['mouse_dpi']
    gamepad_whitelist = configuration['gamepad_whitelist']

    save_directory = tkinter.filedialog.askdirectory(title='Select data save folder')
    file_name = f"{save_directory}/inputs_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    if any(key in mouse_whitelist for key in ('deltaX', 'deltaY')):
        raw_input_thread = threading.Thread(target=listen_for_mouse_movement, daemon=True)
        raw_input_thread.start()
    
    with open(file_name, mode='w', newline='') as file_handle:
        csv_writer = csv.writer(file_handle)
        header = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
        csv_writer.writerow(header)
        print(f'Polling devices for collection (press {kill_key} to stop)...')
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
                kb_row = poll_keyboard(keyboard_whitelist)
                m_row = poll_mouse(mouse_whitelist, mouse_dpi)
                gp_row = poll_gamepad(gamepad_whitelist)
                row = kb_row + m_row + gp_row
                if not (row.count(0) == len(row)):
                    csv_writer.writerow(row)
            time.sleep(1.0 / polling_rate)
    print('Data collection stopped. Inputs saved.')

# =============================================================================
# Dataset
# =============================================================================
class InputDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, sequence_length, whitelist, label=0):
        self.sequence_length = sequence_length
        self.label = label
        data_frame = pandas.read_csv(file_path)
        self.feature_columns = [col for col in whitelist if col in data_frame.columns]
        if 'deltaX' in self.feature_columns:
            data_frame['deltaX'] = scaler.fit_transform(data_frame['deltaX'].values.reshape(-1, 1)).flatten()
        if 'deltaY' in self.feature_columns:
            data_frame['deltaY'] = scaler.fit_transform(data_frame['deltaY'].values.reshape(-1, 1)).flatten()
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
# Models
# =============================================================================
class UnsupervisedModel(lightning.LightningModule):
    def __init__(self, num_features, layers, learning_rate, dropout, sequence_length, mouse_scalers, save_dir, trial_number=None):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length
        self.mouse_scalers = mouse_scalers
        self.learning_rate = learning_rate

        self.encoders = torch.nn.ModuleList()
        dec_dim = num_features
        for enc_dim in layers:
            encoder = torch.nn.LSTM(
                input_size=dec_dim,
                hidden_size=enc_dim,
                batch_first=True,
                dropout=dropout
            )
            self.encoders.append(encoder)
            dec_dim = enc_dim

        self.decoders = torch.nn.ModuleList()
        last_enc_dim = layers[-1]
        for dec_dim in reversed(layers):
            decoder = torch.nn.LSTM(
                input_size=last_enc_dim,
                hidden_size=dec_dim,
                batch_first=True,
                dropout=dropout
            )
            self.decoders.append(decoder)
            last_enc_dim = dec_dim
        self.output_layer = torch.nn.Linear(layers[0], num_features)

        self.loss_function = torch.nn.MSELoss()
        self.train_metric_history = []
        self.val_metric_history = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.test_metric_history = []
        self.epoch_indices = []
        self.epoch_counter = 0

        self.save_dir = save_dir
        self.trial_number = trial_number
        self.figure, self.axes = matplotlib.pyplot.subplots()

    def forward(self, input_sequence):
        output = input_sequence
        for enc_lstm in self.encoders:
            output, (hidden_state, cell_state) = enc_lstm(output)
        for dec_lstm in self.decoders:
            output, (hidden_state, cell_state) = dec_lstm(output)
        reconstruction = self.output_layer(output)
        return reconstruction

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = self.loss_function(reconstruction, inputs)
        self.train_metric_history.append(reconstruction_error.detach().cpu())
        return reconstruction_error

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = self.loss_function(reconstruction, inputs)
        self.val_metric_history.append(reconstruction_error.detach().cpu())
        self.log('val_loss', reconstruction_error, prog_bar=True, on_epoch=True)
        return reconstruction_error
    
    def on_validation_epoch_end(self):
        avg_train_loss = torch.stack(self.train_metric_history).mean().item() if self.train_metric_history else None
        avg_val_loss = torch.stack(self.val_metric_history).mean().item() if self.val_metric_history else None
        if avg_train_loss is None or avg_val_loss is None:
            return
        self.epoch_indices.append(self.epoch_counter)
        self.epoch_counter += 1
        self.avg_train_losses.append(avg_train_loss)
        self.avg_val_losses.append(avg_val_loss)
        self.train_metric_history = []
        self.val_metric_history = []

    def on_fit_end(self):
        self.axes.clear()
        self.axes.plot(self.epoch_indices, self.avg_train_losses, label='Train Loss')
        self.axes.plot(self.epoch_indices, self.avg_val_losses, label='Val Loss')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('MSE')
        self.axes.legend()
        self.figure.savefig(f'{self.save_dir}/trial{self.trial_number}_unsupervised_{time.strftime('%Y%m%d-%H%M%S')}.png')
        matplotlib.pyplot.close(self.figure)
        return super().on_fit_end()
        
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = torch.sqrt(self.loss_function(reconstruction, inputs))
        self.test_metric_history.append(reconstruction_error.detach().cpu())
    
    def on_test_end(self):
        self.axes.clear()
        self.axes.plot(list(range(len(self.test_metric_history))), self.test_metric_history)
        self.axes.set_xlabel('Sequence')
        self.axes.set_ylabel('Reconstruction Error (MSE)')
        self.axes.set_title(f'Average Error: {torch.stack(self.test_metric_history).mean()}')
        self.figure.savefig(f'{self.save_dir}/report_unsupervised_{time.strftime('%Y%m%d-%H%M%S')}.png')
        return super().on_test_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class SupervisedModel(lightning.LightningModule):
    def __init__(self, num_features, layers, learning_rate, dropout, sequence_length, mouse_scalers, save_dir, trial_number=None):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length
        self.mouse_scalers = mouse_scalers
        self.learning_rate = learning_rate

        self.layers = torch.nn.ModuleList()
        input_dim = num_features
        for hidden_dim in layers:
            layer = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=dropout
            )
            self.layers.append(layer)
            input_dim = hidden_dim
        self.classifier_layer = torch.nn.Linear(layers[-1], 1)

        self.loss_function = torch.nn.BCEWithLogitsLoss()

        self.train_metric_history = []
        self.val_metric_history = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.test_metric_history = []
        self.epoch_indices = []
        self.epoch_counter = 0
        
        self.save_dir = save_dir
        self.trial_number = trial_number
        self.figure, self.axes = matplotlib.pyplot.subplots()

    def forward(self, input_sequence):
        output = input_sequence
        for lstm_layer in self.layers:
            output, (hidden_state, cell_state) = lstm_layer(output)
        last_output = output[:, -1, :]
        class_prediction = self.classifier_layer(last_output).squeeze(1)
        return class_prediction
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        class_prediction = self.forward(inputs)
        loss = self.loss_function(class_prediction, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        avg_train_loss = torch.stack(self.train_metric_history).mean().item() if self.train_metric_history else None
        avg_val_loss = torch.stack(self.val_metric_history).mean().item() if self.val_metric_history else None
        if avg_train_loss is None or avg_val_loss is None:
            return
        self.epoch_indices.append(self.epoch_counter)
        self.epoch_counter += 1
        self.avg_train_losses.append(avg_train_loss)
        self.avg_val_losses.append(avg_val_loss)
        self.train_metric_history = []
        self.val_metric_history = []

    def on_fit_end(self):
        self.axes.clear()
        self.axes.plot(self.epoch_indices, self.avg_train_losses, label='Train Loss')
        self.axes.plot(self.epoch_indices, self.avg_val_losses, label='Val Loss')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('BCELoss')
        self.axes.legend()
        self.figure.savefig(f'{self.save_dir}/trial{self.trial_number}_supervised_{time.strftime('%Y%m%d-%H%M%S')}.png')
        matplotlib.pyplot.close(self.figure)
        return super().on_fit_end()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs)
        confidence = torch.sigmoid(logits)
        self.test_metric_history.append(confidence.detach().cpu())

    def on_test_end(self):
        self.axes.clear()
        self.axes.plot(list(range(len(self.test_metric_history))), self.test_metric_history)
        self.axes.set_xlabel('Sequence')
        self.axes.set_ylabel('Confidence')
        self.axes.set_title(f'Average Confidence: {torch.stack(self.test_metric_history).mean()}')
        self.figure.savefig(f'{self.save_dir}/report_supervised_{time.strftime('%Y%m%d-%H%M%S')}.png')
        return super().on_test_end()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# =============================================================================
# Training Process
# =============================================================================
def train_model(configuration):
    model_type = configuration['model_type']
    model_structure = configuration['model_structure']
    sequence_length = configuration['sequence_length']
    tuning_patience = configuration['tuning_patience']
    keyboard_whitelist = configuration['keyboard_whitelist']
    mouse_whitelist = configuration['mouse_whitelist']
    mouse_scalers = configuration['mouse_scalers']
    gamepad_whitelist = configuration['gamepad_whitelist']
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist

    # Preprocessing
    if whitelist:
        train_files = tkinter.filedialog.askopenfilenames(
            title='Select non-cheat training files',
            filetypes=[('CSV Files', '*.csv')]
        )
        if not train_files:
            print('No training files selected. Exiting...')
            return
        val_files = tkinter.filedialog.askopenfilenames(
            title='Select non-cheat validation files',
            filetypes=[('CSV Files', '*.csv')]
        )
        if not val_files:
            print('No validation files selected. Exiting...')
            return

        train_datasets = [InputDataset(file, sequence_length, mouse_scalers, whitelist) for file in train_files]
        val_datasets = [InputDataset(file, sequence_length, mouse_scalers, whitelist) for file in val_files]

        if model_type == 'supervised':
            cheat_train_files = tkinter.filedialog.askopenfilenames(
                title='Select cheat training files',
                filetypes=[('CSV Files', '*.csv')]
            )
            if not cheat_train_files:
                print('No files selected. Exiting...')
                return
            cheat_val_files = tkinter.filedialog.askopenfilenames(
                title='Select cheat validation files',
                filetypes=[('CSV Files', '*.csv')]
            )
            if not cheat_val_files:
                print('No files selected. Exiting...')
                return
            train_datasets += [InputDataset(file, sequence_length, mouse_scalers, whitelist, label=1) for file in cheat_train_files]
            val_datasets += [InputDataset(file, sequence_length, mouse_scalers, whitelist, label=1) for file in cheat_val_files]

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        val_dataset = torch.utils.data.ConcatDataset(val_datasets)
        

        save_dir = tkinter.filedialog.askdirectory(title='Select model/graph save location')

    # Tuning and Training
    class ConsecutivePrunedTrialsCallback:
        def __init__(self, patience):
            self.patience = patience
            self.consecutive_pruned = 0

        def __call__(self, study: optuna.Study, trial: optuna.Trial):
            if trial.state == optuna.trial.TrialState.PRUNED:
                self.consecutive_pruned += 1
            else:
                self.consecutive_pruned = 0
            if self.consecutive_pruned >= self.patience:
                print(f'Stopping study: {self.consecutive_pruned} consecutive pruned trials.')
                study.stop()

    def objective(trial):
        batch_size = trial.suggest_int('batch_size', 32, 512, log=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if model_structure == []:
            num_layers = trial.suggest_int('num_layers', 1, 5)
            layers = []
            for i in range(num_layers):
                layer = trial.suggest_int(f'layer{i}_dim', 1, 256, step=15)
                layers.append(layer)
        else:
            num_layers = len(model_structure)
            layers = model_structure

        learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.01, log=True)
        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        
        if model_type == 'unsupervised':
            model = UnsupervisedModel(
                num_features=len(whitelist),
                layers=layers,
                learning_rate=learning_rate,
                dropout=dropout,
                sequence_length=sequence_length,
                mouse_scalers=mouse_scalers,
                save_dir=save_dir,
                trial_number=trial.number
            )
        else:
            model = SupervisedModel(
                num_features=len(whitelist),
                layers=layers,
                learning_rate=learning_rate,
                dropout=dropout,
                sequence_length=sequence_length,
                mouse_scalers=mouse_scalers,
                save_dir=save_dir,
                trial_number=trial.number
            )

        early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=10,
            mode='min'
        )
        checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=save_dir,
            filename=f"trial_{trial.number}_best",
            save_top_k=1,
            mode='min'
        )
        prune_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_loss')
        trainer = lightning.Trainer(
            max_epochs=512,
            precision='16-mixed',
            callbacks=[early_stop_callback, checkpoint_callback, prune_callback],
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        trainer.fit(model, train_loader, val_loader)
        print(f'[Early Stopping Triggered!] Trial {trial.number} stopped at epoch {trainer.current_epoch}.')
        best_checkpoint = checkpoint_callback.best_model_path
        if best_checkpoint:
            trial.set_user_attr('best_checkpoint', best_checkpoint)
        
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            raise ValueError('Validation loss not found!')
        #param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return val_loss.item()

    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    study = optuna.create_study(direction='minimize')
    consecutive_prune_callback = ConsecutivePrunedTrialsCallback(tuning_patience)
    study.optimize(
        objective,
        n_trials=512,
        callbacks=[consecutive_prune_callback],
        gc_after_trial=True
    )

    importances = optuna.importance.get_param_importances(study)
    print('\nTuning Importances:')
    for key, value in importances.items():
        print(f'{key}:{value}')
    best_trials = study.best_trials
    print('\nBest Trials:')
    for trial in best_trials:
        print(f'Trial: {trial.number} Loss: {trial.values[0]} Params: {trial.values[1]}')
    print('\nPlease copy your desired model from the local models directory.')

# =============================================================================
# Static Analysis Process
# =============================================================================
def run_static_analysis(configuration):
    model_type = configuration['model_type']
    sequence_length = configuration['sequence_length']
    keyboard_whitelist = configuration['keyboard_whitelist']
    mouse_whitelist = configuration['mouse_whitelist']
    gamepad_whitelist = configuration['gamepad_whitelist']
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
    if file and checkpoint:
        if model_type == 'unsupervised':
            model = UnsupervisedModel.load_from_checkpoint(checkpoint)
        else:
            model = SupervisedModel.load_from_checkpoint(checkpoint)
        model.save_dir = report_dir

        test_dataset = InputDataset(file, sequence_length, model.mouse_scalers, whitelist)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        trainer = lightning.Trainer(
            logger=False,
            enable_checkpointing=False,
        )
        trainer.test(model, dataloaders=test_loader, ckpt_path=None)
    else:
        print('Data or model not selected. Exiting.')

# =============================================================================
# Live Analysis Mode
# =============================================================================
def run_live_analysis(configuration, root):
    kill_key = configuration['kill_key']
    capture_bind = configuration['capture_bind']
    model_type = configuration['model_type']
    polling_rate = configuration['polling_rate']
    sequence_length = configuration['sequence_length']
    keyboard_whitelist = configuration['keyboard_whitelist']
    mouse_whitelist = configuration['mouse_whitelist']
    mouse_dpi = configuration['mouse_dpi']
    gamepad_whitelist = configuration['gamepad_whitelist']
    whitelist = keyboard_whitelist + mouse_whitelist + gamepad_whitelist
    
    x_index, y_index = None
    for i, feature in enumerate(whitelist):
        if feature == 'deltaX':
            x_index = i
        if feature == 'deltaY':
            y_index = i
    
    checkpoint = tkinter.filedialog.askopenfilename(
        title='Select model checkpoint file',
        filetypes=[('Checkpoint Files', '*.ckpt')]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'unsupervised':
        model = UnsupervisedModel.load_from_checkpoint(checkpoint)
    else:
        model = SupervisedModel.load_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    if any(key in mouse_whitelist for key in ('deltaX', 'deltaY', 'angle')):
        raw_input_thread = threading.Thread(target=listen_for_mouse_movement, daemon=True)
        raw_input_thread.start()
    sequence = []

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
            kb_row = poll_keyboard(keyboard_whitelist)
            m_row = poll_mouse(mouse_whitelist, mouse_dpi)
            gp_row = poll_gamepad(gamepad_whitelist)
            row = kb_row + m_row + gp_row
            if not (row.count(0) == len(row)):
                sequence.append(row)

        if len(sequence) >= sequence_length:
            if x_index is not None:
                for row in sequence:
                    row[x_index] = scaler.fit_transform(numpy.array(row[x_index]).reshape(-1, 1)).flatten()
            if y_index is not None:
                for row in sequence:
                    row[y_index] = scaler.fit_transform(numpy.array(row[y_index]).reshape(-1, 1)).flatten()
            input_sequence = torch.tensor([sequence[-sequence_length:]], dtype=torch.float32, device=device)
            if model_type == 'unsupervised':
                reconstruction = model(input_sequence)
                reconstruction_error = model.loss_function(reconstruction, input_sequence)
                print(f'Reconstruction Error: {reconstruction_error.item()}')
            else:
                logits = model(input_sequence)
                confidence = torch.sigmoid(logits).mean()
                print(f'Confidence: {confidence.item()}')
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

    matplotlib.use("Agg")
    root = tkinter.Tk()
    root.withdraw()
    with open('config.json', 'r') as file_handle:
        configuration = json.load(file_handle)
    mode = configuration['mode']
    if mode == 'collect':
        collect_input_data(configuration)
    elif mode == 'train':
        train_model(configuration)
    elif mode == 'test':
        run_static_analysis(configuration)
    elif mode == 'deploy':
        run_live_analysis(configuration, root)
    else:
        print(f'Error: Invalid mode specified in configuration file: {mode}')
    root.destroy()

if __name__ == '__main__':
    main()