import logging
import time
import tkinter.filedialog
import optuna
import lightning
import matplotlib.pyplot
import utilities
import keyboard
import torch.utils.data

class KillKeyCallback:
    """Optuna callback to stop a study when a specific key is pressed."""
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

def _create_dataloaders(config: dict, whitelist: list[str], model_class: type[lightning.LightningModule]) -> tuple:
    """Creates and returns training and validation dataloaders."""
    train_files = utilities._prompt_for_csv_files('Select non-cheat training files')
    if not train_files:
        return None, None
    val_files = utilities._prompt_for_csv_files('Select non-cheat validation files')
    if not val_files:
        return None, None

    train_datasets = [utilities.InputDataset(file, config['sequence_length'], whitelist) for file in train_files]
    val_datasets = [utilities.InputDataset(file, config['sequence_length'], whitelist) for file in val_files]

    if model_class.training_type == 'supervised':
        cheat_train_files = utilities._prompt_for_csv_files('Select cheat training files')
        if not cheat_train_files:
            return None, None
        cheat_val_files = utilities._prompt_for_csv_files('Select cheat validation files')
        if not cheat_val_files:
            return None, None
        train_datasets += [utilities.InputDataset(file, config['sequence_length'], whitelist, label=1) for file in cheat_train_files]
        val_datasets += [utilities.InputDataset(file, config['sequence_length'], whitelist, label=1) for file in cheat_val_files]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader

def train_model(config: dict) -> None:
    model_class = config['model_class']
    whitelist = config['keyboard_whitelist'] + config['mouse_whitelist'] + config['gamepad_whitelist']

    # Preprocessing
    if not whitelist:
        return print('No input features selected. Exiting...')
    
    train_loader, val_loader = _create_dataloaders(config, whitelist, model_class)
    if not train_loader or not val_loader:
        print('Dataset loading cancelled. Exiting...')
        return

    save_dir = tkinter.filedialog.askdirectory(title='Select model/graph save location')
    if not save_dir:
        print('No save directory selected. Exiting...')
        return

    # Tuning and Training
    def objective(trial: optuna.Trial) -> float:
        hyperparams : dict = {
            'hidden_dim' : trial.suggest_int('hidden_dim', 1, 256, log=True),
            'num_layers' : trial.suggest_int('num_layers', 1, 6),
            'learning_rate' : trial.suggest_float('learning_rate', 0.00001, 0.01, log=True),
            'dropout' : trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        }
        if hyperparams['num_layers'] == 1:
            hyperparams['dropout'] = 0.0
        
        model = model_class(len(whitelist), hyperparams, config['sequence_length'], save_dir, trial.number)
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
        callbacks=[KillKeyCallback(config['kill_key'])],
        gc_after_trial=True
    )

    importances = optuna.importance.get_param_importances(study)
    print('\nTuning Importances:')
    for key, value in importances.items():
        print(f'{key}:{value}')
    best_trial = study.best_trial
    print(f'\nBest Trial: {best_trial.number} Loss: {best_trial.values[0]}')