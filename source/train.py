import matplotlib.pyplot
import torch.utils.data
import lightning
import logging
import optuna
import time
import utilities

class KillKeyCallback:
    """Optuna callback to stop a study when certain keys are pressed."""
    def __init__(self, config: dict):
        self.config = config
        self.stop_flag = False

    def stop(self):
        print("Stopping study. Finishing current trial...")
        self.stop_flag = True

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if not self.stop_flag:
            if utilities.should_kill(self.config):
                self.stop()

        if self.stop_flag:
            study.stop()

def create_dataloaders(config: dict) -> tuple:
    """Creates and returns training and validation dataloaders."""
    whitelist = config.get('keyboard_whitelist') + config.get('mouse_whitelist') + config.get('gamepad_whitelist')

    all_training_files = config.get('training_files')
    
    if config.get('model_class').training_type == 'supervised':
        all_training_files += config.get('cheat_training_files')
    
    utilities.fit_global_scaler(all_training_files, whitelist)
    
    train_datasets = [utilities.InputDataset(file, config) for file in config.get('training_files')]
    val_datasets = [utilities.InputDataset(file, config) for file in config.get('validation_files')]

    if config.get('model_class').training_type == 'supervised':
        train_datasets += [utilities.InputDataset(file, config, label=1) for file in config.get('cheat_training_files')]
        val_datasets += [utilities.InputDataset(file, config, label=1) for file in config.get('cheat_validation_files')]

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    train_loader = torch.utils.data.DataLoader(train_dataset, sequences_per_batch=config.get('sequences_per_batch'), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, sequences_per_batch=config.get('sequences_per_batch'), shuffle=False)
    return train_loader, val_loader

def objective(trial: optuna.Trial, config: dict, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> float:
    """Objective function for hyperparameter tuning."""
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout_rate = 0.0
    if num_layers > 1:
        dropout_rate = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)

    hyperparams : dict = {
        'hidden_dim' : trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256]),
        'num_layers' : num_layers,
        'learning_rate' : trial.suggest_float('learning_rate', 0.00001, 0.01, log=True),
        'dropout' : dropout_rate
    }
    
    whitelist = config.get('keyboard_whitelist') + config.get('mouse_whitelist') + config.get('gamepad_whitelist')
    model = config.get('model_class')(len(whitelist), hyperparams, config.get('polls_per_sequence'), config.get('save_dir'), trial.number)
    param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)

    early_stop_callback = lightning.pytorch.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-8,
        patience=10,
        mode='min'
    )
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=config.get('save_dir'),
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
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is None:
            # This can happen if training is interrupted before the first validation epoch completes.
            # Pruning or early stopping might also cause this.
            print(f'Trial {trial.number} did not produce a validation loss.')
            raise optuna.exceptions.TrialPruned()
    finally:
        if hasattr(model, 'figure'):
            matplotlib.pyplot.close(model.figure)

    print(f'[Early Stopping Triggered!] Trial {trial.number} stopped at epoch {trainer.current_epoch}.')
    best_checkpoint = checkpoint_callback.best_model_path
    if best_checkpoint:
        trial.set_user_attr('best_checkpoint', best_checkpoint)
    return val_loss.item()

def train_model(config: dict) -> None:
    """Tunes hyperparameters and trains the model based on the provided configuration."""
    train_loader, val_loader = create_dataloaders(config)
    if not train_loader or not val_loader:
        return print('Dataset loading cancelled. Exiting...')

    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective(trial, config, train_loader, val_loader),
        n_trials=2048,
        callbacks=[KillKeyCallback(config)],
        gc_after_trial=True
    )
    
    print('\nTuning Importances:')
    importances = optuna.importance.get_param_importances(study)
    for key, value in importances.items():
        print(f'{key}:{value}')
    best_trial = study.best_trial
    print(f'\nBest Trial: {best_trial.number} Loss: {best_trial.values[0]}')