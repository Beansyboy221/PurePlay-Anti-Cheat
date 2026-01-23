import optuna.integration.pytorch_lightning
import lightning.pytorch.profilers
import lightning.pytorch.callbacks
import optuna.visualization
import lightning
import psutil
import torch
import numpy
import time
import onnx # Eventually export to onnx
import preprocessing, utilities

def train_model(config: object) -> None:
    data_module = preprocessing.TrainingDataModule(config)
    data_module.setup()
    kill_callback = KillKeyCallback(config)
    study = optuna.create_study(
        study_name=f'{config.model_class} {time.ctime()}',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    try:
        study.optimize(
            lambda trial: _objective(trial, config, data_module, kill_callback),
            callbacks=[kill_callback],
            gc_after_trial=True
        )
    except StopIteration:
        print("Study terminated by user.")
    
    if len(study.trials) <= 1:
        print("Not enough trials completed for a comparison. Skipping reporting.")
        return
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val loss: {study.best_value:.6f}")
    figure = optuna.visualization.plot_optimization_history(study)
    figure.write_html(f'{config.save_dir}/optimization_history.html')
    figure = optuna.visualization.plot_param_importances(study)
    figure.write_html(f'{config.save_dir}/param_importances.html')
    figure = optuna.visualization.plot_parallel_coordinate(study)
    figure.write_html(f'{config.save_dir}/parallel_coordinate.html')
    print('Report graphs saved.')
    print('Training complete.')

#region Custom Callbacks
class KillKeyCallback(lightning.pytorch.callbacks.Callback):
    def __init__(self, config: object):
        super().__init__()
        self.config = config
        self.must_stop_study = False

    def on_train_batch_end(self, trainer, *args, **kwargs):
        """Pytorch Lightning hook to stop mid-trial."""
        if not self.must_stop_study and utilities.should_kill(self.config):
            self.must_stop_study = True
            trainer.should_stop = True

    def on_validation_batch_end(self, trainer, *args, **kwargs):
        """Pytorch Lightning hook to stop mid-trial."""
        if not self.must_stop_study and utilities.should_kill(self.config):
            self.must_stop_study = True
            trainer.should_stop = True

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Optuna hook: Stops the study after the objective function returns."""
        if self.must_stop_study:
            study.stop()
#endregion

#region Objective Functions
def _objective(trial: optuna.Trial, config: object, data_module: preprocessing.TrainingDataModule, kill_callback: KillKeyCallback) -> float:
    """Objective function for hyperparameter tuning."""
    model_params = _suggest_model_params(trial)
    data_params: dict = {
        'whitelist': config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist,
        'polling_rate': data_module.polling_rate,
        'ignore_empty_polls': config.ignore_empty_polls,
        'polls_per_sequence': config.polls_per_sequence,
        'sequences_per_batch': config.sequences_per_batch
    }

    model = config.model_class(model_params, data_params)
    model.set_scaler_params(data_module.scaler.means, data_module.scaler.stds)
    _enforce_performance_constraints(config, model, data_params)

    swa_start = trial.suggest_int("swa_epoch_start", 20, 2000)
    swa_lr = model_params.get('learning_rate') * trial.suggest_float("swa_lr_factor", 0.05, 0.5)
    trial_directory = f'{config.save_dir}/trial_{trial.number}'
    trainer = lightning.Trainer(
        max_epochs=-1, # Should I use max_time to limit trial duration?
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=[
            kill_callback,
            lightning.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001, verbose=True),
            lightning.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=trial_directory),
            optuna.integration.pytorch_lightning.PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            lightning.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=swa_lr, swa_epoch_start=swa_start)
        ],
        profiler=lightning.pytorch.profilers.SimpleProfiler(dirpath=trial_directory, filename='performance_log'),
        logger=False,
        enable_model_summary=False
    )

    print(f'\nTrial: {trial.number}')
    trainer.fit(model, datamodule=data_module)
    
    val_loss = trainer.callback_metrics.get('val_loss')
    return val_loss.item() if val_loss is not None else float('inf')
#endregion

#region Helpers
def _suggest_model_params(trial: optuna.Trial) -> dict:
    num_layers = trial.suggest_int('num_layers', 1, 4)
    layer_sizes = []
    for i in range(num_layers):
        layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 8, 256))
    dropout = trial.suggest_float('dropout', 0.0, 0.4, step=0.1) if num_layers > 1 else 0.0
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop", "SGD"])
    model_params: dict = {
        'num_layers': num_layers,
        'layer_sizes': layer_sizes,
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
        'optimizer_name': optimizer_name,
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        'scheduler_name': trial.suggest_categorical("scheduler_name", ["No Scheduler", "ReduceLROnPlateau"]),
        'dropout': dropout
    }
    if optimizer_name != 'Adam':
        model_params['momentum'] = trial.suggest_float("momentum", 0.5, 1.0, log=True)
    return model_params

def _enforce_performance_constraints(config: object, model: lightning.LightningModule, data_params: dict) -> None:
    model.eval() 
    process = psutil.Process()
    dummy_input = torch.randn(1, config.polls_per_sequence, len(data_params.get('whitelist')))
    
    # Inference P99 Latency
    with torch.no_grad():
        for _ in range(10): 
            model(dummy_input)
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
    p99_latency = numpy.percentile(latencies, 99)
    if p99_latency > config.max_inference_latency_ms:
        raise optuna.exceptions.TrialPruned(f"P99 Latency too high: {p99_latency:.2f}ms > {config.max_inference_latency_ms}ms")

    # Sustained CPU & Peak Memory
    # Run a tight loop for a fixed duration to measure sustained load
    duration_sec = 2.0
    start_profile = time.time()
    process.cpu_percent(interval=None) 
    with torch.no_grad():
        while time.time() - start_profile < duration_sec:
            model(dummy_input)
    sustained_cpu = process.cpu_percent(interval=None) # Average usage since last call
    current_memory_mb = process.memory_info().rss / (1024 * 1024) # Resident Set Size in MB
    if sustained_cpu > config.max_cpu_usage_percent:
        raise optuna.exceptions.TrialPruned(f"CPU usage too high: {sustained_cpu:.1f}%")
    if current_memory_mb > config.max_peak_memory_mb:
         raise optuna.exceptions.TrialPruned(f"Memory usage too high: {current_memory_mb:.1f}MB")

    model.train()
#endregion