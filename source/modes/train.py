import optuna.integration.pytorch_lightning
import lightning.pytorch.profilers
import lightning.pytorch.callbacks
import lightning.pytorch.tuner
import optuna.visualization
import lightning
import psutil
import torch
import numpy
import time
#import glob
import json
import onnx
import preprocessing, custom_callbacks

MAX_LAYERS = 4
MAX_LAYER_SIZE = 256

def train_model(config: object) -> None:
    # Load and configure the data
    data_module = preprocessing.TrainingDataModule(config)
    data_module.setup()
    data_params: dict = { # Should the model just read from the data module? Is this necessary?
        'whitelist': config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist,
        'polling_rate': data_module.polling_rate,
        'ignore_empty_polls': config.ignore_empty_polls,
        'polls_per_sequence': config.polls_per_sequence,
        'sequences_per_batch': data_module.batch_size
    }
    kill_callback = custom_callbacks.KillTrainingCallback(config)
    _tune_batch_size(config, data_module, data_params, kill_callback)

    # Start optuna study
    study = optuna.create_study(study_name=f'{config.model_class} {time.ctime()}')
    study.optimize(
        lambda trial: _objective(trial, config, data_module, data_params, kill_callback),
        callbacks=[kill_callback],
        gc_after_trial=True
    )
    if len(study.trials) == 0:
        print("No trials completed. Exiting early.")
        return
    
    # Export the best model to .pt and .onnx
    # trial_dir = f'{config.save_dir}/trial_{study.best_trial.number}'
    # checkpoint_path = glob.glob(f"{trial_dir}/*.ckpt")[0]
    # model = config.model_class.load_from_checkpoint(checkpoint_path, data_params=data_params)
    # model.eval()
    # _export_model_to_torchscript(config, model)
    # _export_model_to_onnx(config, model)
    
    # Generate reports
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val loss: {study.best_value:.6f}")
    if len(study.trials) == 1:
        print("Not enough trials completed for a comparison. Skipping reporting.")
        return
    figure = optuna.visualization.plot_optimization_history(study)
    figure.write_html(f'{config.save_dir}/optimization_history.html')
    figure = optuna.visualization.plot_param_importances(study)
    figure.write_html(f'{config.save_dir}/param_importances.html')
    figure = optuna.visualization.plot_parallel_coordinate(study)
    figure.write_html(f'{config.save_dir}/parallel_coordinate.html')
    print('Report graphs saved.')

#region Objective Functions
def _objective(trial: optuna.Trial, config: object, data_module: preprocessing.TrainingDataModule, data_params: dict, kill_callback: custom_callbacks.KillTrainingCallback) -> float:
    """Objective function for hyperparameter tuning."""
    # Set up model
    model_params = _suggest_model_params(trial)
    model = config.model_class(model_params, data_params)
    model.set_scaler_params(data_module.scaler.means, data_module.scaler.stds)
    _enforce_performance_constraints(config, model, data_params)

    # Set up trainer
    #swa_start = trial.suggest_int("swa_epoch_start", 20, 2000)
    #swa_lr = model_params.get('learning_rate') * trial.suggest_float("swa_lr_factor", 0.05, 0.5)
    trial_directory = f'{config.save_dir}/trial_{trial.number}'
    trainer = lightning.Trainer(
        max_epochs=-1,
        precision="16-mixed",
        callbacks=[
            kill_callback,
            lightning.pytorch.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001, verbose=True),
            lightning.pytorch.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=trial_directory),
            optuna.integration.pytorch_lightning.PyTorchLightningPruningCallback(trial, monitor="val_loss")
            #,lightning.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=swa_lr, swa_epoch_start=swa_start)
        ],
        profiler=lightning.pytorch.profilers.SimpleProfiler(dirpath=trial_directory, filename='performance_log'),
        logger=False,
        enable_model_summary=False
    )

    # Train
    print(f'\nTrial: {trial.number}')
    trainer.fit(model, datamodule=data_module)
    val_loss = trainer.callback_metrics.get('val_loss')
    return val_loss.item() if val_loss is not None else float('inf')
#endregion

#region Helpers
def _tune_batch_size(config: object, data_module: preprocessing.TrainingDataModule, data_params: dict, kill_callback: custom_callbacks.KillTrainingCallback) -> int:
    """Automatically maximizes batch size for the worst-case model from the objective function. Sets batch size in the data module."""
    print('Tuning batch size for hardware...')
    model_params: dict = {
        'num_layers': MAX_LAYERS,
        'layer_sizes': [MAX_LAYER_SIZE] * MAX_LAYERS,
        'dropout': 0.0
    }
    model = config.model_class(model_params, data_params)
    model.set_scaler_params(data_module.scaler.means, data_module.scaler.stds)
    trainer = lightning.Trainer(max_epochs=50, logger=False, enable_model_summary=False, callbacks=kill_callback)
    tuner = lightning.pytorch.tuner.Tuner(trainer)
    optimal_batch_size = tuner.scale_batch_size(model=model, datamodule=data_module) # This function also sets the self.batch_size property in the data module.
    print(f'Optimal batch size found: {optimal_batch_size}')
    return optimal_batch_size

def _suggest_model_params(trial: optuna.Trial) -> dict:
    """Tunes all model hyperparameters for a given optuna trial."""
    model_params: dict = {
        'num_layers':  trial.suggest_int('num_layers', 1, MAX_LAYERS),
        'optimizer_name': trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop", "SGD"]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        'scheduler_name': trial.suggest_categorical("scheduler_name", ["No Scheduler", "ReduceLROnPlateau"])
    }

    # Dependent hyperparams
    layer_sizes = []
    for i in range(model_params['num_layers']):
        layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 8, MAX_LAYER_SIZE))
    model_params['layer_sizes'] = layer_sizes
    if model_params['optimizer_name'] != 'Adam':
        model_params['momentum'] = trial.suggest_float("momentum", 0.5, 1.0, log=True)
    model_params['dropout'] = trial.suggest_float('dropout', 0.0, 0.4, step=0.1) if model_params['num_layers'] > 1 else 0.0

    return model_params

def _enforce_performance_constraints(config: object, model: lightning.LightningModule, data_params: dict) -> None:
    """Limits an optuna trial based on model performance thresholds defined in the config."""
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

def _export_model_to_torchscript(config: object, model: lightning.LightningModule) -> None:
    print('Exporting model to TorchScript...')
    torchscript_path = f'{config.save_dir}/best_model.pt'
    pt_model = model.to_torchscript()
    
    # Add metadata
    params = model.hparams.data_params
    pt_model.register_attribute("whitelist_json", str, str(params['whitelist']))
    pt_model.register_attribute("polling_rate", int, params['polling_rate'])
    pt_model.register_attribute("ignore_empty_polls", bool, params['ignore_empty_polls'])
    pt_model.register_attribute("polls_per_sequence", int, params['polls_per_sequence'])
    pt_model.save(torchscript_path)
    print(f'Successfully saved TorchScript with metadata to: {torchscript_path}')

def _export_model_to_onnx(config: object, model: lightning.LightningModule) -> None:
    print('Exporting model to ONNX...')
    onnx_path = f"{config.save_dir}/model.onnx"
    dummy_input = torch.randn(1, config.polls_per_sequence, len(model.hparams.data_params['whitelist']))
    model.to_onnx(
        onnx_path,
        dummy_input,
        export_params=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # Add metadata
    model_proto = onnx.load(onnx_path)
    meta = model_proto.metadata_props.add()
    meta.key = "data_params"
    meta.value = json.dumps(model.hparams.data_params) # Serialize dict to JSON string
    onnx.save(model_proto, onnx_path)
    print(f"Successfully saved ONNX with metadata to: {onnx_path}")
#endregion