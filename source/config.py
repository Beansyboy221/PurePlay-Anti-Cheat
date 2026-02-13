import tkinter.filedialog
import tkinter.ttk
import pydantic
import tkinter
import tomllib
import typing
import constants, models

#region Custom Validators
def _validate_model_name(model_name: str) -> typing.Any:
    if model_name in models.AVAILABLE_MODELS:
        return models.AVAILABLE_MODELS[model_name]
    raise ValueError(f'Invalid model class')

def _validate_even(value: int) -> int:
    if value % 2 != 0:
        raise ValueError(f'Value must be even, got {value}')
    return value
#endregion

#region App Mode Configs
class SharedConfig(pydantic.BaseModel):
    """Fields pulled directly from the root of the config file."""
    kill_bind_list: typing.List[str] = pydantic.Field(default_factory=lambda: ['ESC'])
    kill_bind_logic: typing.Literal['ANY', 'ALL'] = pydantic.Field(default='ANY')

class CollectConfig(SharedConfig):
    """Fields pulled from the [collect] section of the config file."""
    mode: typing.Literal[constants.AppMode.COLLECT]
    save_dir: pydantic.DirectoryPath = pydantic.Field(validation_alias=pydantic.AliasPath('collect', 'save_dir'))
    polling_rate: int = pydantic.Field(default=60, validation_alias=pydantic.AliasPath('collect', 'polling_rate'))
    ignore_empty_polls: bool = pydantic.Field(default=True, validation_alias=pydantic.AliasPath('collect', 'ignore_empty_polls'))
    capture_bind_list: typing.List[str] = pydantic.Field(
        default_factory=lambda: ['right'], 
        validation_alias=pydantic.AliasPath('collect', 'capture_bind_list')
    )
    capture_bind_logic: typing.Literal['ANY', 'ALL'] = pydantic.Field(
        default='ANY', 
        validation_alias=pydantic.AliasPath('collect', 'capture_bind_logic')
    )
    keyboard_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('collect', 'keyboard_whitelist')
    )
    mouse_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: ['deltaX', 'deltaY'],
        validation_alias=pydantic.AliasPath('collect', 'mouse_whitelist')
    )
    gamepad_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('collect', 'gamepad_whitelist')
    )

class TrainConfig(SharedConfig):
    """Fields pulled from the [train] section of the config file."""
    mode: typing.Literal[constants.AppMode.TRAIN]
    save_dir: pydantic.DirectoryPath = pydantic.Field(validation_alias=pydantic.AliasPath('train', 'save_dir'))
    model_class: typing.Annotated[typing.Any, pydantic.BeforeValidator(_validate_model_name)] = pydantic.Field(validation_alias=pydantic.AliasPath('train', 'model_class'))
    max_inference_latency_ms: int = pydantic.Field(default=10000000, validation_alias=pydantic.AliasPath('train', 'max_inference_latency_ms'))
    max_peak_memory_mb: int = pydantic.Field(default=10000000, validation_alias=pydantic.AliasPath('train', 'max_peak_memory_mb'))
    ignore_empty_polls: bool = pydantic.Field(default=True, validation_alias=pydantic.AliasPath('train', 'ignore_empty_polls'))
    polls_per_sequence: typing.Annotated[int, pydantic.AfterValidator(_validate_even)] = pydantic.Field(default=128, validation_alias=pydantic.AliasPath('train', 'polls_per_sequence'))
    sequences_per_batch: typing.Annotated[int, pydantic.AfterValidator(_validate_even)] = pydantic.Field(default=128, validation_alias=pydantic.AliasPath('train', 'sequences_per_batch'))
    training_file_dir: pydantic.DirectoryPath = pydantic.Field(validation_alias=pydantic.AliasPath('train', 'training_file_dir'))
    validation_file_dir: pydantic.DirectoryPath = pydantic.Field(validation_alias=pydantic.AliasPath('train', 'validation_file_dir'))
    cheat_training_file_dir: typing.Optional[pydantic.DirectoryPath] = pydantic.Field(validation_alias=pydantic.AliasPath('train', 'cheat_training_file_dir')) # Made these optional for now. Remember to handle the case in which they aren't given but needed.
    cheat_validation_file_dir: typing.Optional[pydantic.DirectoryPath] = pydantic.Field(validation_alias=pydantic.AliasPath('train', 'cheat_validation_file_dir'))
    keyboard_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('train', 'keyboard_whitelist')
    )
    mouse_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: ['deltaX', 'deltaY'],
        validation_alias=pydantic.AliasPath('train', 'mouse_whitelist')
    )
    gamepad_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('train', 'gamepad_whitelist')
    )

class TestConfig(SharedConfig):
    """Fields pulled from the [test] section of the config file."""
    mode: typing.Literal[constants.AppMode.TEST]
    save_dir: pydantic.DirectoryPath = pydantic.Field(validation_alias=pydantic.AliasPath('test', 'save_dir'))
    model_file: pydantic.FilePath = pydantic.Field(validation_alias=pydantic.AliasPath('test', 'model_file'))
    model_class: typing.Annotated[typing.Any, pydantic.BeforeValidator(_validate_model_name)] = pydantic.Field(validation_alias=pydantic.AliasPath('test', 'model_class'))
    testing_file_dir: pydantic.DirectoryPath = pydantic.Field(validation_alias=pydantic.AliasPath('test', 'testing_file_dir'))

class DeployConfig(SharedConfig):
    """Fields pulled from the [deploy] section of the config file."""
    mode: typing.Literal[constants.AppMode.DEPLOY]
    save_dir: pydantic.DirectoryPath = pydantic.Field(validation_alias=pydantic.AliasPath('deploy', 'save_dir'))
    model_file: pydantic.FilePath = pydantic.Field(validation_alias=pydantic.AliasPath('deploy', 'model_file'))
    model_class: typing.Annotated[typing.Any, pydantic.BeforeValidator(_validate_model_name)] = pydantic.Field(validation_alias=pydantic.AliasPath('deploy', 'model_class'))
    write_to_file: bool = pydantic.Field(default=True)
    deployment_window_type: constants.WindowType = pydantic.Field(validation_alias=pydantic.AliasPath('deploy', 'deployment_window_type'))
    capture_bind_list: typing.List[str] = pydantic.Field(
        default_factory=lambda: ['right'], 
        validation_alias=pydantic.AliasPath('deploy', 'capture_bind_list')
    )
    capture_bind_logic: typing.Literal['ANY', 'ALL'] = pydantic.Field(
        default='ANY', 
        validation_alias=pydantic.AliasPath('deploy', 'capture_bind_logic')
    )

AppConfig = typing.Annotated[
    typing.Union[CollectConfig, TrainConfig, TestConfig, DeployConfig],
    pydantic.Field(discriminator='mode')
]
#endregion

#region Config Functions
def validate_config(config_dict: dict) -> typing.Optional[AppConfig]:
    """Parses nested TOML dict into a flat Union model."""
    try:
        config = pydantic.TypeAdapter(AppConfig).validate_python(config_dict)
        print(f'Config validated for mode: {config.mode}')
        return config
    except Exception as e:
        print(f'Validation Error:\n{e}')
        return None

def get_config_from_gui() -> dict:
    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.focus_force()
    file_path = tkinter.filedialog.askopenfilename(
        parent=root,
        title='Select TOML configuration file', 
        filetypes=[('TOML Files', '*.toml')]
    )
    root.destroy()
    
    if not file_path:
        return {}
    with open(file_path, 'rb') as file:
        return tomllib.load(file)

def _get_model_class_from_gui() -> None:
    selected_model = []
    root = tkinter.Tk()
    root.attributes('-topmost', True)
    root.focus_force()
    frame = tkinter.ttk.Frame(padding=5)
    frame.pack()
    tkinter.ttk.Label(frame, text='Select model class:').pack()
    combo_box = tkinter.ttk.Combobox(frame, values=list(models.AVAILABLE_MODELS.keys()))
    combo_box.pack()
    def _save_and_exit() -> None:
        model = combo_box.get()
        if model in models.AVAILABLE_MODELS:
            selected_model.append(model)
            root.destroy()
        else:
            print(f'Invalid model class: {model}')
    tkinter.ttk.Button(frame, text='OK', command=_save_and_exit).pack()
    root.mainloop()
    return selected_model[0] if selected_model else None

def populate_missing_configs_from_gui(config: dict) -> dict:
    mode = config.get('mode')
    config.setdefault('train', {})
    config.setdefault('test', {})
    config.setdefault('deploy', {})
    match mode:
        case constants.AppMode.COLLECT:
            collect_config = config['collect']
            if not collect_config.get('save_dir'):
                collect_config['save_dir'] = tkinter.filedialog.askdirectory(title='Select data save directory')
            pass
        case constants.AppMode.TRAIN:
            train_config = config['train']
            if not train_config.get('save_dir'):
                train_config['save_dir'] = tkinter.filedialog.askdirectory(title='Select model save directory')
            if not train_config.get('model_class'):
                train_config['model_class'] = _get_model_class_from_gui()
            if not train_config.get('training_file_dir'):
                train_config['training_file_dir'] = tkinter.filedialog.askdirectory(title='Select training files directory')
            if not train_config.get('validation_file_dir'):
                train_config['validation_file_dir'] = tkinter.filedialog.askdirectory(title='Select validation files directory')
            model_name = train_config.get('model_class')
            if model_name and model_name in models.AVAILABLE_MODELS:
                model_class = models.AVAILABLE_MODELS[model_name]
                if model_class.training_type == constants.TrainingType.SUPERVISED:
                    if not train_config.get('cheat_training_file_dir'):
                        train_config['cheat_training_file_dir'] = tkinter.filedialog.askdirectory(title='Select cheat training files')
                    if not train_config.get('cheat_validation_file_dir'):
                        train_config['cheat_validation_file_dir'] = tkinter.filedialog.askdirectory(title='Select cheat validation files')
        case constants.AppMode.TEST:
            test_config = config['test']
            if not test_config.get('save_dir'):
                test_config['save_dir'] = tkinter.filedialog.askdirectory(title='Select report save directory')
            if not test_config.get('model_file'):
                test_config['model_file'] = tkinter.filedialog.askopenfilename(title='Select model file', filetypes=[('Checkpoint Files', '*.ckpt')])
            if not test_config.get('testing_files'):
                test_config['testing_files'] = tkinter.filedialog.askdirectory(title='Select testing files directory')
        case constants.AppMode.DEPLOY:
            deploy_config = config['deploy']
            if not deploy_config.get('save_dir'):
                deploy_config['save_dir'] = tkinter.filedialog.askdirectory(title='Select data save directory')
            if not deploy_config.get('model_file'):
                deploy_config['model_file'] = tkinter.filedialog.askopenfilename(title='Select model file', filetypes=[('Checkpoint Files', '*.ckpt')])
    return config
#endregion