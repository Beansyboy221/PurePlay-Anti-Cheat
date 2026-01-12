import pydantic
import typing
import constants
import tkinter
import models
import tomllib
import tkinter.filedialog

#region Custom Validators
def _validate_model_name(model_name: str) -> typing.Any:
    if model_name in models.AVAILABLE_MODELS:
        return models.AVAILABLE_MODELS[model_name]
    raise ValueError(f"Invalid model class name: {model_name}")
#endregion

#region Pydantic Models
class SharedConfig(pydantic.BaseModel):
    """Fields pulled directly from the root of the TOML file."""
    save_dir: pydantic.DirectoryPath
    live_graphing: bool = pydantic.Field(default=True)
    kill_bind_list: typing.List[str] = pydantic.Field(default_factory=lambda: ['ESC'])
    kill_bind_logic: typing.Literal['ANY', 'ALL'] = pydantic.Field(default='ANY')

class CollectConfig(SharedConfig):
    """Fields pulled from the [collect] section of the TOML file."""
    mode: typing.Literal[constants.AppMode.COLLECT]
    polling_rate: int = pydantic.Field(default=60, validation_alias=pydantic.AliasPath('collect', 'polling_rate'))
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
        default_factory=lambda: ["deltaX", "deltaY"],
        validation_alias=pydantic.AliasPath('collect', 'mouse_whitelist')
    )
    gamepad_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('collect', 'gamepad_whitelist')
    )

class TrainConfig(SharedConfig):
    """Fields pulled from the [train] section of the TOML file."""
    mode: typing.Literal[constants.AppMode.TRAIN]
    model_class: typing.Annotated[typing.Any, pydantic.BeforeValidator(_validate_model_name)] = pydantic.Field(
        validation_alias=pydantic.AliasPath('train', 'model_class')
    )
    training_files: typing.List[pydantic.FilePath] = pydantic.Field(validation_alias=pydantic.AliasPath('train', 'training_files'))
    validation_files: typing.List[pydantic.FilePath] = pydantic.Field(validation_alias=pydantic.AliasPath('train', 'validation_files'))
    keyboard_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('train', 'keyboard_whitelist')
    )
    mouse_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: ["deltaX", "deltaY"],
        validation_alias=pydantic.AliasPath('train', 'mouse_whitelist')
    )
    gamepad_whitelist: typing.Optional[typing.List[str]] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('train', 'gamepad_whitelist')
    )
    polls_per_sequence: int = pydantic.Field(default=30, validation_alias=pydantic.AliasPath('train', 'polls_per_sequence'))
    sequences_per_batch: int = pydantic.Field(default=64, validation_alias=pydantic.AliasPath('train', 'sequences_per_batch'))

class TestConfig(SharedConfig):
    """Fields pulled from the [test] section of the TOML file."""
    mode: typing.Literal[constants.AppMode.TEST]
    model_file: pydantic.FilePath = pydantic.Field(validation_alias=pydantic.AliasPath('test', 'model_file'))
    testing_files: typing.List[pydantic.FilePath] = pydantic.Field(validation_alias=pydantic.AliasPath('test', 'testing_files'))

class DeployConfig(SharedConfig):
    """Fields pulled from the [deploy] section of the TOML file."""
    mode: typing.Literal[constants.AppMode.DEPLOY]
    model_file: pydantic.FilePath = pydantic.Field(validation_alias=pydantic.AliasPath('deploy', 'model_file'))
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
        print(f"Config validated for mode: {config.mode}")
        return config
    except Exception as e:
        print(f"Validation Error:\n{e}")
        return None

def get_config_from_gui() -> dict:
    file_path = tkinter.filedialog.askopenfilename(
        title='Select TOML configuration file', 
        filetypes=[('TOML Files', '*.toml')]
    )
    if not file_path:
        return {}
    with open(file_path, 'rb') as file:
        return tomllib.load(file)

def populate_missing_configs_from_gui(config: dict) -> dict:
    if not config.save_dir:
        config['save_dir'] = _get_save_dir_from_gui()
    mode = config.mode
    if mode == constants.AppMode.COLLECT:
        pass
    elif mode == constants.AppMode.TRAIN:
        if not config.model_class:
            config['model_class'] = _get_model_class_from_gui()
        if not config.training_files:
            config['training_files'] = _get_training_files_from_gui()
        if not config.validation_files:
            config['validation_files'] = _get_validation_files_from_gui()
    elif mode == constants.AppMode.TEST:
        if not config.model_file:
            config['model_file'] = _get_model_file_from_gui()
        if not config.testing_files:
            config['testing_files'] = _get_testing_files_from_gui()
    elif mode == constants.AppMode.DEPLOY:
        if not config.model_file:
            config['model_file'] = _get_model_file_from_gui()
    return config
#endregion

#region GUI Helper Functions
def _get_model_class_from_gui(config: dict) -> None:
    print(models.AVAILABLE_MODELS)
    root = tkinter.Tk()
    frame = tkinter.ttk.Frame(padding=5)
    frame.pack()
    tkinter.ttk.Label(frame, text='Select model class:').pack()
    combo_box = tkinter.ttk.Combobox(frame, values=list(models.AVAILABLE_MODELS.keys()))
    combo_box.pack()
    def save_and_exit() -> None:
        config['model_class'] = models.AVAILABLE_MODELS[combo_box.get()]
        root.destroy()
    tkinter.ttk.Button(frame, text='OK', command=save_and_exit).pack()
    root.mainloop()

def _get_model_file_from_gui(config: dict) -> None:
    config['model_file'] = tkinter.filedialog.askopenfilename(title='Select model file', filetypes=[('Checkpoint Files', '*.ckpt')])

def _get_training_files_from_gui(config: dict) -> None:
    config['training_files'] = tkinter.filedialog.askopenfilenames(title='Select training files', filetypes=[('HDF5 Files', '*.h5')])
    if not config.model_class:
        return
    if config.model_class.training_type == 'supervised':
        config['cheat_training_files'] = tkinter.filedialog.askopenfilenames(title='Select cheat training files', filetypes=[('HDF5 Files', '*.h5')])

def _get_validation_files_from_gui(config: dict) -> None:
    config['validation_files'] = tkinter.filedialog.askopenfilenames(title='Select validation files', filetypes=[('HDF5 Files', '*.h5')])
    if not config.model_class:
        return
    if config.model_class.training_type == 'supervised':
        config['cheat_validation_files'] = tkinter.filedialog.askopenfilenames(title='Select cheat validation files', filetypes=[('HDF5 Files', '*.h5')])

def _get_testing_files_from_gui(config: dict) -> None:
    config['testing_files'] = tkinter.filedialog.askopenfilenames(title='Select testing files', filetypes=[('HDF5 Files', '*.h5')])

def _get_save_dir_from_gui(config: dict) -> None:
    config['save_dir'] = tkinter.filedialog.askdirectory(title='Select save directory')
#endregion