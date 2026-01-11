import pydantic
import typing
import constants
import tkinter
import models
import tomllib
import tkinter.filedialog

#region Pydantic Type Checkers
class BaseConfig(pydantic.BaseModel):
    mode: constants.AppMode
    save_dir: pydantic.DirectoryPath
    live_graphing: bool = True
    polling_rate: int = pydantic.Field(default=60, gt=0)
    kill_bind_list: typing.List[str] = pydantic.Field(default_factory=lambda: ['ESC'])
    kill_bind_logic: typing.Literal['ANY', 'ALL'] = 'ANY'

class CollectConfig(BaseConfig):
    mode: typing.Literal[constants.AppMode.COLLECT]
    capture_bind_list: typing.List[str] = pydantic.Field(default_factory=lambda: ['F8'])
    capture_bind_logic: typing.Literal['ANY', 'ALL'] = 'ANY'

class TrainConfig(BaseConfig):
    mode: typing.Literal[constants.AppMode.TRAIN]
    model_class: typing.Any  # Mandatory for training
    training_files: typing.List[pydantic.FilePath]
    validation_files: typing.List[pydantic.FilePath]
    polls_per_sequence: int = pydantic.Field(default=30, gt=0)
    sequences_per_batch: int = pydantic.Field(default=64, gt=1)

class TestConfig(BaseConfig):
    mode: typing.Literal[constants.AppMode.TEST]
    model_file: pydantic.FilePath
    testing_files: typing.List[pydantic.FilePath]

class DeployConfig(BaseConfig):
    mode: typing.Literal[constants.AppMode.DEPLOY]
    model_file: pydantic.FilePath
    deployment_window_type: constants.WindowType
    capture_bind_list: typing.List[str] = pydantic.Field(default_factory=lambda: ['F8'])
    capture_bind_logic: typing.Literal['ANY', 'ALL'] = 'ANY'

AppConfig = typing.Union[TrainConfig, TestConfig, DeployConfig, CollectConfig]
#endregion

#region Config Control Functions
def validate_config(config_dict: dict) -> typing.Optional[AppConfig]:
    """Attempts to parse the dict into the specific mode's model."""
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

    with open(file_path, 'rb') as f:
        return tomllib.load(f)

def populate_missing_configs_from_gui(config: dict) -> dict:
    if not config.get('save_dir'):
        config['save_dir'] = _get_save_dir_from_gui()
    mode = config.get('mode')
    if mode == constants.AppMode.COLLECT:
        pass
    elif mode == constants.AppMode.TRAIN:
        if not config.get('model_class'):
            config['model_class'] = _get_model_class_from_gui()
        if not config.get('training_files'):
            config['training_files'] = _get_training_files_from_gui()
        if not config.get('validation_files'):
            config['validation_files'] = _get_validation_files_from_gui()
    elif mode == constants.AppMode.TEST:
        if not config.get('model_file'):
            config['model_file'] = _get_model_file_from_gui()
        if not config.get('testing_files'):
            config['testing_files'] = _get_testing_files_from_gui()
    elif mode == constants.AppMode.DEPLOY:
        if not config.get('model_file'):
            config['model_file'] = _get_model_file_from_gui()
    return config
#endregion

#region Helpers
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
    if not config.get('model_class'):
        return
    if config.get('model_class').training_type == 'supervised':
        config['cheat_training_files'] = tkinter.filedialog.askopenfilenames(title='Select cheat training files', filetypes=[('HDF5 Files', '*.h5')])

def _get_validation_files_from_gui(config: dict) -> None:
    config['validation_files'] = tkinter.filedialog.askopenfilenames(title='Select validation files', filetypes=[('HDF5 Files', '*.h5')])
    if not config.get('model_class'):
        return
    if config.get('model_class').training_type == 'supervised':
        config['cheat_validation_files'] = tkinter.filedialog.askopenfilenames(title='Select cheat validation files', filetypes=[('HDF5 Files', '*.h5')])

def _get_testing_files_from_gui(config: dict) -> None:
    config['testing_files'] = tkinter.filedialog.askopenfilenames(title='Select testing files', filetypes=[('HDF5 Files', '*.h5')])

def _get_save_dir_from_gui(config: dict) -> None:
    config['save_dir'] = tkinter.filedialog.askdirectory(title='Select save directory')
#endregion