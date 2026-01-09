from pydantic import BaseModel, FilePath, DirectoryPath, Field, ConfigDict, TypeAdapter
from typing import List, Optional, Union, Literal, Any
import constants
import tkinter
import models
import json, yaml, tomllib, configparser
import tkinter.filedialog
import pathlib

# region Pydantic Type Checkers
class BaseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) # Allows torch/custom classes
    mode: constants.AppMode
    save_dir: DirectoryPath
    live_graphing: bool = True
    polling_rate: int = Field(default=60, gt=0)

class CollectConfig(BaseConfig):
    mode: Literal[constants.AppMode.COLLECT]
    # Add collection-specific fields here if needed

class TrainConfig(BaseConfig):
    mode: Literal[constants.AppMode.TRAIN]
    model_class: Any  # Mandatory for training
    training_files: List[FilePath]
    validation_files: List[FilePath]
    polls_per_sequence: int = Field(default=30, gt=0)
    sequences_per_batch: int = Field(default=64, gt=1)

class TestConfig(BaseConfig):
    mode: Literal[constants.AppMode.TEST]
    model_file: FilePath
    testing_files: List[FilePath]

class DeployConfig(BaseConfig):
    mode: Literal[constants.AppMode.DEPLOY]
    model_file: FilePath
    deployment_window_type: constants.WindowType

# The Discriminated Union
AppConfig = Union[TrainConfig, TestConfig, DeployConfig, CollectConfig]
# endregion

# region Config Control Functions
def validate_config(config_dict: dict) -> Optional[AppConfig]:
    """Attempts to parse the dict into the specific mode's model."""
    try:
        config = TypeAdapter(AppConfig).validate_python(config_dict)
        print(f"Config validated for mode: {config.mode}")
        return config
    except Exception as e:
        print(f"Validation Error:\n{e}")
        return None

def get_config_from_gui() -> dict:
    file_path = tkinter.filedialog.askopenfilename(
        title='Select configuration file', 
        filetypes=[
            ('All Supported', '*.json *.yaml *.yml *.toml *.ini *.cfg'),
            ('JSON Files', '*.json'),
            ('YAML Files', '*.yaml *.yml'),
            ('TOML Files', '*.toml'),
            ('INI Files', '*.ini *.cfg')
        ]
    )
    
    if not file_path:
        return {}

    path = pathlib.Path(file_path)
    ext = path.suffix.lower()

    if ext in ['.ini', '.cfg']:
        config = configparser.ConfigParser()
        config.read(file_path)
        return {section: dict(config.items(section)) for section in config.sections()}

    with open(file_path, 'rb') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext == '.toml':
            return tomllib.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

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
# endregion

# region Helpers
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
    config['training_files'] = tkinter.filedialog.askopenfilenames(title='Select training files', filetypes=[('CSV Files', '*.csv')])
    if not config.get('model_class'):
        return
    if config.get('model_class').training_type == 'supervised':
        config['cheat_training_files'] = tkinter.filedialog.askopenfilenames(title='Select cheat training files', filetypes=[('CSV Files', '*.csv')])

def _get_validation_files_from_gui(config: dict) -> None:
    config['validation_files'] = tkinter.filedialog.askopenfilenames(title='Select validation files', filetypes=[('CSV Files', '*.csv')])
    if not config.get('model_class'):
        return
    if config.get('model_class').training_type == 'supervised':
        config['cheat_validation_files'] = tkinter.filedialog.askopenfilenames(title='Select cheat validation files', filetypes=[('CSV Files', '*.csv')])

def _get_testing_files_from_gui(config: dict) -> None:
    config['testing_files'] = tkinter.filedialog.askopenfilenames(title='Select testing files', filetypes=[('CSV Files', '*.csv')])

def _get_save_dir_from_gui(config: dict) -> None:
    config['save_dir'] = tkinter.filedialog.askdirectory(title='Select save directory')
# endregion