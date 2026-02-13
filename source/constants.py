import torch
import enum

#region Bind Enums
KeyBind = enum.StrEnum('KeyBind', [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', 
    '/', '.', ',', '<', '>', '?', '!', '@', '#', '$', '%', '^', '&', 
    '(', ')', '_', '=', '{', '}', '[', ']', '|', '\\', ':', ';', 
    ' ', '~', 'enter', 'esc', 'backspace', 'tab', 'space', 'caps lock', 
    'num lock', 'scroll lock', 'home', 'end', 'page up', 'page down', 
    'insert', 'delete', 'left', 'right', 'up', 'down', 'f1', 'f2', 'f3', 
    'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'print screen', 
    'pause', 'break', 'windows', 'menu', 'right alt', 'ctrl', 
    'left shift', 'right shift', 'left windows', 'left alt', 'right windows', 
    'alt gr', 'alt', 'shift', 'right ctrl', 'left ctrl'
])

MouseBind = enum.StrEnum('MouseBind', [
    'left', 'right', 'middle', 'x', 'x2', 'deltaX', 'deltaY'
])

GamepadBind = enum.StrEnum('GamepadBind', [
    'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'START', 'BACK', 
    'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
    'A', 'B', 'X', 'Y', 'LT', 'RT', 'LX', 'LY', 'RX', 'RY'
])
#endregion

#region Bind Categories
ALL_BINDS = list(KeyBind) + list(MouseBind) + list(GamepadBind)
TWO_DIMENSIONAL_BINDS = {
    MouseBind['deltaX'], 
    MouseBind['deltaY'], 
    GamepadBind['LX'], 
    GamepadBind['LY'], 
    GamepadBind['RX'], 
    GamepadBind['RY']
}
ONE_DIMENSIONAL_BINDS = [bind for bind in ALL_BINDS if bind not in TWO_DIMENSIONAL_BINDS]
#endregion

#region Tuning Constants
MAX_HIDDEN_LAYERS = 4
MAX_HIDDEN_SIZE = 256
SUPPORTED_OPTIMIZERS = [
    torch.optim.Adam,
    #torch.optim.AdamW,
    torch.optim.SGD,
    torch.optim.RMSprop,
    #torch.optim.Adagrad,
    #torch.optim.Adadelta,
    #torch.optim.Adamax,
    #torch.optim.Adafactor,
    #torch.optim.ASGD,
    #torch.optim.NAdam,
    #torch.optim.RAdam,
]
SUPPORTED_SCHEDULERS = [
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    #torch.optim.lr_scheduler.ConstantLR,
    #torch.optim.lr_scheduler.LinearLR,
    #torch.optim.lr_scheduler.PolynomialLR,
]
OPTIMIZER_MAP = {optimizer.__name__: optimizer for optimizer in SUPPORTED_OPTIMIZERS}
SCHEDULER_MAP = {scheduler.__name__: scheduler for scheduler in SUPPORTED_SCHEDULERS}
#endregion

#region Config Enums
class AppMode(enum.StrEnum):
    COLLECT = "collect"
    TRAIN = "train"
    TEST = "test"
    DEPLOY = "deploy"

class InputGate(enum.StrEnum):
    ANY = "ANY"
    ALL = "ALL"

class WindowType(enum.StrEnum):
    SLIDING = "sliding"
    TUMBLING = "tumbling"

class TrainingType(enum.StrEnum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
#endregion