import lightning.pytorch.callbacks
import tkinter.ttk
import matplotlib
import lightning
import analyze
import collect
import tkinter
import models
import torch
import train

# =============================================================================
# Configuration GUI
# =============================================================================
def _get_config_from_gui() -> dict | None:
    config = {}
    available_models = models.get_available_models()

    def on_submit():
        nonlocal config
        config['mode'] = mode_var.get()
        config['kill_key'] = kill_key_var.get()
        config['capture_bind'] = capture_bind_var.get()
        config['keyboard_whitelist'] = [item.strip() for item in keyboard_whitelist_var.get().split(',') if item.strip()]
        config['mouse_whitelist'] = [item.strip() for item in mouse_whitelist_var.get().split(',') if item.strip()]
        config['gamepad_whitelist'] = [item.strip() for item in gamepad_whitelist_var.get().split(',') if item.strip()]
        config['model_class'] = available_models.get(model_var.get())
        config['live_graphing'] = live_graphing_var.get()

        try:
            config['polling_rate'] = int(polling_rate_var.get())
            config['sequence_length'] = int(sequence_length_var.get())
            config['batch_size'] = int(batch_size_var.get())
        except ValueError:
            print("Polling rate, sequence length, and batch size must be integers.")
            config = {} # Invalidate config
            win.destroy()
            return

        win.destroy()

    def on_mode_change(*args):
        if mode_var.get() in ['train', 'test', 'deploy']:
            model_menu.config(state='normal')
        else:
            model_menu.config(state='disabled')
            if available_models:
                model_var.set(list(available_models.keys())[0])

    win = tkinter.Toplevel()
    win.title("Configuration")

    # --- Variables ---
    mode_var = tkinter.StringVar(value='collect')
    kill_key_var = tkinter.StringVar(value='\\')
    capture_bind_var = tkinter.StringVar(value='RT')
    keyboard_whitelist_var = tkinter.StringVar(value='')
    mouse_whitelist_var = tkinter.StringVar(value='deltaX, deltaY')
    gamepad_whitelist_var = tkinter.StringVar(value='LT, LX, LY, RX, RY')
    polling_rate_var = tkinter.StringVar(value='60')
    sequence_length_var = tkinter.StringVar(value='30')
    batch_size_var = tkinter.StringVar(value='16')
    model_var = tkinter.StringVar(value=list(available_models.keys())[0] if available_models else '')
    live_graphing_var = tkinter.BooleanVar(value=False)

    # --- Widgets ---
    frame = tkinter.ttk.Frame(win, padding="10")
    frame.grid(row=0, column=0, sticky=(tkinter.W, tkinter.E, tkinter.N, tkinter.S))

    # Mode
    tkinter.ttk.Label(frame, text="Mode:").grid(column=0, row=0, sticky=tkinter.W)
    mode_menu = tkinter.ttk.OptionMenu(frame, variable=mode_var, default='collect', values=('collect', 'train', 'analyze'))
    mode_menu.grid(column=1, row=0, sticky=(tkinter.W, tkinter.E))
    mode_var.trace_add('write', on_mode_change)

    # Model
    tkinter.ttk.Label(frame, text="Model:").grid(column=0, row=1, sticky=tkinter.W)
    model_menu = tkinter.ttk.OptionMenu(frame, variable=model_var, default=model_var.get(), values=list(available_models.keys()))
    model_menu.config(*available_models.keys())
    model_menu.grid(column=1, row=1, sticky=(tkinter.W, tkinter.E))

    # Live Graphing
    tkinter.ttk.Label(frame, text="Live Graphing:").grid(column=0, row=2, sticky=tkinter.W)
    live_graphing_check = tkinter.ttk.Checkbutton(frame, variable=live_graphing_var)
    live_graphing_check.grid(column=1, row=2, sticky=tkinter.W)

    # Kill Key
    tkinter.ttk.Label(frame, text="Kill Key:").grid(column=0, row=1, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=kill_key_var).grid(column=1, row=1, sticky=(tkinter.W, tkinter.E))

    # Capture Bind
    tkinter.ttk.Label(frame, text="Capture Bind:").grid(column=0, row=2, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=capture_bind_var).grid(column=1, row=2, sticky=(tkinter.W, tkinter.E))

    # Whitelists (comma-separated)
    tkinter.ttk.Label(frame, text="Keyboard Whitelist (csv):").grid(column=0, row=3, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=keyboard_whitelist_var).grid(column=1, row=3, sticky=(tkinter.W, tkinter.E))

    tkinter.ttk.Label(frame, text="Mouse Whitelist (csv):").grid(column=0, row=4, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=mouse_whitelist_var).grid(column=1, row=4, sticky=(tkinter.W, tkinter.E))

    tkinter.ttk.Label(frame, text="Gamepad Whitelist (csv):").grid(column=0, row=5, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=gamepad_whitelist_var).grid(column=1, row=5, sticky=(tkinter.W, tkinter.E))

    # Numerical settings
    tkinter.ttk.Label(frame, text="Polling Rate (Hz):").grid(column=0, row=6, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=polling_rate_var).grid(column=1, row=6, sticky=(tkinter.W, tkinter.E))

    tkinter.ttk.Label(frame, text="Sequence Length:").grid(column=0, row=7, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=sequence_length_var).grid(column=1, row=7, sticky=(tkinter.W, tkinter.E))

    tkinter.ttk.Label(frame, text="Batch Size:").grid(column=0, row=8, sticky=tkinter.W)
    tkinter.ttk.Entry(frame, textvariable=batch_size_var).grid(column=1, row=8, sticky=(tkinter.W, tkinter.E))

    # Submit Button
    submit_button = tkinter.ttk.Button(frame, text="Submit", command=on_submit)
    submit_button.grid(column=1, row=10, sticky=tkinter.E, pady=10)

    frame.columnconfigure(1, weight=1)

    on_mode_change()

    win.wait_window()

    return config if config else None

# =============================================================================
# Main Function
# =============================================================================
def main():
    if torch.cuda.is_available():
        processor = torch.cuda.get_device_name(torch.cuda.current_device())
        if 'RTX' in processor or 'Tesla' in processor:
            torch.set_float32_matmul_precision('medium')
            print(f'Tensor Cores detected on device: "{processor}". Using medium precision for matmul.')

    config = _get_config_from_gui()
    if not config:
        return print("Configuration cancelled. Exiting.")

    if not config['live_graphing']:
        matplotlib.use("agg")

    match config['mode']:
        case 'collect':
            collect.collect_input_data(config)
        case 'train':
            train.train_model(config)
        case 'test':
            analyze.run_static_analysis(config)
        case 'deploy':
            analyze.run_live_analysis(config)

if __name__ == '__main__':
    main()