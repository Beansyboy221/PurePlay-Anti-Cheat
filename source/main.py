import matplotlib
import torch
import utilities, analyze, train, collect

# =============================================================================
# Main Function
# =============================================================================
def main():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        processor = torch.cuda.get_device_name(device)
        has_tensor_cores = (major >= 7)

        if has_tensor_cores:
            torch.set_float32_matmul_precision('medium')
            print(f"Tensor Cores detected cc {major}.{minor}) on device '{processor}'. Using medium precision for matmul.")
        else:
            print(f"No Tensor Cores detected (cc {major}.{minor}) on device '{processor}'.")

    config = utilities.get_config_from_gui()
    if not config:
        return print("Configuration cancelled. Exiting.")

    if not config['live_graphing']:
        matplotlib.use("agg")

    match config['mode']:
        case 'collect':
            collect.collect_input_data(config)
        case 'train':
            if not config['model_class']:
                utilities.get_model_class_from_gui(config)
            if not config['training_files']:
                utilities.get_training_files_from_gui(config)
            if not config['validation_files']:
                utilities.get_validation_files_from_gui(config)
            if not utilities.validate_config(config):
                return print("Configuration invalid. Exiting.")
            train.train_model(config)
        case 'test':
            if not config['model_file']:
                utilities.get_model_file_from_gui(config)
            if not config['testing_files']:
                utilities.get_testing_files_from_gui(config)
            if not utilities.validate_config(config):
                return print("Configuration invalid. Exiting.")
            analyze.run_static_analysis(config)
        case 'deploy':
            if not config['model_file']:
                utilities.get_model_file_from_gui(config)
            if not utilities.validate_config(config):
                return print("Configuration invalid. Exiting.")
            analyze.run_live_analysis(config)

if __name__ == '__main__':
    main()