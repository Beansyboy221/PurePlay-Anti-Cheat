import matplotlib
import torch
import utilities, analyze, train, collect

# =============================================================================
# Main Function
# =============================================================================
def main():
    utilities.optimize_cuda_for_hardware()

    config = utilities.get_config_from_gui()
    if not config:
        return print("Configuration cancelled. Exiting.")

    if not config.get('live_graphing'):
        matplotlib.use("agg")
    if not config.get('save_dir'):
        utilities.get_save_dir_from_gui(config)

    match config.get('mode'):
        case 'collect':
            collect.collect_input_data(config)
        case 'train':
            if not config.get('model_class'):
                utilities.get_model_class_from_gui(config)
            if not config.get('training_files'):
                utilities.get_training_files_from_gui(config)
            if not config.get('validation_files'):
                utilities.get_validation_files_from_gui(config)
            if not utilities.validate_config(config):
                return print("Configuration invalid. Exiting.")
            train.train_model(config)
        case 'test':
            if not config.get('model_file'):
                utilities.get_model_file_from_gui(config)
            if not config.get('testing_files'):
                utilities.get_testing_files_from_gui(config)
            if not utilities.validate_config(config):
                return print("Configuration invalid. Exiting.")
            analyze.run_static_analysis(config)
        case 'deploy':
            if not config.get('model_file'):
                utilities.get_model_file_from_gui(config)
            if not utilities.validate_config(config):
                return print("Configuration invalid. Exiting.")
            analyze.run_live_analysis(config)
        case _:
            print(f'Invalid mode: {config.get("mode")}. Exiting...')

if __name__ == '__main__':
    main()