import constants, config, devices
import modes.collect
import modes.train
import modes.test
import modes.deploy
import logging

def main():
    app_config: dict = config.get_config_from_gui()
    app_config: dict = config.populate_missing_configs_from_gui(app_config)
    app_config: config.AppConfig = config.validate_config(app_config)
    if not app_config:
        print("Failed to build a valid configuration. Exiting.")
        return
    
    # Global Configs
    if app_config.mode != constants.AppMode.COLLECT:
        devices.optimize_cuda_for_hardware()
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    
    match app_config.mode:
        case constants.AppMode.COLLECT:
            modes.collect.collect_input_data(app_config)
        case constants.AppMode.TRAIN:
            modes.train.train_model(app_config)
        case constants.AppMode.TEST:
            modes.test.run_static_analysis(app_config)
        case constants.AppMode.DEPLOY:
            modes.deploy.run_live_analysis(app_config)
        case _:
            print(f"Unsupported mode: {app_config.mode}")

if __name__ == '__main__':
    print('Starting PurePlay Anti-Cheat')
    main()