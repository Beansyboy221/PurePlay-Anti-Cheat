import logging
import constants, config, devices

def main():
    config_dict: dict = config.get_config_from_gui() # Pick config file with GUI
    config_dict: dict = config.populate_missing_configs_from_gui(config_dict)
    app_config: config.AppConfig = config.validate_config(config_dict)
    if not app_config:
        print("Failed to build a valid configuration. Exiting.")
        return
    
    # Common Configs
    if app_config.mode != constants.AppMode.COLLECT:
        devices.optimize_cuda_for_hardware()
        logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    
    match app_config.mode:
        case constants.AppMode.COLLECT:
            import modes.collect
            modes.collect.collect_input_data(app_config)
        case constants.AppMode.TRAIN:
            import modes.train
            modes.train.train_model(app_config)
        case constants.AppMode.TEST:
            import modes.test
            modes.test.run_static_analysis(app_config)
        case constants.AppMode.DEPLOY:
            import modes.deploy
            modes.deploy.run_live_analysis(app_config)
        case _:
            print(f'Unsupported mode: {app_config.mode}')

if __name__ == '__main__':
    main()