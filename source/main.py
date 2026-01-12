import matplotlib
import constants, config, devices
import source.modes.collect as collect
import source.modes.train as train
import source.modes.test as test
import source.modes.deploy as deploy

def main():
    devices.optimize_cuda_for_hardware()
    app_config: dict = config.get_config_from_gui()
    app_config: dict = config.populate_missing_configs_from_gui(app_config)
    app_config: config.AppConfig = config.validate_config(app_config)
    if not app_config:
        print("Failed to build a valid configuration. Exiting.")
        return
    
    # Global Configs
    if app_config.live_graphing == False:
        matplotlib.use("agg")
    
    match app_config.mode:
        case constants.AppMode.COLLECT:
            collect.collect_input_data(app_config)
        case constants.AppMode.TRAIN:
            train.train_model(app_config)
        case constants.AppMode.TEST:
            test.run_static_analysis(app_config)
        case constants.AppMode.DEPLOY:
            deploy.run_live_analysis(app_config)
        case _:
            print(f"Unsupported mode: {app_config.mode}")

if __name__ == '__main__':
    main()