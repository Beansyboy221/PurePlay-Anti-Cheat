import matplotlib
import constants, config, utilities
import source.modes.collect as collect
import source.modes.train as train
import source.modes.test as test
import source.modes.deploy as deploy

def main():
    utilities.optimize_cuda_for_hardware()
    user_config: dict = config.get_config_from_gui()
    user_config: dict = config.populate_missing_configs_from_gui(user_config)
    user_config: config.AppConfig = config.validate_config(user_config)
    if not user_config:
        print("Failed to build a valid configuration. Exiting.")
        return
    
    # Global Configs
    if user_config.live_graphing == False:
        matplotlib.use("agg")
    
    match user_config.mode:
        case constants.AppMode.COLLECT:
            collect.collect_input_data(user_config)
        case constants.AppMode.TRAIN:
            train.train_model(user_config)
        case constants.AppMode.TEST:
            test.run_static_analysis(user_config)
        case constants.AppMode.DEPLOY:
            deploy.run_live_analysis(user_config)
        case _:
            print(f"Unsupported mode: {user_config.mode}")

if __name__ == '__main__':
    main()