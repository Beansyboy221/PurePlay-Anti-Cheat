import matplotlib
import torch
import utilities, analyze, train, collect

# =============================================================================
# Main Function
# =============================================================================
def main():
    if torch.cuda.is_available():
        processor = torch.cuda.get_device_name(torch.cuda.current_device())
        if 'RTX' in processor or 'Tesla' in processor:
            torch.set_float32_matmul_precision('medium')
            print(f'Tensor Cores detected on device: "{processor}". Using medium precision for matmul.')

    config = utilities.get_config_from_gui()
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