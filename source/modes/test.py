import torch.utils.data
import lightning
import utilities

def run_static_analysis(config: dict) -> None:
    """Performs static analysis on selected data files using a pre-trained model."""
    whitelist = config.get('keyboard_whitelist') + config.get('mouse_whitelist') + config.get('gamepad_whitelist')

    model = config.get('model_class').load_from_checkpoint(config.get('model_file'))
    config['polls_per_sequence'] = model.hparams.polls_per_sequence
    model.save_dir = config.get('save_dir')

    utilities.fit_global_scaler(config.get('testing_files'), whitelist)
    for file in config.get('testing_files'):
        test_dataset = utilities.InputDataset(file, config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
        trainer.test(model, dataloaders=test_loader, ckpt_path=None)
