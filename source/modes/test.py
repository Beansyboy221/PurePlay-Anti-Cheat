import torch.utils.data
import lightning
import torch
import os
import preprocessing, devices

def run_static_analysis(config: object) -> None:
    """Performs static analysis on selected data files using a pre-trained model."""
    model = config.model_class.load_from_checkpoint(config.model_file)

    testing_files = [os.path.join(config.testing_file_dir, file_path) for file_path in os.listdir(config.testing_file_dir)]
    for file_path in testing_files:
        test_dataset = preprocessing.InputDataset(
            file_path,
            model.hparams.data_params.get('polls_per_sequence'),
            model.hparams.data_params.get('whitelist'),
            model.hparams.data_params.get('ignore_empty_polls')
        )
        
        if test_dataset.polling_rate != model.hparams.data_params.get('polling_rate'):
            raise ValueError(f'Model expects polling rate: {model.hparams.data_params.get("polling_rate")}. File: "{file_path}" has polling rate: {test_dataset.polling_rate}.')

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=devices.CPU_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )
        trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
        trainer.test(model, dataloaders=test_loader, ckpt_path=None)