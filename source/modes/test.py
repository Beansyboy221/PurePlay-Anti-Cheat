import torch.utils.data
import lightning
import torch
import os
import utilities, devices

def run_static_analysis(config: object) -> None:
    """Performs static analysis on selected data files using a pre-trained model."""
    #Switch to only use exported onnx models
    model = config.model_class.load_from_checkpoint(config.model_file)
    model.save_dir = config.save_dir

    polling_rate = None
    testing_files = [os.path.join(config.testing_file_dir, file_path) for file_path in os.listdir(config.testing_file_dir)]
    for file_path in testing_files:
        test_dataset = utilities.InputDataset(
            file_path,
            model.hparams.data_params.get('polls_per_sequence'),
            model.hparams.data_params.get('whitelist'),
            model.hparams.data_params.get('ignore_empty_polls')
        )

        if polling_rate is None:
            polling_rate = test_dataset.polling_rate
        else:
            assert polling_rate == model.hparams.data_params.get('polling_rate'), "Inconsistent polling rates between model and test files."
            assert polling_rate == test_dataset.polling_rate, "Inconsistent polling rates across test files."
            
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
