import torch.utils.data
import lightning
import utilities

def run_static_analysis(config: object) -> None:
    """Performs static analysis on selected data files using a pre-trained model."""
    model = config.model_class.load_from_checkpoint(config.model_file)
    model.save_dir = config.save_dir

    polling_rate = None
    for file_path in config.testing_files:
        test_dataset = utilities.InputDataset(
            file_path,
            model.hparams.polls_per_sequence,
            model.hparams.whitelist,
            ignore_empty_polls=True
        )

        if polling_rate is None:
            polling_rate = test_dataset.polling_rate
        else:
            assert polling_rate == test_dataset.polling_rate, "Inconsistent polling rates across test files."
            
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
        trainer.test(model, dataloaders=test_loader, ckpt_path=None)
