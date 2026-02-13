import lightning
import preprocessing

def run_static_analysis(config: object) -> None:
    """Performs static analysis on selected data files using a pre-trained model."""
    # Load model class from file rather than config
    # Add option to load from torchscript, checkpoint, or onnx
    model = config.model_class.load_from_checkpoint(config.model_file)

    data_module = preprocessing.TestingDataModule(config, model)
    data_module.setup()

    trainer = lightning.Trainer(
        logger=False,
        enable_checkpointing=False
    )
    trainer.test(model, datamodule=data_module)
