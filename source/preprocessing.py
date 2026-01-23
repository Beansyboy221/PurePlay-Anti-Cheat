import pyarrow.parquet
import lightning
import polars
import torch
import numpy
import os
import constants, devices
    
class PolarsStandardScaler:
    def __init__(self):
        self.means = None
        self.stds = None
        self.columns = []

    def fit(self, file_paths, scalable_columns):
        if not scalable_columns: return
        
        self.columns = scalable_columns
        stats = (
            polars.scan_parquet(file_paths).select([
                *[polars.col(c).mean().alias(f"{c}_mean") for c in scalable_columns],
                *[polars.col(c).std().alias(f"{c}_std") for c in scalable_columns]
            ]).collect(streaming=True)
        )

        self.means = numpy.array([stats[f"{c}_mean"][0] for c in scalable_columns], dtype=numpy.float32)
        self.stds = numpy.array([stats[f"{c}_std"][0] for c in scalable_columns], dtype=numpy.float32)
        self.stds[self.stds == 0] = 1.0

class InputDataset(torch.utils.data.Dataset):
    """Dataset for input sequences stored in Parquet files."""
    def __init__(self, file_path: str, polls_per_sequence: int, whitelist: list[str], ignore_empty_polls: bool = True, label: int = 0):
        self.polls_per_sequence = polls_per_sequence
        self.polling_rate = pyarrow.parquet.read_metadata(file_path).metadata.get(b"polling_rate", b"1000").decode("utf-8")
        self.label = label
        lazy_frame = polars.scan_parquet(file_path)
        
        # Filter out empty polls if needed (should I filter out whole sequences to preserve temporal structure?)
        if ignore_empty_polls:
            lazy_frame = lazy_frame.filter(polars.sum_horizontal(whitelist) != 0)
        
        data_frame = lazy_frame.select(whitelist).collect()
        data_array = data_frame.to_numpy(writable=True).astype(numpy.float32)
        
        # Trim excess rows to make full sequences
        num_sequences = len(data_array) // self.polls_per_sequence
        total_rows = num_sequences * self.polls_per_sequence
        data_array = data_array[:total_rows].reshape(num_sequences, self.polls_per_sequence, -1)
        
        self.data_tensor = torch.from_numpy(data_array)

    def __len__(self):
        """Returns the number of sequences in the dataset."""
        return len(self.data_tensor)

    def __getitem__(self, index):
        """Returns a sequence and its label."""
        return self.data_tensor[index], torch.tensor(self.label, dtype=torch.float32)
    
class TrainingDataModule(lightning.LightningDataModule):
    def __init__(self, config: object):
        super().__init__()
        self.config = config
        self.whitelist = config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist
        self.is_supervised = config.model_class.training_type == constants.TrainingType.SUPERVISED
        self.scaler = PolarsStandardScaler()
        
        self.train_dataset = None
        self.val_dataset = None
        self.polling_rate = None

    def setup(self, stage: str = None):
        """Creates and sets up training and validation datasets."""
        training_files = [os.path.join(self.config.training_file_dir, file_path) for file_path in os.listdir(self.config.training_file_dir)]
        validation_files = [os.path.join(self.config.validation_file_dir, file_path) for file_path in os.listdir(self.config.validation_file_dir)]
        
        cheat_training_files = []
        cheat_validation_files = []
        if self.is_supervised:
            cheat_training_files = [os.path.join(self.config.cheat_training_file_dir, file_path) for file_path in os.listdir(self.config.cheat_training_file_dir)]
            cheat_validation_files = [os.path.join(self.config.cheat_validation_file_dir, file_path) for file_path in os.listdir(self.config.cheat_validation_file_dir)]
        
        all_fit_files = training_files + (cheat_training_files if self.is_supervised else [])
        self.scaler.fit(all_fit_files, self.whitelist)

        def make_datasets(file_paths: list, label: int = 0):
            return [InputDataset(file_path, self.config.polls_per_sequence, self.whitelist, self.config.ignore_empty_polls, label=label) for file_path in file_paths]

        train_datasets = make_datasets(training_files)
        val_datasets = make_datasets(validation_files)
        if self.is_supervised:
            train_datasets += make_datasets(cheat_training_files, label=1)
            val_datasets += make_datasets(cheat_validation_files, label=1)

        all_datasets = train_datasets + val_datasets
        self.polling_rate = all_datasets[0].polling_rate
        if not all(dataset.polling_rate == self.polling_rate for dataset in all_datasets):
            raise ValueError("Inconsistent polling rates across files.")
        
        self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            shuffle=True, 
            batch_size=self.config.sequences_per_batch,
            num_workers=devices.CPU_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            shuffle=False, 
            batch_size=self.config.sequences_per_batch,
            num_workers=devices.CPU_WORKERS,
            pin_memory=True,
            persistent_workers=True
        )