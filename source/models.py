import matplotlib.pyplot
import lightning
import torch
import time

# =============================================================================
# Models
# =============================================================================
class LSTMAutoencoder(lightning.LightningModule):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length
        self.learning_rate = hyperparams['learning_rate']

        self.encoder = torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            batch_first=True,
            dropout=hyperparams['dropout']
        )
        self.decoder = torch.nn.LSTM(
            input_size=hyperparams['hidden_dim'],
            hidden_size=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            batch_first=True,
            dropout=hyperparams['dropout']
        )
        self.output_layer = torch.nn.Linear(hyperparams['hidden_dim'], num_features)

        self.loss_function = torch.nn.MSELoss()
        self.train_metric_history = []
        self.val_metric_history = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.test_metric_history = []
        self.epoch_indices = []
        self.epoch_counter = 0

        self.save_dir = save_dir
        self.trial_number = trial_number
        self.figure, self.axes = matplotlib.pyplot.subplots()

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        output = input_sequence
        output, (hidden_state, cell_state) = self.encoder(output)
        output, (hidden_state, cell_state) = self.decoder(output)
        reconstruction = self.output_layer(output)
        return reconstruction

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = self.loss_function(reconstruction, inputs)
        self.train_metric_history.append(reconstruction_error.detach().cpu())
        return reconstruction_error

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = self.loss_function(reconstruction, inputs)
        self.val_metric_history.append(reconstruction_error.detach().cpu())
        self.log('val_loss', reconstruction_error, prog_bar=True, on_epoch=True)
        return reconstruction_error
    
    def on_validation_epoch_end(self) -> None:
        avg_train_loss = torch.stack(self.train_metric_history).mean().item() if self.train_metric_history else None
        avg_val_loss = torch.stack(self.val_metric_history).mean().item() if self.val_metric_history else None
        if avg_train_loss is None or avg_val_loss is None:
            return
        self.epoch_indices.append(self.epoch_counter)
        self.epoch_counter += 1
        self.avg_train_losses.append(avg_train_loss)
        self.avg_val_losses.append(avg_val_loss)
        self.train_metric_history = []
        self.val_metric_history = []

    def on_fit_end(self) -> None:
        self.axes.clear()
        self.axes.plot(self.epoch_indices, self.avg_train_losses, label='Train Loss')
        self.axes.plot(self.epoch_indices, self.avg_val_losses, label='Val Loss')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('MSE')
        self.axes.legend()
        self.figure.savefig(f'{self.save_dir}/trial{self.trial_number}_unsupervised_{time.strftime('%Y%m%d-%H%M%S')}.png')
        matplotlib.pyplot.close(self.figure)
        return super().on_fit_end()
        
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inputs, labels = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = torch.sqrt(self.loss_function(reconstruction, inputs))
        self.test_metric_history.append(reconstruction_error.detach().cpu())
    
    def on_test_end(self) -> None:
        self.axes.clear()
        self.axes.plot(list(range(len(self.test_metric_history))), self.test_metric_history)
        self.axes.set_xlabel('Sequence')
        self.axes.set_ylabel('Reconstruction Error (MSE)')
        self.axes.set_title(f'Average Error: {torch.stack(self.test_metric_history).mean()}')
        self.figure.savefig(f'{self.save_dir}/report_unsupervised_{time.strftime('%Y%m%d-%H%M%S')}.png')
        return super().on_test_end()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class LSTMBinaryClassifier(lightning.LightningModule):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length
        self.learning_rate = hyperparams['learning_rate']

        self.lstm= torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            batch_first=True,
            dropout=hyperparams['dropout']
        )
        self.classifier_layer = torch.nn.Linear(hyperparams['hidden_dim'], 1)

        self.loss_function = torch.nn.BCEWithLogitsLoss()

        self.train_metric_history = []
        self.val_metric_history = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.test_metric_history = []
        self.epoch_indices = []
        self.epoch_counter = 0
        
        self.save_dir = save_dir
        self.trial_number = trial_number
        self.figure, self.axes = matplotlib.pyplot.subplots()

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        output = input_sequence
        output, (hidden_state, cell_state) = self.lstm(output)
        last_output = output[:, -1, :]
        class_prediction = self.classifier_layer(last_output).squeeze(1)
        return class_prediction
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        class_prediction = self.forward(inputs)
        loss = self.loss_function(class_prediction, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self) -> None:
        avg_train_loss = torch.stack(self.train_metric_history).mean().item() if self.train_metric_history else None
        avg_val_loss = torch.stack(self.val_metric_history).mean().item() if self.val_metric_history else None
        if avg_train_loss is None or avg_val_loss is None:
            return
        self.epoch_indices.append(self.epoch_counter)
        self.epoch_counter += 1
        self.avg_train_losses.append(avg_train_loss)
        self.avg_val_losses.append(avg_val_loss)
        self.train_metric_history = []
        self.val_metric_history = []

    def on_fit_end(self) -> None:
        self.axes.clear()
        self.axes.plot(self.epoch_indices, self.avg_train_losses, label='Train Loss')
        self.axes.plot(self.epoch_indices, self.avg_val_losses, label='Val Loss')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('BCELoss')
        self.axes.legend()
        self.figure.savefig(f'{self.save_dir}/trial{self.trial_number}_supervised_{time.strftime('%Y%m%d-%H%M%S')}.png')
        matplotlib.pyplot.close(self.figure)
        return super().on_fit_end()

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inputs, labels = batch
        logits = self.forward(inputs)
        confidence = torch.sigmoid(logits)
        self.test_metric_history.append(confidence.detach().cpu())

    def on_test_end(self) -> None:
        self.axes.clear()
        self.axes.plot(list(range(len(self.test_metric_history))), self.test_metric_history)
        self.axes.set_xlabel('Sequence')
        self.axes.set_ylabel('Confidence')
        self.axes.set_ylim(0, 1)
        self.axes.yaxis.get_major_formatter().set_useOffset(False)
        self.axes.set_title(f'Average Confidence: {torch.stack(self.test_metric_history).mean()}')
        self.figure.savefig(f'{self.save_dir}/report_supervised_{time.strftime('%Y%m%d-%H%M%S')}.png')
        return super().on_test_end()
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class FeedforwardAutoencoder(lightning.LightningModule):
    pass

class FeedforwardBinaryClassifier(lightning.LightningModule):
    pass

class OneDimensionalCNNAutoencoder(lightning.LightningModule):
    pass

class OneDimensionalCNNBinaryClassifier(lightning.LightningModule):
    pass