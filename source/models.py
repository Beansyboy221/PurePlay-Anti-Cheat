from abc import ABC, abstractmethod
import matplotlib.pyplot
import lightning
import torch
import time

# =============================================================================
# Model Registry
# =============================================================================
AVAILABLE_MODELS = {}

def register_model(model: type[lightning.LightningModule]) -> type[lightning.LightningModule]:
    """Decorator to register a model class."""
    AVAILABLE_MODELS[model.__name__] = model
    return model

def get_available_models() -> dict[str, type[lightning.LightningModule]]:
    """Returns a dictionary of available model classes."""
    return AVAILABLE_MODELS

# =============================================================================
# Base Models
# =============================================================================
class BaseModel(lightning.LightningModule, ABC):
    """Base model for handling shared training, validation, and plotting logic."""
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__()
        self.save_hyperparameters()
        self.sequence_length = sequence_length
        self.learning_rate = hyperparams['learning_rate']
        self.num_features = num_features

        self.train_metric_history = []
        self.val_metric_history = []
        self.avg_train_losses = []
        self.avg_val_losses = []
        self.test_metric_history = []
        self.epoch_indices = []
        self.epoch_counter = 0

        self.save_dir = save_dir
        self.trial_number = trial_number

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @abstractmethod
    def on_fit_end(self) -> None:
        super().on_fit_end()

    @abstractmethod
    def on_test_end(self) -> None:
        super().on_test_end()

class AutoencoderBase(BaseModel):
    """Base class for all autoencoder models."""
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.training_type = 'unsupervised'
        self.loss_function = torch.nn.MSELoss()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, _ = batch
        reconstruction = self.forward(inputs)
        loss = self.loss_function(reconstruction, inputs)
        self.train_metric_history.append(loss.detach().cpu())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, _ = batch
        reconstruction = self.forward(inputs)
        loss = self.loss_function(reconstruction, inputs)
        self.val_metric_history.append(loss.detach().cpu())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inputs, _ = batch
        reconstruction = self.forward(inputs)
        reconstruction_error = torch.sqrt(self.loss_function(reconstruction, inputs))
        self.test_metric_history.append(reconstruction_error.detach().cpu())

    def on_fit_end(self) -> None:
        figure, axes = matplotlib.pyplot.subplots()
        axes.plot(self.epoch_indices, self.avg_train_losses, label='Train Loss')
        axes.plot(self.epoch_indices, self.avg_val_losses, label='Val Loss')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('MSE')
        axes.legend()
        figure.savefig(f'{self.save_dir}/trial{self.trial_number}_{self.training_type}_{time.strftime("%Y%m%d-%H%M%S")}.png')
        matplotlib.pyplot.close(figure)
        return super().on_fit_end()

    def on_test_end(self) -> None:
        figure, axes = matplotlib.pyplot.subplots()
        axes.plot(list(range(len(self.test_metric_history))), self.test_metric_history)
        axes.set_xlabel('Sequence')
        axes.set_ylabel('Reconstruction Error (RMSE)')
        if self.test_metric_history:
            axes.set_title(f'Average Error: {torch.stack(self.test_metric_history).mean()}')
        figure.savefig(f'{self.save_dir}/report_{self.training_type}_{time.strftime("%Y%m%d-%H%M%S")}.png')
        matplotlib.pyplot.close(figure)
        return super().on_test_end()

class ClassifierBase(BaseModel):
    """Base class for all binary classifier models."""
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.training_type = 'supervised'
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_function(logits, labels)
        self.train_metric_history.append(loss.detach().cpu())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        logits = self.forward(inputs)
        loss = self.loss_function(logits, labels)
        self.val_metric_history.append(loss.detach().cpu())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inputs, _ = batch
        logits = self.forward(inputs)
        confidence = torch.sigmoid(logits)
        self.test_metric_history.append(confidence.detach().cpu())

    def on_fit_end(self) -> None:
        figure, axes = matplotlib.pyplot.subplots()
        axes.plot(self.epoch_indices, self.avg_train_losses, label='Train Loss')
        axes.plot(self.epoch_indices, self.avg_val_losses, label='Val Loss')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('BCELoss')
        axes.legend()
        figure.savefig(f'{self.save_dir}/trial{self.trial_number}_{self.training_type}_{time.strftime("%Y%m%d-%H%M%S")}.png')
        matplotlib.pyplot.close(figure)
        return super().on_fit_end()

    def on_test_end(self) -> None:
        figure, axes = matplotlib.pyplot.subplots()
        axes.plot(list(range(len(self.test_metric_history))), self.test_metric_history)
        axes.set_xlabel('Sequence')
        axes.set_ylabel('Confidence')
        axes.set_ylim(0, 1)
        axes.yaxis.get_major_formatter().set_useOffset(False)
        if self.test_metric_history:
            axes.set_title(f'Average Confidence: {torch.stack(self.test_metric_history).mean()}')
        figure.savefig(f'{self.save_dir}/report_{self.training_type}_{time.strftime("%Y%m%d-%H%M%S")}.png')
        matplotlib.pyplot.close(figure)
        return super().on_test_end()

# =============================================================================
# Feedforward Models
# =============================================================================
@register_model
class FeedforwardAutoencoder(AutoencoderBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        input_dim = num_features * sequence_length
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hyperparams['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(hyperparams['dropout'])
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hyperparams['hidden_dim'], input_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, self.sequence_length, self.num_features)

@register_model
class FeedforwardBinaryClassifier(ClassifierBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        input_dim = num_features * sequence_length
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hyperparams['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(hyperparams['dropout']),
            torch.nn.Linear(hyperparams['hidden_dim'], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.layers(x).squeeze(-1)

# =============================================================================
# Fully Connected Models
# =============================================================================
@register_model
class FullyConnectedAutoencoder(AutoencoderBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_features, hyperparams['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(hyperparams['dropout'])
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hyperparams['hidden_dim'], num_features),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, seq_len, self.num_features)

@register_model
class FullyConnectedBinaryClassifier(ClassifierBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, hyperparams['hidden_dim']),
            torch.nn.ReLU(),
            torch.nn.Dropout(hyperparams['dropout']),
            torch.nn.Linear(hyperparams['hidden_dim'], 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size * seq_len, -1)
        logits = self.layers(x)
        return logits.view(batch_size, seq_len).mean(dim=1)

# =============================================================================
# 1D CNN Models
# =============================================================================
@register_model
class OneDimensionalCNNAutoencoder(AutoencoderBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(num_features, hyperparams['hidden_dim'], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.Conv1d(hyperparams['hidden_dim'], hyperparams['hidden_dim'] // 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, stride=2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(hyperparams['hidden_dim'] // 2, hyperparams['hidden_dim'], kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(hyperparams['hidden_dim'], num_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # (B, C, L)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.permute(0, 2, 1) # (B, L, C)

@register_model
class OneDimensionalCNNBinaryClassifier(ClassifierBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(num_features, hyperparams['hidden_dim'], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, stride=2),
            torch.nn.Dropout(hyperparams['dropout'])
        )
        # Calculate the flattened size after conv layers
        conv_output_size = sequence_length // 2
        flattened_size = hyperparams['hidden_dim'] * conv_output_size
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) # (B, C, L)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc_layers(x).squeeze(-1)

# =============================================================================
# GRU Models
# =============================================================================
@register_model
class GRUAutoencoder(AutoencoderBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.encoder = torch.nn.GRU(
            input_size=num_features,
            hidden_size=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            batch_first=True,
            dropout=hyperparams['dropout']
        )
        self.decoder = torch.nn.GRU(
            input_size=hyperparams['hidden_dim'],
            hidden_size=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            batch_first=True,
            dropout=hyperparams['dropout']
        )
        self.output_layer = torch.nn.Linear(hyperparams['hidden_dim'], num_features)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        output = input_sequence
        output, hidden_state = self.encoder(output)
        output, hidden_state = self.decoder(output)
        reconstruction = self.output_layer(output)
        return reconstruction

@register_model
class GRUBinaryClassifier(ClassifierBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.gru= torch.nn.GRU(
            input_size=num_features,
            hidden_size=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            batch_first=True,
            dropout=hyperparams['dropout']
        )
        self.classifier_layer = torch.nn.Linear(hyperparams['hidden_dim'], 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        output = input_sequence
        output, hidden_state = self.gru(output)
        last_output = output[:, -1, :]
        class_prediction = self.classifier_layer(last_output).squeeze(1)
        return class_prediction

# =============================================================================
# LSTM Models
# =============================================================================
@register_model
class LSTMAutoencoder(AutoencoderBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
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

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        output = input_sequence
        output, (hidden_state, cell_state) = self.encoder(output)
        output, (hidden_state, cell_state) = self.decoder(output)
        reconstruction = self.output_layer(output)
        return reconstruction

@register_model
class LSTMBinaryClassifier(ClassifierBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        self.lstm= torch.nn.LSTM(
            input_size=num_features,
            hidden_size=hyperparams['hidden_dim'],
            num_layers=hyperparams['num_layers'],
            batch_first=True,
            dropout=hyperparams['dropout']
        )
        self.classifier_layer = torch.nn.Linear(hyperparams['hidden_dim'], 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        output = input_sequence
        output, (hidden_state, cell_state) = self.lstm(output)
        last_output = output[:, -1, :]
        class_prediction = self.classifier_layer(last_output).squeeze(1)
        return class_prediction

# =============================================================================
# Transformer Models
# =============================================================================
@register_model
class TransformerAutoencoder(AutoencoderBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=num_features, nhead=4, dim_feedforward=hyperparams['hidden_dim'], dropout=hyperparams['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=hyperparams['num_layers'])
        self.output_layer = torch.nn.Linear(num_features, num_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)  # (S, B, E)
        encoded = self.transformer_encoder(x)
        decoded = self.output_layer(encoded)
        return decoded.permute(1, 0, 2)  # (B, S, E)

@register_model
class TransformerBinaryClassifier(ClassifierBase):
    def __init__(self, num_features: int, hyperparams: dict, sequence_length: int, save_dir: str, trial_number: int = None):
        super().__init__(num_features, hyperparams, sequence_length, save_dir, trial_number)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=num_features, nhead=4, dim_feedforward=hyperparams['hidden_dim'], dropout=hyperparams['dropout'])
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=hyperparams['num_layers'])
        self.classifier_layer = torch.nn.Linear(num_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)  # (S, B, E)
        encoded = self.transformer_encoder(x)
        last_output = encoded[-1, :, :]  # (B, E)
        class_prediction = self.classifier_layer(last_output).squeeze(1)
        return class_prediction