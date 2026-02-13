from abc import ABC as AbstractBaseClass
import plotly.express
import torchmetrics
import lightning
import pydantic
import typing
import pandas
import torch
import constants

#region Runtime Model Registry
AVAILABLE_MODELS = {}

def register_model(model: type[lightning.LightningModule]) -> type[lightning.LightningModule]:
    """Decorator to register a model class."""
    AVAILABLE_MODELS[model.__name__] = model
    return model

def get_available_models() -> dict[str, type[lightning.LightningModule]]:
    """Returns a dictionary of available model classes."""
    return AVAILABLE_MODELS
#endregion

#region Metadata Configs
class DataConfig(pydantic.BaseModel):
    """Parameters defining the dataset and input structure."""
    whitelist: typing.List[str]
    polling_rate: int
    ignore_empty_polls: bool
    polls_per_sequence: int

    @property
    def features_per_poll(self) -> int:
        return len(self.whitelist)

class ModelConfig(pydantic.BaseModel):
    """Parameters defining the neural network architecture and optimization."""
    hidden_layers: int = pydantic.Field(gt=0)
    hidden_size: int = pydantic.Field(gt=0)
    latent_size: int = pydantic.Field(gt=0)
    dropout: float = pydantic.Field(default=0.0, ge=0.0, le=0.5)
    optimizer_name: str = 'Adam'
    scheduler_name: typing.Optional[str] = None
    learning_rate: float = pydantic.Field(default=1e-3, gt=0)
    weight_decay: float = pydantic.Field(default=0, ge=0)
#endregion

#region Base Models
class BaseModel(lightning.LightningModule, AbstractBaseClass):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__()
        self.save_hyperparameters()
        self.model_params = model_params
        self.data_params = data_params
        
        self.register_buffer('scaler_mean', torch.zeros(self.data_params.features_per_poll))
        self.register_buffer('scaler_std', torch.ones(self.data_params.features_per_poll))

        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer_class = constants.OPTIMIZER_MAP[self.model_params.optimizer_name]
        optimizer = optimizer_class(self.parameters(), lr=self.model_params.learning_rate, weight_decay=self.model_params.weight_decay)
        if self.model_params.scheduler_name:
            scheduler_class = constants.SCHEDULER_MAP[self.model_params.scheduler_name]
            scheduler = scheduler_class(optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                },
            }
        return optimizer
    
    def set_scaler_params(self, mean_array, std_array):
        """Loads scaler parameters into model buffers."""
        self.scaler_mean = torch.tensor(mean_array, dtype=torch.float32, device=self.device)
        self.scaler_std = torch.tensor(std_array, dtype=torch.float32, device=self.device)

    def scale_data(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Applies scaling on the current torch device."""
        return (input_sequence - self.scaler_mean) / self.scaler_std

class AutoencoderBase(BaseModel):
    training_type = constants.TrainingType.UNSUPERVISED

    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)
        self.loss_function = torch.nn.MSELoss()

    def _common_step(self, batch, batch_idx, stage: str):
        inputs, labels = batch
        scaled_inputs = self.scale_data(inputs)
        reconstruction = self.forward(scaled_inputs)
        loss = self.loss_function(reconstruction, scaled_inputs)
        self.log(
            name=f'{stage}_loss', 
            value=loss, 
            on_step=False, 
            on_epoch=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, 'test')
        self.test_step_outputs.append({
            'Batch_Index': batch_idx, 
            'MSE_Loss': loss.detach().item()
    })

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        figure = plotly.express.line(
            data_frame=pandas.DataFrame(self.test_step_outputs),
            x='Batch Index', 
            y='MSE Loss',
            title=f'{self.__class__.__name__} Reconstruction History:',
        )
        figure.write_html(f'{self.__class__.__name__}_reconstruction_history.html')
        self.test_step_outputs.clear()

class ClassifierBase(BaseModel):
    training_type = constants.TrainingType.SUPERVISED

    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.test_accuracy = torchmetrics.BinaryAccuracy()
    
    def _common_step(self, batch):
        """Core logic shared by train, val, and test."""
        inputs, labels = batch
        scaled_inputs = self.scale_data(inputs)
        logits = self.forward(scaled_inputs)
        labels = labels.float().view_as(logits)
        loss = self.loss_function(logits, labels)
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch)
        self.test_accuracy(logits, labels)
        probability = torch.sigmoid(logits)
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': self.test_accuracy,
            'test_mean_confidence': probability.mean()
        }, on_epoch=True)
        self.test_step_outputs.append({
            'Batch_Index': batch_idx,
            'Probability': probability.detach().cpu().item(),
            'True_Label': int(labels.detach().cpu().item())
        })

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        figure = plotly.express.line(
            data_frame=pandas.DataFrame(self.test_step_outputs),
            x='Batch Index',
            y='Classification',
            title=f'{self.__class__.__name__} Classification History:',
        )
        figure.write_html(f'{self.__class__.__name__}_classification_history.html')
        self.test_step_outputs.clear()
#endregion

#region Dense Models
@register_model
class DenseAutoencoder(AutoencoderBase):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)
        input_dimension = self.data_params.polls_per_sequence * self.data_params.features_per_poll
        
        # Generate symmetrical cascading layer sizes
        encoder_sizes = [input_dimension]
        if self.model_params.hidden_layers == 1:
            encoder_sizes.append(self.model_params.latent_size)
        else:
            for i in range(self.model_params.hidden_layers):
                ratio = i / (self.model_params.hidden_layers - 1) # Ratio from 0.0 (hidden_size) to 1.0 (latent_size)
                layer_size = max(1, int(self.model_params.hidden_size * (self.model_params.latent_size / self.model_params.hidden_size) ** ratio))
                encoder_sizes.append(layer_size)
        decoder_sizes = list(reversed(encoder_sizes))
        
        encoder_layers = []
        for i in range(len(encoder_sizes) - 1):
            encoder_layers.append(torch.nn.Linear(encoder_sizes[i], encoder_sizes[i+1]))
            if i < len(encoder_sizes) - 2:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_sizes[i+1]))
                encoder_layers.append(torch.nn.ELU())
                encoder_layers.append(torch.nn.Dropout(self.model_params.dropout))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(len(decoder_sizes) - 1):
            decoder_layers.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i < len(decoder_sizes) - 2:
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_sizes[i+1]))
                decoder_layers.append(torch.nn.ELU())
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        flattened_input = input_sequence.flatten(start_dim=1)
        encoded_sequence = self.encoder(flattened_input)
        decoded_sequence = self.decoder(encoded_sequence)
        return decoded_sequence.unflatten(dim=1, sizes=(self.data_params.polls_per_sequence, self.data_params.features_per_poll))

@register_model
class DenseBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)
        input_dimension = self.data_params.polls_per_sequence * self.data_params.features_per_poll
        
        layers = []
        current_dimension = input_dimension
        for _ in range(self.model_params.hidden_layers):
            layers.append(torch.nn.Linear(current_dimension, self.model_params.hidden_size))
            layers.append(torch.nn.BatchNorm1d(self.model_params.hidden_size))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.Dropout(self.model_params.dropout))
            current_dimension = self.model_params.hidden_size
        layers = layers[:-3]
        layers.append(torch.nn.Linear(current_dimension, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        flattened_input = input_sequence.flatten(start_dim=1) # (Batch, Sequence*Feature)
        logits = self.layers(flattened_input)
        return logits.view(-1)
#endregion

#region 1D CNN Models
@register_model
class CNNAutoencoder(AutoencoderBase):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)
        input_channels = self.data_params.features_per_poll

        encoder_sizes = [input_channels]
        if self.model_params.hidden_layers == 1:
            encoder_sizes.append(self.model_params.latent_size)
        else:
            for i in range(self.model_params.hidden_layers):
                ratio = i / (self.model_params.hidden_layers - 1) # Ratio from 0.0 (hidden_size) to 1.0 (latent_size)
                layer_size = max(1, int(self.model_params.hidden_size * (self.model_params.latent_size / self.model_params.hidden_size) ** ratio))
                encoder_sizes.append(layer_size)
        encoder_layers = []
        for i in range(len(encoder_sizes) - 1):
            encoder_layers.append(torch.nn.Conv1d(encoder_sizes[i], encoder_sizes[i+1], kernel_size=3, stride=2, padding=1))
            if i < len(encoder_sizes) - 2:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_sizes[i+1]))
                encoder_layers.append(torch.nn.ELU())
                encoder_layers.append(torch.nn.Dropout1d(self.model_params.dropout))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_sizes = list(reversed(encoder_sizes))
        decoder_layers = []
        for i in range(len(decoder_sizes) - 1):
            decoder_layers.append(torch.nn.Upsample(scale_factor=2, mode='nearest')) # Scale factor must match encoder stride
            decoder_layers.append(torch.nn.Conv1d(decoder_sizes[i], decoder_sizes[i+1], kernel_size=3, padding=1))
            if i < len(decoder_sizes) - 2:
                decoder_layers.append(torch.nn.BatchNorm1d(decoder_sizes[i+1]))
                decoder_layers.append(torch.nn.ELU())
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1) # (Batch, Channel(Feature), Sequence)
        encoded_sequence = self.encoder(permuted_input)
        decoded_sequence = self.decoder(encoded_sequence)
        return decoded_sequence.permute(0, 2, 1) # (Batch, Sequence, Feature)

@register_model
class CNNBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)
        
        layers = []
        current_channels = self.data_params.features_per_poll
        for _ in range(self.model_params.hidden_layers):
            layers.append(torch.nn.Conv1d(current_channels, self.model_params.hidden_size, kernel_size=3))
            layers.append(torch.nn.BatchNorm1d(self.model_params.hidden_size))
            layers.append(torch.nn.ELU())
            layers.append(torch.nn.Dropout1d(self.model_params.dropout))
            current_channels = self.model_params.hidden_size
        layers = layers[:-3]
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(current_channels, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1) # (Batch, Channel(Feature), Sequence)
        extracted_features = self.feature_extractor(permuted_input)
        pooled_features = torch.mean(extracted_features, dim=2)
        logits = self.classifier(pooled_features)
        return logits.view(-1)
#endregion

#region GRU Models
@register_model
class GRUAutoencoder(AutoencoderBase):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)
        
        self.encoder = torch.nn.GRU(
            input_size=self.data_params.features_per_poll, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout if self.model_params.hidden_layers > 1 else 0, 
            batch_first=True
        )
        self.compressor = torch.nn.Linear(self.model_params.hidden_size, self.model_params.latent_size)
        self.decompressor = torch.nn.Linear(self.model_params.latent_size, self.model_params.hidden_size)
        self.decoder = torch.nn.GRU(
            input_size=self.model_params.latent_size, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout if self.model_params.hidden_layers > 1 else 0, 
            batch_first=True
        )
        self.reconstructor = torch.nn.Linear(self.model_params.hidden_size, self.data_params.features_per_poll)
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        encoded_sequence, hidden_state = self.encoder(input_sequence)
        latent_vector = self.compressor(hidden_state[-1])
        repeat_vector = latent_vector.unsqueeze(1).repeat(1, self.data_params.polls_per_sequence, 1)
        context_vector = self.decompressor(latent_vector) 
        context_vector = context_vector.unsqueeze(0).repeat(self.model_params.hidden_layers, 1, 1)
        decoded_sequence, hidden_state = self.decoder(repeat_vector, context_vector) # Am I passing too much?
        return self.reconstructor(decoded_sequence)

@register_model
class GRUBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)

        self.feature_extractor = torch.nn.GRU(
            input_size=self.data_params.features_per_poll, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout if self.model_params.hidden_layers > 1 else 0,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(self.model_params.hidden_size, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        data, hidden_state = self.feature_extractor(input_sequence)
        pooled_data = data.mean(dim=1) # Globally pool to get even coverage of the sequence
        logits = self.classifier(pooled_data)
        return logits.view(-1)
#endregion

#region LSTM Models
@register_model
class LSTMAutoencoder(AutoencoderBase):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)
        
        self.encoder = torch.nn.LSTM(
            input_size=self.data_params.features_per_poll, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout if self.model_params.hidden_layers > 1 else 0, 
            batch_first=True
        )
        self.compressor = torch.nn.Linear(self.model_params.hidden_size*2, self.model_params.latent_size)
        self.decompressor = torch.nn.Linear(self.model_params.latent_size, self.model_params.hidden_size)
        self.decoder = torch.nn.LSTM(
            input_size=self.model_params.latent_size, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout if self.model_params.hidden_layers > 1 else 0, 
            batch_first=True
        )
        self.reconstructor = torch.nn.Linear(self.model_params.hidden_size, self.data_params.features_per_poll)
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        encoded_sequence, (hidden_state, cell_state) = self.encoder(input_sequence)
        final_states = torch.cat((hidden_state[-1], cell_state[-1]), dim=-1) 
        latent_vector = self.compressor(final_states)
        context_vector = self.decompressor(latent_vector) 
        repeat_vector = latent_vector.unsqueeze(1).repeat(1, self.data_params.polls_per_sequence, 1)
        final_states = context_vector.unsqueeze(0).repeat(self.model_params.hidden_layers, 1, 1)
        decoded_sequence, (hidden_state, cell_state) = self.decoder(repeat_vector, (final_states, final_states))
        return self.reconstructor(decoded_sequence)

@register_model
class LSTMBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: ModelConfig, data_params: DataConfig):
        super().__init__(model_params, data_params)

        self.feature_extractor = torch.nn.LSTM(
            input_size=self.data_params.features_per_poll, 
            hidden_size=self.model_params.hidden_size,
            num_layers=self.model_params.hidden_layers,
            dropout=self.model_params.dropout if self.model_params.hidden_layers > 1 else 0,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(self.model_params.hidden_size, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        data, (hidden_state, cell_state) = self.feature_extractor(input_sequence)
        pooled_data = data.mean(dim=1) # Globally pool to get even coverage of the sequence
        logits = self.classifier(pooled_data)
        return logits.view(-1)
#endregion