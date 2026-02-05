from abc import ABC as AbstractBaseClass
import plotly.express
import torchmetrics
import lightning
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

#region Base Models
class BaseModel(lightning.LightningModule, AbstractBaseClass):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model_params = model_params
        self.data_params = data_params
        self.layer_sizes = model_params.get("layer_sizes")[:model_params.get("num_layers")]

        self.register_buffer("scaler_mean", torch.zeros(len(self.data_params.get('whitelist'))))
        self.register_buffer("scaler_std", torch.ones(len(self.data_params.get('whitelist'))))

        self.test_step_outputs = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_name = self.model_params.get('optimizer_name', 'Adam')
        learning_rate = self.model_params.get('learning_rate', 0.001)
        weight_decay = self.model_params.get('weight_decay', 0)
        scheduler = self.model_params.get('scheduler')
        optimizer, scheduler = None, None
        match optimizer_name:
            case 'Adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
            case 'RMSprop':
                momentum = self.model_params.get('momentum')
                optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
            case 'SGD':
                momentum = self.model_params.get('momentum')
                optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        match scheduler: # Do I want chained schedulers? What other schedulers would be good? Do I want to tune scheduler params?
            case 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        if scheduler:
            return (optimizer, scheduler)
        return optimizer
    
    def set_scaler_params(self, mean_array, std_array):
        """Loads scaler parameters into model buffers."""
        self.scaler_mean = torch.tensor(mean_array, dtype=torch.float32)
        self.scaler_std = torch.tensor(std_array, dtype=torch.float32)

    def scale_data(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Applies scaling on the current torch device."""
        return (input_sequence - self.scaler_mean) / self.scaler_std

class AutoencoderBase(BaseModel):
    training_type = constants.TrainingType.UNSUPERVISED

    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        loss_function = torch.nn.MSELoss()
        object.__setattr__(self, "loss_function", loss_function) # To avoid PyTorch Lightning issues with assigning to self.loss_function
        self.decoder_layer_sizes = self.layer_sizes[::-1][1:] # Doesn't include i/o layers

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        scaled_inputs = self.scale_data(inputs)
        reconstruction = self.forward(scaled_inputs)
        loss = self.loss_function(reconstruction, scaled_inputs)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        scaled_inputs = self.scale_data(inputs)
        reconstruction = self.forward(scaled_inputs)
        loss = self.loss_function(reconstruction, scaled_inputs)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inputs, labels = batch
        scaled_inputs = self.scale_data(inputs)
        reconstruction = self.forward(scaled_inputs)
        loss = self.loss_function(reconstruction, scaled_inputs)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.test_step_outputs.append({'Batch_Index': batch_idx, 'MSE_Loss': loss.detach().item()})

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        figure = plotly.express.line(
            data_frame=pandas.DataFrame(self.test_step_outputs),
            x='Batch Index', 
            y='MSE Loss',
            title=f"{self.__class__.__name__} Reconstruction History:",
        )
        figure.write_html(f"{self.__class__.__name__}_reconstruction_history.html")
        self.test_step_outputs.clear()

class ClassifierBase(BaseModel):
    training_type = constants.TrainingType.SUPERVISED

    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        loss_function = torch.nn.BCEWithLogitsLoss()
        object.__setattr__(self, "loss_function", loss_function)
        self.test_accuracy = torchmetrics.BinaryAccuracy()

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        scaled_inputs = self.scale_data(inputs)
        logits = self.forward(scaled_inputs) # Logits are the raw, unnormalized outputs from the last layer (pass through sigmoid for probabilities or loss function for loss calculation)
        labels = labels.float().view_as(logits)
        loss = self.loss_function(logits, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        scaled_inputs = self.scale_data(inputs)
        logits = self.forward(scaled_inputs)
        labels = labels.float().view_as(logits)
        loss = self.loss_function(logits, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        inputs, labels = batch
        scaled_inputs = self.scale_data(inputs)
        logits = self.forward(scaled_inputs)
        labels = labels.float().view_as(logits)
        loss = self.loss_function(logits, labels)
        self.test_accuracy(logits, labels)
        probability = torch.sigmoid(logits)
        metrics = {
            'test_loss': loss,
            'test_accuracy': self.test_accuracy,
            'test_mean_confidence': probability.mean()
        }
        self.log_dict(metrics, on_epoch=True)
        self.test_step_outputs.append({
            "Batch_Index": batch_idx,
            "Probability": probability.detach().cpu().numpy().item(),
            "True_Label": int(labels.detach().cpu().numpy().item())
        })

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        figure = plotly.express.line(
            data_frame=pandas.DataFrame(self.test_step_outputs),
            x='Batch Index',
            y='Classification',
            title=f"{self.__class__.__name__} Classification History:",
        )
        figure.write_html(f"{self.__class__.__name__}_classification_history.html")
        self.test_step_outputs.clear()
#endregion

#region Dense Models
@register_model
class DenseAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist')) * self.data_params.get('polls_per_sequence')
        
        encoder_layers = []
        current_dimension = input_dimension
        for layer_size in self.layer_sizes:
            encoder_layers.append(torch.nn.Linear(current_dimension, layer_size))
            encoder_layers.append(torch.nn.BatchNorm1d(layer_size))
            encoder_layers.append(torch.nn.LeakyReLU())
            encoder_layers.append(torch.nn.Dropout(model_params.get('dropout')))
            current_dimension = layer_size
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        for layer_size in self.decoder_layer_sizes:
            decoder_layers.append(torch.nn.Linear(current_dimension, layer_size))
            decoder_layers.append(torch.nn.LeakyReLU())
            current_dimension = layer_size
        decoder_layers.append(torch.nn.Linear(current_dimension, input_dimension))
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        flattened_input = input_sequence.flatten(start_dim=1) # (Batch, Sequence*Feature)
        encoded_output = self.encoder(flattened_input)
        reconstructed_output = self.decoder(encoded_output)
        return reconstructed_output.view(input_sequence.size())

@register_model
class DenseBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist')) * self.data_params.get('polls_per_sequence')
        
        layers = []
        current_dimension = input_dimension
        for layer_size in self.layer_sizes:
            layers.append(torch.nn.Linear(current_dimension, layer_size))
            layers.append(torch.nn.BatchNorm1d(layer_size))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout(model_params.get('dropout')))
            current_dimension = layer_size
        layers.append(torch.nn.Linear(current_dimension, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        flattened_input = input_sequence.flatten(start_dim=1) # (Batch, Sequence*Feature)
        logits = self.layers(flattened_input)
        return logits.squeeze(-1)
#endregion

#region 1D CNN Models
@register_model
class CNNAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist'))

        encoder_layers = []
        input_channels = input_dimension
        for i, output_channels in enumerate(self.layer_sizes):
            encoder_layers.append(torch.nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1))
            encoder_layers.append(torch.nn.BatchNorm1d(output_channels))
            encoder_layers.append(torch.nn.LeakyReLU())
            if i < len(self.layer_sizes) - 1:
                encoder_layers.append(torch.nn.Dropout1d(model_params.get('dropout')))
            input_channels = output_channels
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        for output_channels in self.decoder_layer_sizes:
            decoder_layers.append(torch.nn.ConvTranspose1d(input_channels, output_channels, kernel_size=3, padding=1))
            decoder_layers.append(torch.nn.BatchNorm1d(output_channels))
            decoder_layers.append(torch.nn.LeakyReLU())
            input_channels = output_channels
        decoder_layers.append(torch.nn.Conv1d(input_channels, input_dimension, kernel_size=1))
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1) # (Batch, Channel(Feature), Sequence)
        encoded_output = self.encoder(permuted_input)
        reconstructed_output = self.decoder(encoded_output)
        return reconstructed_output.permute(0, 2, 1) # (Batch, Sequence, Feature)

@register_model
class CNNBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        
        layers = []
        input_channels = len(self.data_params.get('whitelist'))
        for output_channels in self.layer_sizes:
            layers.append(torch.nn.Conv1d(input_channels, output_channels, kernel_size=3))
            layers.append(torch.nn.BatchNorm1d(output_channels))
            layers.append(torch.nn.LeakyReLU())
            layers.append(torch.nn.Dropout1d(model_params.get('dropout')))
            input_channels = output_channels
        self.feature_extractor = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(input_channels, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1) # (Batch, Channel(Feature), Sequence)
        extracted_features = self.feature_extractor(permuted_input)
        pooled_features = torch.mean(extracted_features, dim=2)
        logits = self.classifier(pooled_features)
        return logits.squeeze(-1)
#endregion

#region GRU Models
@register_model
class GRUAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist'))
        
        self.encoder_layers = torch.nn.ModuleList()
        current_dimension = input_dimension
        for i, layer_size in enumerate(self.layer_sizes):
            self.encoder_layers.append(torch.nn.GRU(current_dimension, layer_size, batch_first=True))
            if i < len(self.layer_sizes) - 1:
                self.encoder_layers.append(torch.nn.Dropout(model_params.get('dropout')))
            current_dimension = layer_size
        
        self.decoder_layers = torch.nn.ModuleList()
        for layer_size in self.decoder_layer_sizes:
            self.decoder_layers.append(torch.nn.GRU(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
        self.decoder_layers.append(torch.nn.GRU(current_dimension, model_params.get('layer_sizes')[0], batch_first=True))
        self.reconstruction_layer = torch.nn.Linear(current_dimension, input_dimension)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        data = input_sequence
        hidden_state = None
        encoder_states = []
        for encoder_layer in self.encoder_layers:
            if isinstance(encoder_layer, torch.nn.GRU):
                data, hidden_state = encoder_layer(data)
                encoder_states.append(hidden_state)
            else:
                data = encoder_layer(data)
        polls_per_sequence = input_sequence.size(1)
        latent_vector = hidden_state[-1].unsqueeze(1)
        repeat_vector = latent_vector.repeat(1, polls_per_sequence, 1)
        data = repeat_vector
        for i, decoder_layer in enumerate(self.decoder_layers):
            if i < len(encoder_states):
                data, hidden_state = decoder_layer(data, encoder_states[i])
            else:
                data, hidden_state = decoder_layer(data)
        return self.reconstruction_layer(data)

@register_model
class GRUBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist'))

        self.layers = torch.nn.ModuleList()
        current_dimension = input_dimension
        for layer_size in self.layer_sizes:
            self.layers.append(torch.nn.GRU(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
        self.dropout = torch.nn.Dropout(model_params.get('dropout'))
        self.classifier = torch.nn.Linear(current_dimension, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        current_data = input_sequence
        final_hidden_state = None
        for layer in self.layers:
            current_data, hidden_state = layer(current_data)
        final_hidden_state = self.dropout(hidden_state[-1])
        logits = self.classifier(final_hidden_state)
        return logits.squeeze(-1)
#endregion

#region LSTM Models
@register_model
class LSTMAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist'))

        self.encoder_layers = torch.nn.ModuleList()
        current_dimension = input_dimension
        for i, layer_size in enumerate(self.layer_sizes):
            self.encoder_layers.append(torch.nn.LSTM(current_dimension, layer_size, batch_first=True))
            if i < len(self.layer_sizes) - 1:
                self.encoder_layers.append(torch.nn.Dropout(model_params.get('dropout')))
            current_dimension = layer_size
            
        self.decoder_layers = torch.nn.ModuleList()
        for layer_size in self.decoder_layer_sizes:
            self.decoder_layers.append(torch.nn.LSTM(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
        self.decoder_layers.append(torch.nn.LSTM(current_dimension, model_params.get('layer_sizes')[0], batch_first=True))
        self.reconstruction_layer = torch.nn.Linear(current_dimension, input_dimension)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        data = input_sequence
        hidden_state, cell_state = None, None
        encoder_states = []
        for encoder_layer in self.encoder_layers:
            if isinstance(encoder_layer, torch.nn.LSTM):
                data, (hidden_state, cell_state) = encoder_layer(data, (hidden_state, cell_state))
                encoder_states.append((hidden_state, cell_state))
            else:
                data = encoder_layer(data)
        polls_per_sequence = input_sequence.size(1)
        latent_vector = hidden_state[-1].unsqueeze(1)
        repeat_vector = latent_vector.repeat(1, polls_per_sequence, 1)
        data = repeat_vector
        for i, decoder_layer in enumerate(self.decoder_layers):
            if i < len(encoder_states):
                data, (hidden_state, cell_state) = decoder_layer(data, encoder_states[i])
            else:
                data, (hidden_state, cell_state) = decoder_layer(data)
        return self.reconstruction_layer(data)

@register_model
class LSTMBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist'))

        self.layers = torch.nn.ModuleList()
        current_dimension = input_dimension
        for layer_size in self.layer_sizes:
            self.layers.append(torch.nn.LSTM(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
        self.dropout = torch.nn.Dropout(model_params.get('dropout'))
        self.classifier = torch.nn.Linear(current_dimension, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        current_data = input_sequence
        hidden_state = None
        for layer in self.layers:
            current_data, (hidden_state, cell_state) = layer(current_data)
        final_hidden_state = self.dropout(hidden_state[-1])
        logits = self.classifier(final_hidden_state)
        return logits.squeeze(-1)
#endregion

#region Transformer Models
# @register_model
# class TransformerAutoencoder(AutoencoderBase):
#     def __init__(self, model_params: dict, data_params: dict):
#         super().__init__(model_params, data_params)
#         encoder_layer = torch.nn.TransformerEncoderLayer(d_model=len(self.data_params.get('whitelist')), nhead=4, dim_feedforward=model_params['hidden_dim'], dropout=model_params['dropout'])
#         self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=model_params['num_layers'])
#         self.output_layer = torch.nn.Linear(len(self.data_params.get('whitelist')), len(self.data_params.get('whitelist')))
    
#     def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
#         input_sequence = input_sequence.permute(1, 0, 2)  # (S, B, E)
#         encoded = self.transformer_encoder(input_sequence)
#         decoded = self.output_layer(encoded)
#         return decoded.permute(1, 0, 2)  # (B, S, E)

# @register_model
# class TransformerBinaryClassifier(ClassifierBase):
#     def __init__(self, model_params: dict, data_params: dict):
#         super().__init__(model_params, data_params)
#         encoder_layer = torch.nn.TransformerEncoderLayer(d_model=len(self.data_params.get('whitelist')), nhead=4, dim_feedforward=model_params['hidden_dim'], dropout=model_params['dropout'])
#         self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=model_params['num_layers'])
#         self.classifier_layer = torch.nn.Linear(len(self.data_params.get('whitelist')), 1)
    
#     def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
#         input_sequence = input_sequence.permute(1, 0, 2)  # (S, B, E)
#         encoded = self.transformer_encoder(input_sequence)
#         last_output = encoded[-1, :, :]  # (B, E)
#         class_prediction = self.classifier_layer(last_output).squeeze(1)
#         return class_prediction
#endregion