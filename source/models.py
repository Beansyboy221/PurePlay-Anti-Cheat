from abc import ABC
import lightning
import torch
import constants
import torchmetrics

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
class BaseModel(lightning.LightningModule, ABC):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model_params = model_params
        self.data_params = data_params
        self.layer_sizes = model_params["layer_sizes"][:model_params["num_layers"]]

        self.register_buffer("scaler_mean", torch.zeros(len(self.data_params.get('whitelist'))))
        self.register_buffer("scaler_std", torch.ones(len(self.data_params.get('whitelist'))))

    def set_scaler_params(self, mean_array, std_array):
        """Loads scaler parameters into model buffers."""
        self.scaler_mean = torch.tensor(mean_array, dtype=torch.float32)
        self.scaler_std = torch.tensor(std_array, dtype=torch.float32)

    def scale_data(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """Applies scaling on the current torch device."""
        return (input_sequence - self.scaler_mean) / self.scaler_std

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_name = self.model_params.get('optimizer_name')
        learning_rate = self.model_params.get('learning_rate')
        weight_decay = self.model_params.get('weight_decay')
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

class AutoencoderBase(BaseModel):
    training_type = constants.TrainingType.UNSUPERVISED

    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        loss_function = torch.nn.MSELoss()
        object.__setattr__(self, "loss_function", loss_function) # To avoid PyTorch Lightning issues with assigning to self.loss_function

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, _ = batch
        scaled_inputs = self.scale_data(inputs)
        reconstruction = self.forward(scaled_inputs)

        loss = self.loss_function(reconstruction, scaled_inputs)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, _ = batch
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
        logits = self.forward(scaled_inputs) # Logits are the raw, unnormalized scores output by the last layer (pass through sigmoid for probabilities or loss function for loss calculation)
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
        probabilities = torch.sigmoid(logits)
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_accuracy', self.test_accuracy, on_epoch=True)
        self.log('test_mean_confidence', probabilities.mean(), on_epoch=True)
#endregion

#region Registered Models

#region Feedforward Models
@register_model
class FeedforwardAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist')) * self.data_params.get('polls_per_sequence')
        
        encoder_layers = []
        current_dimension = input_dimension
        for layer_size in self.layer_sizes:
            encoder_layers.append(torch.nn.Linear(current_dimension, layer_size))
            encoder_layers.append(torch.nn.ReLU())
            encoder_layers.append(torch.nn.Dropout(model_params['dropout']))
            current_dimension = layer_size
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        reversed_layer_sizes = self.layer_sizes[::-1][1:] + [input_dimension]
        for layer_size in reversed_layer_sizes:
            decoder_layers.append(torch.nn.Linear(current_dimension, layer_size))
            if layer_size != input_dimension:
                decoder_layers.append(torch.nn.ReLU())
            current_dimension = layer_size
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        flattened_sequence = input_sequence.view(batch_size, -1)
        latent_representation = self.encoder(flattened_sequence)
        reconstructed_sequence = self.decoder(latent_representation)
        return reconstructed_sequence.view(batch_size, self.data_params.get('polls_per_sequence'), len(self.data_params.get('whitelist')))

@register_model
class FeedforwardBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        input_dimension = len(self.data_params.get('whitelist')) * self.data_params.get('polls_per_sequence')
        
        layers = []
        current_dimension = input_dimension
        for layer_size in self.layer_sizes:
            layers.append(torch.nn.Linear(current_dimension, layer_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(model_params['dropout']))
            current_dimension = layer_size
        layers.append(torch.nn.Linear(current_dimension, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        flattened_sequence = input_sequence.view(batch_size, -1)
        logits = self.network(flattened_sequence)
        return logits.squeeze(-1)
#endregion

#region Fully Connected Models
@register_model
class FullyConnectedAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        
        encoder_layers = []
        current_dimension = len(self.data_params.get('whitelist'))
        for layer_size in self.layer_sizes:
            encoder_layers.append(torch.nn.Linear(current_dimension, layer_size))
            encoder_layers.append(torch.nn.ReLU())
            encoder_layers.append(torch.nn.Dropout(model_params['dropout']))
            current_dimension = layer_size
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        reversed_layer_sizes = self.layer_sizes[::-1][1:] + [len(self.data_params.get('whitelist'))]
        for layer_size in reversed_layer_sizes:
            decoder_layers.append(torch.nn.Linear(current_dimension, layer_size))
            if layer_size != len(self.data_params.get('whitelist')):
                decoder_layers.append(torch.nn.ReLU())
            current_dimension = layer_size
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        batch_size, polls, features = input_sequence.size()
        reshaped_input = input_sequence.view(-1, features)
        latent_representation = self.encoder(reshaped_input)
        reconstructed_output = self.decoder(latent_representation)
        return reconstructed_output.view(batch_size, polls, features)

@register_model
class FullyConnectedBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        
        layers = []
        current_dimension = len(self.data_params.get('whitelist'))
        for layer_size in self.layer_sizes:
            layers.append(torch.nn.Linear(current_dimension, layer_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(model_params['dropout']))
            current_dimension = layer_size
        layers.append(torch.nn.Linear(current_dimension, 1))
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        batch_size, polls, features = input_sequence.size()
        reshaped_input = input_sequence.view(-1, features)
        logits_per_poll = self.network(reshaped_input)
        return logits_per_poll.view(batch_size, polls).mean(dim=1)
#endregion

#region 1D CNN Models
@register_model
class OneDimensionalCNNAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        
        encoder_layers = []
        current_channels = len(self.data_params.get('whitelist'))
        for layer_size in self.layer_sizes:
            encoder_layers.append(torch.nn.Conv1d(current_channels, layer_size, kernel_size=3, padding=1))
            encoder_layers.append(torch.nn.ReLU())
            current_channels = layer_size
        self.encoder = torch.nn.Sequential(*encoder_layers)

        decoder_layers = []
        reversed_channel_sizes = self.layer_sizes[::-1][1:] + [len(self.data_params.get('whitelist'))]
        for layer_size in reversed_channel_sizes:
            decoder_layers.append(torch.nn.Conv1d(current_channels, layer_size, kernel_size=3, padding=1))
            if layer_size != len(self.data_params.get('whitelist')):
                decoder_layers.append(torch.nn.ReLU())
            current_channels = layer_size
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1) 
        encoded_channels = self.encoder(permuted_input)
        decoded_channels = self.decoder(encoded_channels)
        return decoded_channels.permute(0, 2, 1)

@register_model
class OneDimensionalCNNBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        
        convolutional_blocks = []
        current_channels = len(self.data_params.get('whitelist'))
        for layer_size in self.layer_sizes:
            convolutional_blocks.append(torch.nn.Conv1d(current_channels, layer_size, kernel_size=3, padding=1))
            convolutional_blocks.append(torch.nn.ReLU())
            convolutional_blocks.append(torch.nn.Dropout(model_params['dropout']))
            current_channels = layer_size
        
        self.feature_extractor = torch.nn.Sequential(*convolutional_blocks)
        self.classifier_head = torch.nn.Linear(current_channels, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        permuted_input = input_sequence.permute(0, 2, 1)
        extracted_features = self.feature_extractor(permuted_input)
        pooled_features = torch.mean(extracted_features, dim=2)
        logits = self.classifier_head(pooled_features)
        return logits.squeeze(-1)
#endregion

#region GRU Models
@register_model
class GRUAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        
        self.encoder_recurrent_stack = torch.nn.ModuleList()
        current_dimension = len(self.data_params.get('whitelist'))
        for layer_size in self.layer_sizes:
            self.encoder_recurrent_stack.append(torch.nn.GRU(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
        
        self.decoder_recurrent_stack = torch.nn.ModuleList()
        reversed_layer_sizes = self.layer_sizes[::-1][1:] + [model_params['layer_sizes'][0]]
        for layer_size in reversed_layer_sizes:
            self.decoder_recurrent_stack.append(torch.nn.GRU(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
            
        self.reconstruction_layer = torch.nn.Linear(current_dimension, len(self.data_params.get('whitelist')))

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        current_data = input_sequence
        for gru_layer in self.encoder_recurrent_stack:
            current_data, last_hidden_state = gru_layer(current_data)
        context_vector = last_hidden_state[-1].unsqueeze(1)
        decoder_input = context_vector.repeat(1, self.data_params.get('polls_per_sequence'), 1)
        current_data = decoder_input
        for gru_layer in self.decoder_recurrent_stack:
            current_data, _ = gru_layer(current_data)
            
        return self.reconstruction_layer(current_data)

@register_model
class GRUBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        self.recurrent_stack = torch.nn.ModuleList()
        current_dimension = len(self.data_params.get('whitelist'))
        for layer_size in self.layer_sizes:
            self.recurrent_stack.append(torch.nn.GRU(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
            
        self.classifier_layer = torch.nn.Linear(current_dimension, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        current_data = input_sequence
        final_hidden_state = None
        for gru_layer in self.recurrent_stack:
            current_data, final_hidden_state = gru_layer(current_data)
        logits = self.classifier_layer(final_hidden_state[-1])
        return logits.squeeze(-1)
#endregion

#region LSTM Models
@register_model
class LSTMAutoencoder(AutoencoderBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        self.encoder_recurrent_stack = torch.nn.ModuleList()
        current_dimension = len(self.data_params.get('whitelist'))
        for layer_size in self.layer_sizes:
            self.encoder_recurrent_stack.append(torch.nn.LSTM(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
            
        self.decoder_recurrent_stack = torch.nn.ModuleList()
        reversed_layer_sizes = self.layer_sizes[::-1][1:] + [model_params['layer_sizes'][0]]
        for layer_size in reversed_layer_sizes:
            self.decoder_recurrent_stack.append(torch.nn.LSTM(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
            
        self.reconstruction_layer = torch.nn.Linear(current_dimension, len(self.data_params.get('whitelist')))

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        current_data = input_sequence
        for lstm_layer in self.encoder_recurrent_stack:
            current_data, (hidden_state, cell_state) = lstm_layer(current_data)
        
        context_vector = hidden_state[-1].unsqueeze(1)
        decoder_input = context_vector.repeat(1, self.data_params.get('polls_per_sequence'), 1)
        
        current_data = decoder_input
        for lstm_layer in self.decoder_recurrent_stack:
            current_data, _ = lstm_layer(current_data)
            
        return self.reconstruction_layer(current_data)

@register_model
class LSTMBinaryClassifier(ClassifierBase):
    def __init__(self, model_params: dict, data_params: dict):
        super().__init__(model_params, data_params)
        self.recurrent_stack = torch.nn.ModuleList()
        current_dimension = len(self.data_params.get('whitelist'))
        for layer_size in self.layer_sizes:
            self.recurrent_stack.append(torch.nn.LSTM(current_dimension, layer_size, batch_first=True))
            current_dimension = layer_size
            
        self.classifier_layer = torch.nn.Linear(current_dimension, 1)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        current_data = input_sequence
        final_hidden_state = None
        for lstm_layer in self.recurrent_stack:
            current_data, (final_hidden_state, final_cell_state) = lstm_layer(current_data)
        
        logits = self.classifier_layer(final_hidden_state[-1])
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

#endregion