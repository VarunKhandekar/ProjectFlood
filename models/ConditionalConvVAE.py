import torch
import torch.nn as nn

class ConvCVAE(nn.Module):
    def __init__(self, img_channels, condition_channels, latent_dim, LSTM_model, dropout_prob):
        super(ConvCVAE, self).__init__()

        self.LSTM_model = LSTM_model
        for param in self.LSTM_model.parameters():
            param.requires_grad = False
        self.preceding_rainfall_days = LSTM_model.preceding_rainfall_days
        self.forecast_rainfall_days = LSTM_model.forecast_rainfall_days
        self.dropout_prob = dropout_prob
        self.latent_dims = latent_dim
        
        # Encoder: Convolutional layers to downsample the image
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(img_channels + condition_channels, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 64 x 64
            nn.BatchNorm2d(64),  
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x 32 x 32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256 x 16 x 16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Dropout(p=self.dropout_prob),

            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 256),
            # nn.ReLU(),
            nn.Tanh()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        # self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        # self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        # self.fc_mu = nn.Linear(2048 * 2 * 2, latent_dim)  # Flatten to latent space
        # self.fc_logvar = nn.Linear(2048 * 2 * 2, latent_dim)

        # Convolutions for conditioning
        self.condition_conv = nn.Sequential(
            nn.Conv2d(condition_channels, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 128 x 128
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1) # Output: 1 x 16 x 16
        )

        # Decoder: Fully connected layers followed by transposed convolution layers
        # self.fc_decode = nn.Linear(latent_dim + condition_channels, 256 * 16 * 16)
        self.fc_decode = nn.Linear(latent_dim + (condition_channels*16*16), 256 * 16 * 16)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 128 x 32 x 32
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob), 

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 64 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32 x 128 x 128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),

            nn.ConvTranspose2d(32, img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: img_channels x 256 x 256
            nn.Sigmoid()
        )
    
    # Encoder: Takes in the image and condition, returns latent mean and log variance
    def encode(self, x, condition):
        # x = torch.cat([x, condition_img], dim=1)  # Concatenate along the channel dimension
        x = x.unsqueeze(1)
        print("x_shape: ", x.shape, "condition_shape: ", condition.shape)
        x = torch.cat([x, condition], dim=1)  # Concatenate along the channel dimension
        assert not torch.isnan(x).any(), f"NaN detected in x, {x}"
        print(x.shape)
        print(f"Input x min: {x.min().item()}, max: {x.max().item()}")
        print(f"Condition min: {condition.min().item()}, max: {condition.max().item()}")

        

        # Check weights
        # first_conv_layer = self.encoder_conv[0]
        # weights = first_conv_layer.weight.data
        # biases = first_conv_layer.bias.data if first_conv_layer.bias is not None else None

        # # Check for NaNs or Infs in weights
        # assert not torch.isnan(weights).any(), "Weights of the first convolutional layer contain NaNs"
        # assert not torch.isinf(weights).any(), "Weights of the first convolutional layer contain Infs"

        # # Check for NaNs or Infs in biases
        # if biases is not None:
        #     assert not torch.isnan(biases).any(), "Biases of the first convolutional layer contain NaNs"
        #     assert not torch.isinf(biases).any(), "Biases of the first convolutional layer contain Infs"

        # print("Weights and biases of the first convolutional layer are valid.")

        # for idx, layer in enumerate(self.encoder_conv):
        #     x = layer(x)
        #     if torch.isnan(x).any():
        #         print(f"NaN detected after layer {idx}: {layer}")
        #         break
        #     elif torch.isinf(x).any():
        #         print(f"Inf detected after layer {idx}: {layer}")
        #         break
        #     else:
        #         print(f"Layer {idx}: {layer}, output min: {x.min().item()}, max: {x.max().item()}")
        # assert not torch.isnan(x).any(), f"NaN detected in x, {x}"
        # h = x
        h = self.encoder_conv(x)
        # assert not torch.isnan(h).any(), f"NaN detected in h, {h}"
        # print(h.shape)
        # h_flat = torch.flatten(h, start_dim=1) # Flatten before fully connected layer
        # assert not torch.isnan(h_flat).any(), f"NaN detected in h_flat, {h_flat}"
        # # print(h_flat.shape)
        # # h = h.view(h.size(0), -1)  # Flatten before fully connected layer
        # return self.fc_mu(h_flat), self.fc_logvar(h_flat)
        return self.fc_mu(h), self.fc_logvar(h)
    
    # Reparameterization trick: Sample from the latent space
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # random noise of the same shape as std
        # print((mu + eps * std).shape)
        z = mu + eps * std
        assert not torch.isnan(z).any(), "NaN detected in z"
        # print("mu: ", mu, "logvar: ", logvar)
        # print(torch.exp(0.5 * logvar))
        assert not torch.isinf(z).any(), "inf detected in z"
        return z
    
    # Decoder: Takes the latent vector and condition to reconstruct the image
    def decode(self, z, condition):
        condition_encoded = self.condition_conv(condition)
        # condition_encoded = torch.mean(condition_encoded, dim=1)
        # print(condition_encoded.shape)
        condition_flat = torch.flatten(condition_encoded, start_dim=1)
        # print(z.shape, condition_flat.shape)
        z = torch.cat([z, condition_flat], dim=1)  # Concatenate latent vector and condition
        # print(z.shape)
        
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 16, 16)  # Reshape to feature maps
        # print(h.shape)
        assert not torch.isnan(h).any(), "NaN detected after fc_decode"

        # # Check weights
        # first_conv_layer = self.decoder_conv[0]
        # weights = first_conv_layer.weight.data
        # biases = first_conv_layer.bias.data if first_conv_layer.bias is not None else None

        # # Check for NaNs or Infs in weights
        # assert not torch.isnan(weights).any(), "Weights of the first convolutional layer contain NaNs"
        # assert not torch.isinf(weights).any(), "Weights of the first convolutional layer contain Infs"

        # # Check for NaNs or Infs in biases
        # if biases is not None:
        #     assert not torch.isnan(biases).any(), "Biases of the first convolutional layer contain NaNs"
        #     assert not torch.isinf(biases).any(), "Biases of the first convolutional layer contain Infs"

        # for idx, layer in enumerate(self.decoder_conv):
        #     h = layer(h)
        #     if torch.isnan(h).any():
        #         print(f"NaN detected after layer {idx}: {layer}")
        #         break
        #     elif torch.isinf(h).any():
        #         print(f"Inf detected after layer {idx}: {layer}")
        #         break
        #     else:
        #         print(f"Layer {idx}: {layer}, output min: {h.min().item()}, max: {h.max().item()}")
        # assert not torch.isnan(h).any(), f"NaN detected in h, {h}"
        # reconstructed = h


        reconstructed = self.decoder_conv(h)
        reconstructed = reconstructed.squeeze(dim=1) # drop channel dimension as it is 1
        return reconstructed
    
    # Forward pass through the model
    def forward(self, x, condition_sequence):
        assert not torch.isnan(condition_sequence).any(), "NaN detected in condition_sequence input"
        assert not torch.isinf(condition_sequence).any(), "Inf detected in condition_sequence"
        # print(condition_sequence.shape)
        self.LSTM_model.eval()

        condition = self.LSTM_model(condition_sequence)
        self.LSTM_model.hidden_state = None  # Reset the hidden state
        print(condition.shape)
        condition = condition.unsqueeze(1)
        print(condition.shape)
        # assert not torch.isnan(x).any(), f"NaN detected in input x {x}"
        assert not torch.isnan(condition).any(), f"NaN detected in condition {condition}"
        mu, logvar = self.encode(x, condition)
        print(f"mu min: {mu.min().item()}, max: {mu.max().item()}")
        print(f"logvar min: {logvar.min().item()}, max: {logvar.max().item()}")
        assert not torch.isnan(mu).any(), f"NaN detected in mu, {mu}"
        assert not torch.isnan(logvar).any(), f"NaN detected in logvar {logvar}"
        assert not torch.isinf(mu).any(), f"inf detected in mu, {mu}"
        assert not torch.isinf(logvar).any(), f"inf detected in logvar {logvar}"
        z = self.reparameterize(mu, logvar)
        print(f"z min: {z.min().item()}, max: {z.max().item()}")
        assert not torch.isnan(z).any(), f"NaN detected in z {z}"
        reconstructed = self.decode(z, condition)
        assert not torch.isnan(reconstructed).any(), f"NaN detected in reconstructed {reconstructed}"
        # print("Target min: ", torch.min(x), "Target max: ", torch.max(x))
        print("Reconstructed min: ", torch.min(reconstructed), "Reconstructed max: ", torch.max(reconstructed))
        return reconstructed, mu, logvar