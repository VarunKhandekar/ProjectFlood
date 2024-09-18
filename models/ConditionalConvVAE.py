import torch
import torch.nn as nn

class ConvCVAE(nn.Module):
    def __init__(self, img_channels, condition_channels, latent_dim, LSTM_model, dropout_prob):
        super(ConvCVAE, self).__init__()

        self.LSTM_model = LSTM_model
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
            nn.BatchNorm2d(256)
        )
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
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
        # condition_img = condition.view(condition.size(0), condition.size(1), 1, 1)  # Reshape condition
        # condition_img = condition_img.expand(-1, -1, x.size(2), x.size(3))  # Expand condition to match image size
        # x = torch.cat([x, condition_img], dim=1)  # Concatenate along the channel dimension
        x = x.unsqueeze(1)
        # print(x.shape, condition.shape)
        x = torch.cat([x, condition], dim=1)  # Concatenate along the channel dimension
        # print(x.shape)
        h = self.encoder_conv(x)
        # print(h.shape)
        h_flat = torch.flatten(h, start_dim=1) # Flatten before fully connected layer
        # print(h_flat.shape)
        # h = h.view(h.size(0), -1)  # Flatten before fully connected layer
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)
    
    # Reparameterization trick: Sample from the latent space
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # random noise of the same shape as std
        print((mu + eps * std).shape)
        return mu + eps * std
    
    # Decoder: Takes the latent vector and condition to reconstruct the image
    def decode(self, z, condition):
        condition_encoded = self.condition_conv(condition)
        # condition_encoded = torch.mean(condition_encoded, dim=1)
        print(condition_encoded.shape)
        condition_flat = torch.flatten(condition_encoded, start_dim=1)
        print(z.shape, condition_flat.shape)
        z = torch.cat([z, condition_flat], dim=1)  # Concatenate latent vector and condition
        print(z.shape)
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 16, 16)  # Reshape to feature maps
        return self.decoder_conv(h)
    
    # Forward pass through the model
    def forward(self, x, condition_sequence):
        condition = self.LSTM_model(condition_sequence)
        condition = condition.unsqueeze(1)
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar
