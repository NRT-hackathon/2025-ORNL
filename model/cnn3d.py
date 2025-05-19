import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    """
    3D CNN model for sequence prediction with improved segmentation capabilities.
    
    This model takes a sequence of images and predicts a segmentation mask.
    It uses skip connections and improved upsampling for better segmentation.
    """
    
    def __init__(self, max_seq_length=5, in_channels=1, out_channels=2):
        """
        Initialize the 3D CNN model.
        
        Args:
            max_seq_length: Maximum sequence length
            in_channels: Number of input channels per frame
            out_channels: Number of output channels (classes)
        """
        super(CNN3D, self).__init__()
        
        self.max_seq_length = max_seq_length
        
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Temporal compression
        self.temp_conv = nn.Conv3d(128, 128, kernel_size=(self.max_seq_length, 1, 1))
        
        # Decoder path with skip connections
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),  # 192 = 64 (from upconv) + 128 (from skip)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),  # 96 = 32 (from upconv) + 64 (from skip)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, padding=1),  # 48 = 16 (from upconv) + 32 (from skip)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )
        
        # Final activation
        self.final_activation = nn.Sigmoid() if out_channels == 1 else nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        batch_size, seq_len, height, width = x.shape
        
        # Add channel dimension
        x = x.unsqueeze(1)
        
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # Temporal compression
        if seq_len < self.max_seq_length:
            x = F.adaptive_max_pool3d(x, (1, x.size(3), x.size(4)))
        else:
            x = self.temp_conv(x)
        
        # Remove temporal dimension
        x = x.squeeze(2)
        
        # Decoder path with skip connections
        x = self.upconv1(x)
        # Take the last frame's features for skip connection
        skip3 = enc3[:, :, -1]  # [batch_size, 128, height, width]
        x = torch.cat([x, skip3], dim=1)  # [batch_size, 192, height, width]
        x = self.dec1(x)  # [batch_size, 64, height, width]
        
        x = self.upconv2(x)  # [batch_size, 32, height*2, width*2]
        skip2 = enc2[:, :, -1]  # [batch_size, 64, height*2, width*2]
        x = torch.cat([x, skip2], dim=1)  # [batch_size, 96, height*2, width*2]
        x = self.dec2(x)  # [batch_size, 32, height*2, width*2]
        
        x = self.upconv3(x)  # [batch_size, 16, height*4, width*4]
        skip1 = enc1[:, :, -1]  # [batch_size, 32, height*4, width*4]
        x = torch.cat([x, skip1], dim=1)  # [batch_size, 48, height*4, width*4]
        x = self.dec3(x)  # [batch_size, out_channels, height*4, width*4]
        
        # Apply final activation
        x = self.final_activation(x)
        
        return x 