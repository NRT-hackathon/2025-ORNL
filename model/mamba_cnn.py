import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightCNNBackbone(nn.Module):
    """Lightweight CNN backbone with reduced complexity"""
    def __init__(self, in_channels=1, base_channels=32):  # Reduced base channels
        super().__init__()
        
        # Simplified encoder with depthwise separable convolutions
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels),  # Depthwise
            nn.Conv2d(base_channels, base_channels, 1),  # Pointwise
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1, groups=base_channels),
            nn.Conv2d(base_channels*2, base_channels*2, 1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU()
        )
        
        # Simplified decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1, groups=base_channels),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU()
        )
        
        # Efficient upsampling
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*2, base_channels, 1)
        )
        
    def forward(self, x):
        # Encoder path
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        
        # Decoder path with skip connection
        x = self.up1(x2)
        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)
        
        return x

class MemoryEfficientMambaBlock(nn.Module):
    """Memory-efficient Mamba block with sequential processing"""
    def __init__(self, d_model, d_state=8, expand=2, dropout=0.1):  # Reduced d_state
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Shared projection weights
        self.proj = nn.Linear(d_model, d_model * expand)
        self.state_proj = nn.Linear(d_model * expand, d_state)
        self.out_proj = nn.Linear(d_state, d_model)
        
        # State initialization
        self.initial_state = nn.Parameter(torch.zeros(1, d_state))
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch, seq_len, height*width, d_model]
        batch, seq_len, spatial, d_model = x.shape
        
        # Initialize output tensor
        y = torch.zeros_like(x)
        
        # Process each frame sequentially
        for i in range(seq_len):
            # Process current frame
            frame = x[:, i]  # [batch, spatial, d_model]
            frame_norm = self.norm(frame)
            
            # Project and update state
            proj = self.proj(frame_norm)  # [batch, spatial, d_model*expand]
            proj = F.silu(proj)
            
            # Update state sequentially
            state_updates = self.state_proj(proj)  # [batch, spatial, d_state]
            
            # Initialize state for this frame
            current_state = self.initial_state.expand(batch, -1)  # [batch, d_state]
            current_state = F.silu(current_state.unsqueeze(1) + state_updates)  # [batch, spatial, d_state]
            
            # Project to output
            out = self.out_proj(current_state)  # [batch, spatial, d_model]
            out = self.dropout(out)
            
            # Reshape and store output
            y[:, i:i+1] = out.unsqueeze(1)  # Add sequence dimension back
        
        return x + y

class OptimizedMambaCNN(nn.Module):
    """Memory-optimized Mamba model with CNN backbone"""
    def __init__(
        self,
        img_size=64,
        in_channels=1,
        max_seq_length=5,
        num_classes=2,
        base_channels=32,  # Reduced base channels
        d_state=8,  # Reduced state dimension
        depth=2,
        dropout=0.1,
        use_checkpointing=True
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.use_checkpointing = use_checkpointing
        
        # Lightweight CNN backbone
        self.backbone = LightweightCNNBackbone(in_channels, base_channels)
        
        # Efficient feature projection
        self.feature_proj = nn.Conv2d(base_channels, base_channels, 1)
        
        # Memory-efficient Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MemoryEfficientMambaBlock(base_channels, d_state, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Lightweight segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels),
            nn.Conv2d(base_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, num_classes, 1)
        )
        
    def _process_mamba_block(self, features, block):
        """Helper function for gradient checkpointing"""
        return block(features)
        
    def forward(self, x):
        """
        Memory-efficient forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes, height, width]
        """
        batch_size, seq_len, height, width = x.shape
        
        # Process frames sequentially to save memory
        features = []
        for i in range(seq_len):
            frame = x[:, i].unsqueeze(1)  # [batch_size, 1, height, width]
            
            # Extract features with checkpointing
            if self.use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                frame_features = checkpoint(self.backbone, frame)
            else:
                frame_features = self.backbone(frame)
            
            frame_features = self.feature_proj(frame_features)
            frame_features = frame_features.flatten(2).transpose(1, 2)
            features.append(frame_features)
        
        # Stack features
        features = torch.stack(features, dim=1)
        
        # Process through Mamba blocks with checkpointing
        for block in self.mamba_blocks:
            if self.use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                features = checkpoint(self._process_mamba_block, features, block)
            else:
                features = block(features)
        
        # Get last frame's features
        last_features = features[:, -1]
        last_features = last_features.transpose(1, 2).reshape(batch_size, -1, height, width)
        
        # Generate segmentation map
        output = self.seg_head(last_features)
        
        return output 