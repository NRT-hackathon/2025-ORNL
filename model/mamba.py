import torch
import torch.nn as nn
import torch.nn.functional as F

class ScanningMLP(nn.Module):
    """Simple MLP layer with scan-like sequential processing"""
    def __init__(self, d_model, d_state, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        
        # Project input to higher dimension
        self.in_proj = nn.Linear(d_model, d_model * expand)
        
        # State projection matrices
        self.state_proj = nn.Linear(d_model * expand, d_state)
        self.out_proj = nn.Linear(d_state, d_model)
        
        # State initialization
        self.initial_state = nn.Parameter(torch.zeros(1, d_state))
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        batch, seq_len, _ = x.shape
        
        # Project to higher dimension
        x_proj = self.in_proj(x)  # [batch, seq_len, d_model*expand]
        x_proj = F.silu(x_proj)
        
        # Initialize state
        state = self.initial_state.expand(batch, -1)  # [batch, d_state]
        
        outputs = []
        # Scan through sequence
        for i in range(seq_len):
            # Update state
            state_input = x_proj[:, i]  # [batch, d_model*expand]
            state_update = self.state_proj(state_input)  # [batch, d_state]
            state = F.silu(state + state_update)
            
            # Project to output
            out = self.out_proj(state)  # [batch, d_model]
            outputs.append(out)
        
        # Stack outputs
        return torch.stack(outputs, dim=1)  # [batch, seq_len, d_model]

class SelectiveScanLayer(nn.Module):
    """Approximation of the selective scan operation from Mamba"""
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM Parameters
        self.A_log = nn.Parameter(torch.randn(1, 1, d_state))
        self.D = nn.Parameter(torch.randn(d_state, d_model))
        
        # Input projections
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.x_proj = nn.Linear(d_model, d_state)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Forward pass of the selective scan.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        # Store original input for residual and get shape
        batch, seq_len, d_model = x.shape
        residual = x
        
        # Apply layer norm
        x_norm = self.norm(x)
        
        # Project inputs
        xz = self.in_proj(x_norm)  # [batch, seq_len, 2*d_model]
        x_proj, z = torch.chunk(xz, 2, dim=-1)  # each [batch, seq_len, d_model]
        
        # Apply activation
        x_proj = F.silu(x_proj)
        z = torch.sigmoid(z)
        
        # Project x for state update
        u = self.x_proj(x_proj)  # [batch, seq_len, d_state]
        
        # Convert parameters
        A = -torch.exp(self.A_log)  # [1, 1, d_state]
        D = self.D  # [d_state, d_model]
        
        # Initialize state
        h = torch.zeros(batch, self.d_state, device=x.device)  # [batch, d_state]
        
        outputs = []
        # Scan through sequence
        for i in range(seq_len):
            # SSM recurrence
            h = torch.exp(A) * h + u[:, i]
            out = h @ D  # [batch, d_model]
            out = out * z[:, i]  # Apply gating
            outputs.append(out)
        
        # Stack outputs and apply output projection
        y = torch.stack(outputs, dim=1)  # [batch, seq_len, d_model]
        y = self.out_proj(y)
        y = self.dropout(y)
        
        # Debug: Print shapes before addition
        # import sys
        # print(f"Residual shape: {residual.shape}, y shape: {y.shape}", file=sys.stderr)
        
        # Check for dimension match and reshape if needed
        if residual.shape != y.shape:
            # Ensure y has the same shape as residual through adaptive reshaping
            y = y.view(batch, seq_len, d_model)
        
        # Residual connection
        return residual + y

class MambaBlock(nn.Module):
    """Mamba block with convolution and selective scan"""
    def __init__(self, d_model=256, d_state=16, expand=2, dropout=0.1, kernel_size=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Local convolution for capturing local patterns
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1),
            groups=d_model
        )
        
        # Selective scan layer
        self.scan = SelectiveScanLayer(d_model, d_state, dropout)
        
        # Feedforward
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expand, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass of the Mamba block.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        # Store original input for residual
        residual = x
        
        # Apply convolution
        x_norm = self.norm1(x)
        x_conv = x_norm.transpose(1, 2)  # [batch, d_model, seq_len]
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_model]
        x_conv = x_conv[:, :x.size(1)]  # Truncate padding
        
        # First residual
        x = residual + x_conv
        
        # Apply selective scan (with its own residual connection internally)
        x = self.scan(x)
        
        # Apply MLP with residual
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp
        
        return x

class MambaUnet(nn.Module):
    """
    Mamba-based model for sequence prediction that uses Mamba blocks
    for sequential processing and a U-Net like architecture for spatial processing.
    Memory-optimized version with gradient checkpointing.
    """
    def __init__(
        self,
        img_size=64,
        in_channels=1,
        max_seq_length=5,
        num_classes=2,
        d_model=64,  # Reduced from 256
        d_state=16,
        depth=2,
        expand=2,
        dropout=0.1,
        use_checkpointing=True
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        self.use_checkpointing = use_checkpointing
        
        # Input embedding - reduced parameters
        self.embedding = nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1)
        
        # Positional embedding for the flattened spatial dimension (H*W)
        self.pos_embed = nn.Parameter(torch.zeros(1, img_size * img_size, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Temporal embedding
        self.time_embed = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        nn.init.normal_(self.time_embed, std=0.02)
        
        # Combined spatial-temporal Mamba blocks to reduce memory
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Spatial processing (U-Net style) - with reduced channels
        # Encoder
        self.down1 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(d_model, d_model * 2, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(d_model * 2, d_model * 2, kernel_size=3, padding=1)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(d_model * 2, d_model, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(d_model * 2, d_model, kernel_size=4, stride=2, padding=1)
        
        # Final prediction layer
        self.final = nn.Conv2d(d_model * 2, num_classes, kernel_size=1)
        
    def _process_frame(self, frame, time_idx):
        """Process a single frame - helps with checkpointing"""
        B = frame.shape[0]
        H, W = self.img_size, self.img_size
        
        # Embed frame
        frame_emb = self.embedding(frame)  # [B, d_model, H, W]
        
        # Reshape to sequence format
        frame_emb_flat = frame_emb.reshape(B, -1, H * W).transpose(1, 2)  # [B, H*W, d_model]
        
        # Add positional embedding
        frame_emb_flat = frame_emb_flat + self.pos_embed
        
        # Add temporal embedding
        temp_idx = time_idx % self.max_seq_length
        frame_emb_flat = frame_emb_flat + self.time_embed[:, temp_idx:temp_idx+1]
        
        return frame_emb_flat
    
    def _run_mamba_block(self, x, block):
        """Helper for checkpointing a Mamba block"""
        return block(x)
        
    def forward(self, x):
        """
        Forward pass of the model with memory optimizations.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, height, width]
            
        Returns:
            Tensor of shape [batch_size, num_classes, height, width]
        """
        B, seq_len, H, W = x.shape
        
        # Process last frame only for spatial structure, others contribute temporal info
        last_frame = x[:, -1].unsqueeze(1)  # [B, 1, H, W]
        
        # Process the last frame fully
        last_emb = self._process_frame(last_frame, seq_len-1)
        
        # Initialize the combined embedding with the last frame
        combined_emb = last_emb
        
        # Process previous frames with gradient checkpointing to save memory
        # Process frames in reverse order (more recent frames first)
        for i in range(seq_len-2, -1, -1):
            # Extract current frame
            frame = x[:, i].unsqueeze(1)  # [B, 1, H, W]
            
            # Process frame with checkpointing to save memory
            if self.use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                frame_emb = checkpoint(self._process_frame, frame, i)
            else:
                frame_emb = self._process_frame(frame, i)
            
            # Fuse information with a simple element-wise operation
            # This is memory-efficient and effective for temporal fusion
            combined_emb = combined_emb + frame_emb * 0.5  # Weight recent frames more
        
        # Process through Mamba blocks with checkpointing
        for i, block in enumerate(self.mamba_blocks):
            if self.use_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                combined_emb = checkpoint(self._run_mamba_block, combined_emb, block)
            else:
                combined_emb = block(combined_emb)
                
        # Reshape to spatial format for U-Net
        spatial_emb = combined_emb.transpose(1, 2).reshape(B, -1, H, W)  # [B, d_model, H, W]
        
        # U-Net processing with reduced intermediate storage
        x1 = F.relu(self.down1(spatial_emb))
        x2 = F.relu(self.down2(x1))
        x2 = F.relu(self.bottleneck(x2))
        
        x = F.relu(self.up1(x2))
        x = torch.cat([x, x1], dim=1)
        x1 = None  # Free memory
        
        x = F.relu(self.up2(x))
        x = torch.cat([x, spatial_emb], dim=1)
        spatial_emb = None  # Free memory
        
        # Final prediction
        x = self.final(x)
        
        return x 