import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """
    Convert input images to patch embeddings with position and temporal embeddings.
    """
    def __init__(self, img_size=64, patch_size=16, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
        
        # Positional embedding for spatial position
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches, embed_dim))
        
        # Temporal embedding for sequence position
        self.temporal_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

class Attention(nn.Module):
    """
    Multi-head self-attention module.
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MLP.
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer for sequence prediction with temporal embeddings.
    Optimized for segmentation tasks by using all patch embeddings.
    """
    def __init__(
        self,
        img_size=64,
        patch_size=16,
        in_channels=1,
        max_seq_length=5,
        num_classes=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.max_seq_length = max_seq_length
        self.num_patches = (img_size // patch_size) ** 2
        self.num_classes = num_classes
        
        # Patch embedding layer
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding for spatial positions
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Temporal embedding for sequence positions
        self.temporal_embed = nn.Parameter(torch.randn(1, max_seq_length, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, height, width]
            
        Returns:
            Output tensor of shape [batch_size, num_classes, height, width]
        """
        B, seq_len, H, W = x.shape
        
        # Note: Input is already padded by custom_collate_fn, no need to pad again
        
        # Process last frame in the sequence for segmentation
        # For segmentation prediction, we primarily care about the last frame
        frame = x[:, -1]  # [B, H, W]
        frame = frame.unsqueeze(1)  # Add channel dimension [B, 1, H, W]
        
        # Create patch embeddings
        patches = self.patch_embed(frame)  # [B, embed_dim, h, w]
        h_patches = w_patches = H // self.patch_size
        
        # Reshape to sequence of patches
        patches = rearrange(patches, 'b c h w -> b (h w) c')  # [B, n_patches, embed_dim]
        
        # Add position embeddings
        patches = patches + self.pos_embed
        
        # Process the patch sequence through transformer
        x = self.dropout(patches)
        x = self.blocks(x)
        x = self.norm(x)  # [B, n_patches, embed_dim]
        
        # Apply segmentation head to each patch
        x = self.seg_head(x)  # [B, n_patches, num_classes]
        
        # Reshape to 2D spatial layout
        x = x.reshape(B, h_patches, w_patches, self.num_classes)
        x = x.permute(0, 3, 1, 2)  # [B, num_classes, h_patches, w_patches]
        
        # Upsample to original image resolution
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x 