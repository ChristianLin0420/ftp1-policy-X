import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import pathlib
import logging
from typing import Dict, List, Tuple
from openpi.models_pytorch.pi0_pytorch import get_safe_dtype
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
import safetensors
import safetensors.torch
from openpi.models_pytorch.t3_tactile_encoder import (
    T3SharedChunkConfig,
    TransformerTrunk,
    ViTEncoder,
    T3TactileEncoderConfig,
)
from openpi.models_pytorch.ftp1_model_config import FTP1TactileTokenizerConfig
from openpi.tactile_identity import build_tactile_sensor_type_shape_key_from_config
from openpi.tactile_identity import canonicalize_tactile_encoder_type
from openpi.tactile_identity import canonicalize_tactile_sensor_name
from openpi.tactile_identity import get_tactile_encoder_shape_from_config_shape
from openpi.tactile_identity import get_tactile_encoder_shape_from_data_shape


class FourierStateEncoder(nn.Module):
    def __init__(self, state_dim, output_dim, fourier_dim=8, min_period=1e-3, max_period=1.0):
        """Fourier State Encoder.
        
        Args:
            state_dim: Dimension of input state
            output_dim: Dimension of output embedding
            fourier_dim: Dimension of Fourier encoding for each state dimension
            min_period: Minimum period for Fourier encoding
            max_period: Maximum period for Fourier encoding
        """
        super().__init__()
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.fourier_dim = fourier_dim
        self.min_period = min_period
        self.max_period = max_period
        
        # MLP to map concatenated [state, fourier_encoding] to output_dim
        # Input: state_dim + state_dim * fourier_dim
        # Output: output_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + state_dim * fourier_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )
    
    def _fourier_encode(self, x, dimension, min_period, max_period):
        """Apply Fourier encoding to a tensor.
        
        Args:
            x: (B, T, state_dim) or (B, state_dim) tensor
            dimension: Dimension of Fourier encoding
            min_period: Minimum period
            max_period: Maximum period
            
        Returns:
            (B, T, state_dim, dimension) or (B, state_dim, dimension) tensor (same shape as input with added dimension)
        """
        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        
        device = x.device
        dtype = get_safe_dtype(torch.float32, device.type)
        
        # Create frequency bands
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction
        
        # Compute scaling factor: 1 / period * 2 * pi
        scaling_factor = 1.0 / period * 2 * math.pi  # (dimension // 2,)
        
        # Apply to each dimension of state
        # x shape: (B, T, state_dim) or (B, state_dim)
        # scaling_factor shape: (dimension // 2,)
        # We need to broadcast: x.unsqueeze(-1) * scaling_factor
        # For (B, T, state_dim): unsqueeze(-1) -> (B, T, state_dim, 1), then * scaling_factor -> (B, T, state_dim, dimension // 2)
        # For (B, state_dim): unsqueeze(-1) -> (B, state_dim, 1), then * scaling_factor -> (B, state_dim, dimension // 2)
        sin_input = x.unsqueeze(-1) * scaling_factor  # (B, T, state_dim, dimension // 2) or (B, state_dim, dimension // 2)
        
        # Concatenate sin and cos
        fourier_enc = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)  # (B, T, state_dim, dimension) or (B, state_dim, dimension)
        
        return fourier_enc
    
    def forward(self, state):
        """Forward pass.
        
        Args:
            state: (B, T, state_dim) tensor
            
        Returns:
            (B, T, output_dim) tensor
        """
        # Apply Fourier encoding to each dimension of state
        # Shape: (B, T, state_dim) -> (B, T, state_dim, fourier_dim)
        fourier_enc = self._fourier_encode(state, self.fourier_dim, self.min_period, self.max_period)
        
        # Flatten Fourier encoding: (B, T, state_dim, fourier_dim) -> (B, T, state_dim * fourier_dim)
        fourier_enc_flat = fourier_enc.reshape(state.shape[0], state.shape[1], -1)
        
        # Concatenate original state and Fourier encoding
        # Shape: (B, T, state_dim + state_dim * fourier_dim)
        # Note: fourier_enc_flat is float64 (for precision in Fourier encoding), state might be float32
        combined = torch.cat([state, fourier_enc_flat], dim=2)
        
        # Convert to float32 after normalization (Fourier encoding is a form of normalization)
        # This ensures all normalized data is float32, as required
        combined = combined.to(dtype=torch.float32)
        
        # Pass through MLP
        # Shape: (B, T, state_dim + state_dim * fourier_dim) -> (B, T, output_dim)
        output = self.mlp(combined.reshape(-1, combined.shape[-1])).reshape(state.shape[0], state.shape[1], -1)
        
        return output


class MatrixCNNEncoder(nn.Module):
    """Simple CNN encoder for small matrix data (H, W) or (H, W, D).
    First applies Fourier encoding to expand D dimension to D2, then applies CNN."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_dim: int,
        num_layers: int = 2,
        fourier_dim: int = 8,
        min_period: float = 1e-3,
        max_period: float = 1.0,
    ):
        """
        Args:
            input_shape: (H, W) or (H, W, D) shape
            output_dim: Output token dimension
            hidden_dim: Hidden dimension for CNN layers
            num_layers: Number of CNN layers
            fourier_dim: Dimension of Fourier encoding for each D dimension
            min_period: Minimum period for Fourier encoding
            max_period: Maximum period for Fourier encoding
        """
        super().__init__()
        hidden_dim = output_dim * 2
        self.input_shape = input_shape
        self.fourier_dim = fourier_dim
        self.min_period = min_period
        self.max_period = max_period
        
        # Determine input channels (D dimension)
        if len(input_shape) == 2:
            # (H, W) -> treat as (1, H, W), will expand to (fourier_dim, H, W)
            d_dim = 1
            h, w = input_shape
        else:  # len(input_shape) == 3
            # (H, W, D) -> will expand to (D * fourier_dim, H, W)
            d_dim = input_shape[2]
            h, w = input_shape[0], input_shape[1]
        
        self.d_dim = d_dim
        
        # After Fourier encoding, we concatenate original and Fourier encoding
        # So expanded_channels = d_dim (original) + d_dim * fourier_dim (Fourier encoding)
        expanded_channels = d_dim + d_dim * fourier_dim
        
        # Build CNN layers (input is now expanded_channels after Fourier encoding)
        layers = []
        current_channels = expanded_channels
        current_h, current_w = h, w
        
        for i in range(num_layers):
            # Conv layer
            layers.append(nn.Conv2d(
                current_channels,
                hidden_dim,
                kernel_size=3,
                padding=1,
                stride=1
            ))
            layers.append(nn.GELU())
            
            # Optional pooling for larger matrices
            if current_h > 4 or current_w > 4:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_h = current_h // 2
                current_w = current_w // 2
            
            current_channels = hidden_dim
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate flattened size after CNN
        # Run a dummy forward to get output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, expanded_channels, h, w)
            dummy_output = self.cnn(dummy_input)
            flattened_size = dummy_output.numel() // dummy_output.shape[0]
        
        # Final projection to output_dim
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def _fourier_encode(self, x, dimension, min_period, max_period):
        """Apply Fourier encoding to the D dimension of a matrix.
        
        Args:
            x: (B, D, H, W) tensor
            dimension: Dimension of Fourier encoding for each D dimension
            min_period: Minimum period
            max_period: Maximum period
            
        Returns:
            (B, D * dimension, H, W) tensor
        """
        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        
        device = x.device
        dtype = get_safe_dtype(torch.float32, device.type)
        
        B, D, H, W = x.shape
        
        # Create frequency bands
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction
        
        # Compute scaling factor: 1 / period * 2 * pi
        scaling_factor = 1.0 / period * 2 * math.pi  # (dimension // 2,)
        
        # Apply to each D dimension
        # x shape: (B, D, H, W)
        # Reshape to (B, D, H*W) for easier processing
        x_flat = x.reshape(B, D, H * W)  # (B, D, H*W)
        
        # Apply Fourier encoding: (B, D, H*W) -> (B, D, H*W, dimension // 2)
        sin_input = x_flat.unsqueeze(-1) * scaling_factor  # (B, D, H*W, dimension // 2)
        
        # Concatenate sin and cos: (B, D, H*W, dimension // 2) -> (B, D, H*W, dimension)
        fourier_enc = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)  # (B, D, H*W, dimension)
        
        # Reshape: (B, D, H*W, dimension) -> (B, D, dimension, H*W) -> (B, D * dimension, H, W)
        fourier_enc = fourier_enc.permute(0, 1, 3, 2).reshape(B, D * dimension, H, W)
        
        return fourier_enc
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W) or (B, H, W, D) or (B, D, H, W) tensor
        Returns:
            (B, output_dim) tensor
        """
        # Ensure input is (B, D, H, W)
        if x.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            # Check if it's (B, H, W, D) or (B, D, H, W)
            # If last dimension matches expected D dimension, it's (B, H, W, D)
            if x.shape[-1] == self.d_dim and len(self.input_shape) == 3:
                # (B, H, W, D) -> (B, D, H, W)
                x = x.permute(0, 3, 1, 2)
            # Otherwise assume it's already (B, D, H, W)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Now x is (B, D, H, W)
        B, D, H, W = x.shape
        assert D == self.d_dim, f"Expected D={self.d_dim}, got {D}"
        
        # Apply Fourier encoding: (B, D, H, W) -> (B, D * fourier_dim, H, W)
        x_fourier = self._fourier_encode(x, self.fourier_dim, self.min_period, self.max_period)
        x_fourier = x_fourier.to(torch.float32)
        
        # Concatenate original data and Fourier encoding: (B, D, H, W) + (B, D * fourier_dim, H, W) -> (B, D + D * fourier_dim, H, W)
        x_combined = torch.cat([x, x_fourier], dim=1)  # Concatenate along channel dimension
        x_combined = x_combined.to(torch.float32)
        
        # CNN: (B, D + D * fourier_dim, H, W) -> (B, hidden_dim, H', W')
        x = self.cnn(x_combined)
        
        # Project to output_dim: (B, hidden_dim, H', W') -> (B, output_dim)
        x = self.proj(x)
        
        return x


class SharedImageChunkEncoder(nn.Module):
    """Encapsulates shared_image_chunk_encoder and image_proj together.
    They are always created or not created together.
    """
    def __init__(self, embed_dim: int, token_dim: int, shared_chunk_config=None,
                 load_t3_pretrained_checkpoint=False, cache_t3_pretrained_checkpoint_dir=None,
                 tokens_per_area: int = 1, spatial_pool_grid: int = 3):
        """
        Args:
            embed_dim: Embedding dimension (default 768)
            token_dim: Output token dimension
            shared_chunk_config: T3SharedChunkConfig (uses default if None)
            tokens_per_area: 1 keeps the legacy CLS-only readout; 1 + spatial_pool_grid**2
                emits CLS plus a pooled patch grid (PACT).
            spatial_pool_grid: side length of the adaptive pool over the ViT patch grid.
        """
        super().__init__()
        if shared_chunk_config is None:
            shared_chunk_config = T3SharedChunkConfig()
        self.tokens_per_area = tokens_per_area
        self.spatial_pool_grid = spatial_pool_grid
        
        # For shared_chunk_encoder, we use a generic name since it's shared across all sensors
        # The pretrained checkpoint loading for shared_chunk is not implemented yet
        # (T3 pretrained checkpoints are for individual sensor encoders, not shared chunk)
        self.shared_chunk_encoder = TransformerTrunk(
            embed_dim=shared_chunk_config.embed_dim,
            depth=shared_chunk_config.depth,
            num_heads=shared_chunk_config.num_heads,
            mlp_ratio=shared_chunk_config.mlp_ratio,
            pooling_type=shared_chunk_config.pooling_type,
            load_t3_pretrained_checkpoint=load_t3_pretrained_checkpoint,  # Shared chunk doesn't use T3 pretrained checkpoints
            cache_t3_pretrained_checkpoint_dir=cache_t3_pretrained_checkpoint_dir,
            tokenizer_name=None,
            sensor_name=None
        )
        # Create projector from embed_dim to token_dim
        self.image_proj = nn.Linear(shared_chunk_config.embed_dim, token_dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, num_patches+1, embed_dim) tensor from ViT encoder
        Returns:
            (B, tokens_per_area, token_dim) tensor after shared chunk encoding and projection.
        """
        # Apply shared chunk encoder: (B, num_patches+1, embed_dim) -> (B, num_patches+1, embed_dim)
        tokens = self.shared_chunk_encoder(tokens)

        if self.tokens_per_area == 1:
            # Legacy FTP-1 readout: keep only CLS, discard every patch token.
            readout = tokens[:, :1, :]  # (B, 1, embed_dim)
        else:
            cls_token = tokens[:, :1, :]  # (B, 1, embed_dim)
            patch_tokens = tokens[:, 1:, :]  # (B, num_patches, embed_dim)
            bsz, num_patches, embed_dim = patch_tokens.shape
            side = int(round(math.sqrt(num_patches)))
            if side * side != num_patches:
                raise ValueError(
                    f"Expected a square patch grid for spatial pooling, got {num_patches} patches."
                )
            # (B, P, E) -> (B, E, side, side) -> pool -> (B, E, g, g) -> (B, g*g, E)
            grid = patch_tokens.transpose(1, 2).reshape(bsz, embed_dim, side, side)
            pooled = F.adaptive_avg_pool2d(grid, (self.spatial_pool_grid, self.spatial_pool_grid))
            pooled = pooled.flatten(2).transpose(1, 2)  # (B, g*g, E)
            readout = torch.cat([cls_token, pooled], dim=1)  # (B, 1 + g*g, E)

        # Project from embed_dim to token_dim.
        return self.image_proj(readout)  # (B, tokens_per_area, token_dim)


class TactileDataEncoder(nn.Module):
    """Encoder for a single tactile data type based on its shape D."""
    def __init__(
        self, 
        tokenizer_name,
        sensor_name,
        encoder_type,
        data_shape: Tuple[int, ...], 
        token_dim: int,
        tokenizer_config=None,
        image_encoder_config=None,
        load_t3_pretrained_checkpoint=False,
        cache_t3_pretrained_checkpoint_dir=None,
    ):
        """
        Args:
            tokenizer_name: Tokenizer name (e.g., "gripper_image_224_224_3") for identification
            sensor_name: Sensor name (e.g., "gripper", "digit", "mini") for T3 pretrained checkpoint loading
            encoder_type: Type of encoder ('binary', 'state', 'image', 'matrix')
            data_shape: Shape of D (e.g., (72,), (1,), (224, 224, 3))
            token_dim: Output token dimension
            tokenizer_config: FTP1TactileTokenizerConfig for encoder parameters
            image_encoder_config: T3TactileEncoderConfig for image encoder (uses default if None)
        """
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.sensor_name = sensor_name
        self.encoder_type = encoder_type
        self.data_shape = data_shape
        self.token_dim = token_dim
        self.tokenizer_config = tokenizer_config if tokenizer_config is not None else FTP1TactileTokenizerConfig()
        self.load_t3_pretrained_checkpoint = self.tokenizer_config.load_t3_pretrained_checkpoint
        self.cache_t3_pretrained_checkpoint_dir = self.tokenizer_config.cache_t3_pretrained_checkpoint_dir
        
        # Use default config if not provided
        if image_encoder_config is None:
            image_encoder_config = T3TactileEncoderConfig()
        self.image_encoder_config = image_encoder_config
        
        if self.encoder_type == 'binary':
            # D=(1,): Binary embedding (0/1)
            self.embedding = nn.Embedding(2, token_dim)  # 0 and 1
        elif self.encoder_type == 'state':
            # D=(d,): Use FourierStateEncoder for better representation
            input_dim = data_shape[0]
            # Use config if provided, otherwise use defaults
            fourier_dim = tokenizer_config.fourier_dim
            min_period = tokenizer_config.fourier_min_period
            max_period = tokenizer_config.fourier_max_period
            self.fourier_encoder = FourierStateEncoder(
                state_dim=input_dim,
                output_dim=token_dim,
                fourier_dim=fourier_dim,
                min_period=min_period,
                max_period=max_period,
            )
        elif self.encoder_type == 'matrix':
            # D is matrix shape: (H, W) or (H, W, D)
            # First applies Fourier encoding to expand D dimension, then uses CNN
            # Get config parameters if available, otherwise use defaults
            # Note: hidden_dim is automatically set to output_dim * 2 in MatrixCNNEncoder
            cnn_num_layers = tokenizer_config.cnn_num_layers if tokenizer_config else 2
            fourier_dim = tokenizer_config.fourier_dim if tokenizer_config else 8
            min_period = tokenizer_config.fourier_min_period if tokenizer_config else 1e-3
            max_period = tokenizer_config.fourier_max_period if tokenizer_config else 1.0
            
            self.cnn_encoder = MatrixCNNEncoder(
                input_shape=data_shape,
                output_dim=token_dim,
                num_layers=cnn_num_layers,
                fourier_dim=fourier_dim,
                min_period=min_period,
                max_period=max_period,
            )
        elif self.encoder_type == 'image':
            # D is image shape: (H, W) or (H, W, 3) or (3, H, W) or (1, H, W) or (H, W, 1)
            # Lightweight Vision Transformer (similar to SigLIP but much smaller)
            if len(data_shape) == 2:
                in_channels = 1
                h, w = data_shape
            else:
                if data_shape[0] == 3:
                    in_channels = 3
                    h, w = data_shape[1], data_shape[2]
                else:
                    in_channels = data_shape[2]
                    h, w = data_shape[0], data_shape[1]
        
            
            patch_size = image_encoder_config.patch_size     # Only patch_size needs to be set
            # Ensure patch_size divides image size
            if h % patch_size != 0 or w % patch_size != 0:
                raise ValueError(f"Image size {h}x{w} is not divisible by patch size {patch_size}")
            
            # Create image encoder instance with default embed_dim (768)
            self.vit_encoder = ViTEncoder(
                tokenizer_name=self.tokenizer_name,
                sensor_name=self.sensor_name,
                img_size=(h, w),
                patch_size=image_encoder_config.patch_size,
                in_chans=in_channels,
                embed_dim=image_encoder_config.encoder_embed_dim,  # Use default 768
                depth=image_encoder_config.encoder_depth,  # Use default 3
                num_heads=image_encoder_config.encoder_heads,  # Use default 12
                mlp_ratio=image_encoder_config.mlp_ratio,  # Use default 4
                load_t3_pretrained_checkpoint=load_t3_pretrained_checkpoint,
                cache_t3_pretrained_checkpoint_dir=cache_t3_pretrained_checkpoint_dir
            )
            
        else:
            raise ValueError(f"Unsupported data type: {encoder_type} with data shape: {data_shape}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, *D) tensor
        Returns:
            (B, T, N, token_dim) tensor for all encoder types
        """
        B, T, N = x.shape[:3]
        
        if self.encoder_type == 'binary':
            # Check if values are 0/1
            assert x.shape[-1] == 1, "Binary encoder expects a single channel"
            # Check values are in [0, 1] before converting to long
            x_float = x.squeeze(-1)  # (B, T, N)
            assert (x_float >= (0.0 - 1e-8)).all() and (x_float <= (1.0 + 1e-8)).all(), "Binary encoder expects values in [0, 1]"
            x = x_float.long()  # (B, T, N)
            # Embed: (B, T, N) -> (B, T, N, token_dim)
            tokens = self.embedding(x)
        elif self.encoder_type == 'state':
            # Reshape: (B, T, N, D) -> (B*N, T, D)
            # FourierStateEncoder expects (B, T, state_dim), so we need to reshape
            x_reshaped = x.reshape(B * N, T, x.shape[-1])  # (B*N, T, D)
            if x_reshaped.dtype != torch.float32:
                x_reshaped = x_reshaped.to(torch.float32)
            # FourierStateEncoder: (B*N, T, D) -> (B*N, T, token_dim)
            tokens_reshaped = self.fourier_encoder(x_reshaped)  # (B*N, T, token_dim)
            # Reshape back: (B*N, T, token_dim) -> (B, T, N, token_dim)
            tokens = tokens_reshaped.reshape(B, N, T, self.token_dim).transpose(1, 2)  # (B, T, N, token_dim)
        elif self.encoder_type == 'image':
            # Reshape for ViT: (B, T, N, *D) -> (B*T*N, C, H, W)
            if len(self.data_shape) == 2:
                # (B, T, N, H, W) -> (B*T*N, H, W) -> (B*T*N, 1, H, W)
                x_img = x.reshape(B * T * N, self.data_shape[0], self.data_shape[1]).unsqueeze(1)
                # (B*T*N, 1, H, W) -> (B*T*N, 3, H, W)
                x_img = x_img.repeat(1, 3, 1, 1)
            elif len(self.data_shape) == 3:
                if self.data_shape[0] == 3:
                    # (B, T, N, 3, H, W) -> (B*T*N, 3, H, W)
                    x_img = x.reshape(B * T * N, 3, self.data_shape[1], self.data_shape[2])
                else:
                    # (B, T, N, H, W, 3) -> (B*T*N, H, W, 3) -> (B*T*N, 3, H, W)
                    x_img = x.reshape(B * T * N, self.data_shape[0], self.data_shape[1], self.data_shape[2])
                    x_img = x_img.permute(0, 3, 1, 2)  # (B*T*N, 3, H, W)
            else:
                raise ValueError(f"Unexpected image shape: {self.data_shape}")
            if x_img.dtype != torch.float32:
                x_img = x_img.to(torch.float32)
            
            # ViT: (B*T*N, C, H, W) -> (B*T*N, num_patches + 1, embed_dim)
            tokens_flat = self.vit_encoder(x_img)  # (B*T*N, num_patches + 1, embed_dim=768)
            
            # Image tactile always uses the shared chunk encoder in FTP1HptTactileEncoder.
            # Reshape: (B*T*N, num_patches + 1, embed_dim) -> (B, T, N, num_patches + 1, embed_dim)
            tokens = tokens_flat.reshape(B, T, N, -1, self.image_encoder_config.encoder_embed_dim)
        elif self.encoder_type == 'matrix':
            # Reshape for CNN: (B, T, N, *D) -> (B*T*N, *D)
            if len(self.data_shape) == 2:
                # (B, T, N, H, W) -> (B*T*N, H, W)
                x_matrix = x.reshape(B * T * N, self.data_shape[0], self.data_shape[1])
            elif len(self.data_shape) == 3:
                # (B, T, N, H, W, D) -> (B*T*N, H, W, D)
                x_matrix = x.reshape(B * T * N, self.data_shape[0], self.data_shape[1], self.data_shape[2])
            else:
                raise ValueError(f"Unexpected matrix shape: {self.data_shape}")
            
            if x_matrix.dtype != torch.float32:
                x_matrix = x_matrix.to(torch.float32)
            
            # CNN: (B*T*N, *D) -> (B*T*N, token_dim)
            tokens_flat = self.cnn_encoder(x_matrix)  # (B*T*N, token_dim)
            
            # Reshape: (B*T*N, token_dim) -> (B, T, N, token_dim)
            tokens = tokens_flat.reshape(B, T, N, self.token_dim)  # (B, T, N, token_dim)
        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")
        
        return tokens


class FTP1HptTactileEncoder(nn.Module):
    def __init__(self, config, token_dim):
        """
        Args:
            config: Model config containing tactile_input_config_file and tactile_tokenizer_config
            token_dim: Output token dimension
        """
        super().__init__()
        self.input_config_path = config.tactile_input_config_file
        self.token_dim = token_dim
        self.tokenizer_config = config.tactile_tokenizer_config
        self.total_num_tactile_tokens = 2 * self.tokenizer_config.single_hand_num_tactile_areas
        self.func_area_idx_embedding = nn.Embedding(self.total_num_tactile_tokens, token_dim)

        # [PACT] Tokens emitted per function area. With tokens_per_area == 1 the encoder keeps
        # FTP-1's scatter into `total_num_tactile_tokens` canonical slots. With > 1 it emits a
        # compact sequence of (num_present_areas * tokens_per_area) tokens instead, so the
        # sequence length tracks the areas that actually exist on the embodiment rather than the
        # 48-slot worst case. Area identity is still carried by `func_area_idx_embedding`, which
        # is indexed by the true function-area id, so the pretrained table is reused unchanged.
        self.tokens_per_area = self.tokenizer_config.tokens_per_area
        self.spatial_token_embedding = (
            nn.Embedding(self.tokens_per_area, token_dim) if self.tokens_per_area > 1 else None
        )
        self.contact_gating = bool(getattr(config, "contact_gating", False))
        
        # Load configuration first to check if there are image type data
        with open(self.input_config_path, 'r') as f:
            self.config = json.load(f)
        
        # Check if config contains image type data
        self.has_image_type = False
        for domain_name, domain_config in self.config.items():
            for tactile_key, tactile_config in domain_config.items():
                encoder_type = tactile_config.get('type', '')
                if encoder_type == 'image':
                    self.has_image_type = True
                    break
            if self.has_image_type:
                break
        
        # Image tactile always routes through the shared chunk encoder.
        self.shared_image_chunk_encoder = None
        if self.has_image_type:
            # Use default config without modification
            shared_chunk_config = T3SharedChunkConfig()  # embed_dim=768, depth=9, num_heads=12, mlp_ratio=4
            self.shared_image_chunk_encoder = SharedImageChunkEncoder(
                embed_dim=shared_chunk_config.embed_dim,
                token_dim=token_dim,
                shared_chunk_config=shared_chunk_config,
                load_t3_pretrained_checkpoint=self.tokenizer_config.load_t3_pretrained_checkpoint,
                cache_t3_pretrained_checkpoint_dir=self.tokenizer_config.cache_t3_pretrained_checkpoint_dir,
                tokens_per_area=self.tokens_per_area,
                spatial_pool_grid=self.tokenizer_config.spatial_pool_grid,
            )
            
            # Freeze shared chunk encoder (t3_chunk) if frozen_shared_chunk is True
            # Note: Only freeze shared_chunk_encoder, not image_proj
            if self.tokenizer_config.frozen_shared_chunk:
                for param in self.shared_image_chunk_encoder.shared_chunk_encoder.parameters():
                    param.requires_grad = False

        # Create tokenizers with key format: f"{sensor}_{type}_{*tactile_shape}".
        # All tactile groups from the same sensor, type, and per-area shape share one tokenizer.
        self.tokenizers = nn.ModuleDict()

        # Store mapping from tactile_key to config entries for runtime sensor/function-area resolution.
        self.tactile_key_info = {}

        for domain_name, domain_config in self.config.items():
            for tactile_key, tactile_config in domain_config.items():
                sensor = canonicalize_tactile_sensor_name(tactile_config['sensor'])
                function_areas = tactile_config['function_areas']
                shape = tactile_config['shape']  # [T, N, *D]
                encoder_type = canonicalize_tactile_encoder_type(tactile_config['type'])
                tactile_shape = get_tactile_encoder_shape_from_config_shape(shape)
                tokenizer_key = build_tactile_sensor_type_shape_key_from_config(sensor, encoder_type, shape)

                if tactile_key not in self.tactile_key_info:
                    self.tactile_key_info[tactile_key] = {}
                sensor_entries = self.tactile_key_info[tactile_key].setdefault(sensor, [])
                duplicate_entries = [
                    entry
                    for entry in sensor_entries
                    if tuple(entry["function_areas"]) == tuple(function_areas)
                    and tuple(entry["tactile_shape"]) == tactile_shape
                ]
                if duplicate_entries:
                    if any(entry["encoder_type"] != encoder_type for entry in duplicate_entries):
                        raise ValueError(
                            f"Ambiguous tactile config for tactile_key={tactile_key}, sensor={sensor}, "
                            f"function_areas={function_areas}, tactile_shape={tactile_shape}: "
                            f"multiple encoder types found {[entry['encoder_type'] for entry in duplicate_entries] + [encoder_type]}"
                        )
                else:
                    sensor_entries.append(
                        {
                            "function_areas": function_areas,
                            "tactile_shape": tactile_shape,
                            "encoder_type": encoder_type,
                            "tokenizer_key": tokenizer_key,
                        }
                    )

                if tokenizer_key not in self.tokenizers:
                    self.tokenizers[tokenizer_key] = TactileDataEncoder(
                        tokenizer_name=tokenizer_key,
                        sensor_name=sensor,
                        encoder_type=encoder_type,
                        data_shape=tactile_shape,
                        token_dim=token_dim,
                        tokenizer_config=self.tokenizer_config,
                        load_t3_pretrained_checkpoint=self.tokenizer_config.load_t3_pretrained_checkpoint,
                        cache_t3_pretrained_checkpoint_dir=self.tokenizer_config.cache_t3_pretrained_checkpoint_dir
                    )
        
        # Unified projection and normalization for all tactile tokens
        # This helps learn a unified representation across different tokenizers
        # Sensor type information is implicitly encoded in the tokenizer weights
        self.unified_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    def forward(self, tactiles, tactile_function_areas, tactile_sensors, tactile_contact=None):
        """
        Args:
            tactiles: dict[str, torch.Tensor] with keys like 'left_tactile_palm', 'left_tactile_fingers', etc.
                     Each tensor has shape (B, T, N, *D)
            tactile_contact: optional dict[str, torch.Tensor] of shape (B, N) bool. When
                     contact_gating is enabled, areas reported as not-in-contact keep their CLS
                     token and have their spatial tokens masked out of attention.
            tactile_function_areas: dict[str, torch.Tensor] with same keys as tactiles
                                   Each tensor has shape (B, N) with function area indices (same across batch)
            tactile_sensors: dict[str, list[str]] with same keys as tactiles
                            Each value is a list of sensor names (strings) of length B
        Returns:
            tokens: (B, T * total_num_tactile_tokens, token_dim) tensor
                    If T > 1, arranged as [t=0 tokens], [t=1 tokens], ...
            pad_masks: (B, T * total_num_tactile_tokens) bool tensor
                    True means valid tactile token, False means padded blank token.
        """
        tactile_keys = list(tactiles.keys())
        batch_size = tactiles[tactile_keys[0]].shape[0]
        device = tactiles[tactile_keys[0]].device
        T = tactiles[tactile_keys[0]].shape[1]
        
        # Assert that all tactile_function_areas have shape (B, N)
        for tactile_key in tactile_keys:
            func_areas = tactile_function_areas[tactile_key]
            assert func_areas.dim() == 2, f"tactile_function_areas[{tactile_key}] must have shape (B, N), got {func_areas.shape}"
        
        # Initialize output tokens/masks:
        # - token values for missing areas are strict zeros (non-learnable blank)
        # - pad mask for missing areas is False so downstream attention ignores them
        output_tokens = torch.zeros(batch_size, T, self.total_num_tactile_tokens, self.token_dim, device=device)
        output_pad_masks = torch.zeros(batch_size, T, self.total_num_tactile_tokens, dtype=torch.bool, device=device)

        # [PACT] Accumulators for the compact spatial layout (tokens_per_area > 1).
        compact_tokens: List[torch.Tensor] = []
        compact_areas: List[torch.Tensor] = []
        compact_contact: List[torch.Tensor] = []
        compact_spatial_valid: List[torch.Tensor] = []

        # Process each tactile_key and fill in the corresponding function areas
        for tactile_key in tactile_keys:

            tactile_data = tactiles[tactile_key]  # (B, T, N, *D)
            func_areas = tactile_function_areas[tactile_key]  # (B, N)
            sensors = tactile_sensors[tactile_key]  # List of length B, each element is a list of strings
            
            B, T_data, N = tactile_data.shape[:3]
            assert T_data == T, f"Time dimension mismatch: {T_data} != {T}"
            
            # Since all samples in batch are from the same domain, use the first sample's sensor.
            sensor_name = sensors[0][0] if isinstance(sensors[0], list) else sensors[0]
            sensor_name = canonicalize_tactile_sensor_name(sensor_name)
            actual_areas = sorted(func_areas[0].tolist())
            actual_tactile_shape = get_tactile_encoder_shape_from_data_shape(
                tactile_data.shape,
                tactile_function_areas=func_areas,
            )

            # Get info for this (tactile_key, sensor_name) pair
            if tactile_key not in self.tactile_key_info:
                raise ValueError(f"No config found for tactile_key: {tactile_key}")
            sensor_info_map = self.tactile_key_info[tactile_key]
            if sensor_name not in sensor_info_map:
                # Fallback: some zarr datasets store the sensor name as an integer (e.g. 0)
                # instead of the proper string. Try to resolve via str() first, then match
                # by actual function_areas + tactile shape from the batch.
                sensor_name_str = canonicalize_tactile_sensor_name(sensor_name)
                if sensor_name_str in sensor_info_map:
                    sensor_name = sensor_name_str
                else:
                    matched_sensors = [
                        s for s, entries in sensor_info_map.items()
                        if any(
                            sorted(entry["function_areas"]) == actual_areas
                            and tuple(entry["tactile_shape"]) == actual_tactile_shape
                            for entry in entries
                        )
                    ]
                    if len(matched_sensors) == 1:
                        logging.warning(
                            f"sensor_name={repr(sensor_name)} not found for tactile_key={tactile_key}. "
                            f"Matched by function_areas {actual_areas} and tactile_shape {actual_tactile_shape} "
                            f"→ using sensor '{matched_sensors[0]}'. "
                            f"Please check the zarr data for this dataset (sensor field may be storing "
                            f"an integer instead of a string)."
                        )
                        sensor_name = matched_sensors[0]
                    else:
                        raise ValueError(
                            f"No config found for tactile_key: {tactile_key}, sensor: {sensor_name}. "
                            f"Available sensors: {list(sensor_info_map.keys())}. "
                            f"Fallback by function_areas {actual_areas} and tactile_shape {actual_tactile_shape} "
                            f"found {len(matched_sensors)} "
                            f"candidates {matched_sensors} (ambiguous). "
                            f"Please fix the zarr sensor name field for this dataset."
                        )
            sensor_entries = sensor_info_map[sensor_name]
            matching_entries = [
                entry for entry in sensor_entries if tuple(entry["tactile_shape"]) == actual_tactile_shape
            ]
            if not matching_entries:
                raise ValueError(
                    f"No config entry found for tactile_key={tactile_key}, sensor={sensor_name}, "
                    f"tactile_shape={actual_tactile_shape}. Available entries: {sensor_entries}"
                )
            exact_area_entries = [
                entry for entry in matching_entries if sorted(entry["function_areas"]) == actual_areas
            ]
            if exact_area_entries:
                matching_entries = exact_area_entries

            tokenizer_keys = {entry["tokenizer_key"] for entry in matching_entries}
            if len(tokenizer_keys) > 1:
                raise ValueError(
                    f"Ambiguous tokenizer resolution for tactile_key={tactile_key}, sensor={sensor_name}, "
                    f"tactile_shape={actual_tactile_shape}, function_areas={actual_areas}. "
                    f"Matching entries: {matching_entries}"
                )
            tokenizer_key = next(iter(tokenizer_keys))
            if tokenizer_key not in self.tokenizers:
                raise ValueError(f"No tokenizer found for key: {tokenizer_key}")

            tokenizer = self.tokenizers[tokenizer_key]
            encoded_tokens = tokenizer(tactile_data)

            if tokenizer.encoder_type == 'image':
                if self.shared_image_chunk_encoder is None:
                    raise ValueError(
                        f"shared_image_chunk_encoder is required for image tactile tokenizer {tokenizer_key}"
                    )
                # encoded_tokens shape: (B, T, N, num_patches+1, embed_dim)
                B_tokens, T_tokens, N_tokens, num_patches_plus_one, embed_dim = encoded_tokens.shape
                tokens_flat = encoded_tokens.reshape(B_tokens * T_tokens * N_tokens, num_patches_plus_one, embed_dim)
                readout = self.shared_image_chunk_encoder(tokens_flat)  # (B*T*N, P, token_dim)
                encoded_tokens = readout.reshape(
                    B_tokens, T_tokens, N_tokens, self.tokens_per_area, self.token_dim
                )
                if self.tokens_per_area == 1:
                    encoded_tokens = encoded_tokens.squeeze(3)  # (B, T, N, token_dim)
            elif self.tokens_per_area > 1:
                # Non-image tokenizers emit a single token per area. Place it in the CLS slot and
                # leave the spatial slots as inactive zeros so the sequence layout stays uniform.
                encoded_tokens = F.pad(
                    encoded_tokens.unsqueeze(3), (0, 0, 0, self.tokens_per_area - 1)
                )  # (B, T, N, P, token_dim)
            
            # Fill encoded tokens into output_tokens based on function_area indices
            # All tokenizers now return (B, T, N, token_dim)
            # Since func_areas are the same across batch, we can use the first sample's func_areas as indices
            # Get fill indices: func_areas[0] is (N,), expand to (T, N) for all time steps
            fill_idx = func_areas[0].unsqueeze(0).expand(T, -1)  # (N,) -> (1, N) -> (T, N)
            
            # Assert that all indices are valid
            assert (fill_idx >= 0).all() and (fill_idx < self.total_num_tactile_tokens).all(), \
                f"Invalid function_area indices in {tactile_key}: {fill_idx}. Must be in [0, {self.total_num_tactile_tokens})"

            if self.tokens_per_area > 1:
                # [PACT] Compact layout: keep the areas that exist instead of scattering into
                # `total_num_tactile_tokens` mostly-empty slots.
                compact_tokens.append(encoded_tokens)  # (B, T, N, P, token_dim)
                compact_areas.append(func_areas)  # (B, N)
                # Only image tokenizers produce real spatial tokens; others fill the CLS slot only.
                has_spatial = tokenizer.encoder_type == 'image'
                compact_spatial_valid.append(
                    torch.full((batch_size, N), has_spatial, dtype=torch.bool, device=device)
                )
                if tactile_contact is not None and tactile_key in tactile_contact:
                    in_contact = tactile_contact[tactile_key].to(device=device, dtype=torch.bool)
                else:
                    in_contact = torch.ones(batch_size, N, dtype=torch.bool, device=device)
                compact_contact.append(in_contact)
                continue

            # Fill tokens for each time step
            for t in range(T):
                # Fill all positions at once using advanced indexing
                # output_tokens[:, t, fill_idx[t], :] shape: (B, N, token_dim)
                # encoded_tokens[:, t, :, :] shape: (B, N, token_dim)
                output_tokens[:, t, fill_idx[t], :] = encoded_tokens[:, t, :, :]
                output_pad_masks[:, t, fill_idx[t]] = True

        if self.tokens_per_area > 1:
            # [PACT] Compact spatial layout. Sequence is [area0 tokens][area1 tokens]... per step,
            # where each area contributes CLS ++ pooled patch grid.
            tokens = torch.cat(compact_tokens, dim=2)  # (B, T, N_all, P, token_dim)
            areas = torch.cat(compact_areas, dim=1)  # (B, N_all)
            spatial_valid = torch.cat(compact_spatial_valid, dim=1)  # (B, N_all)
            in_contact = torch.cat(compact_contact, dim=1)  # (B, N_all)
            bsz, n_steps, n_areas, n_per_area, dim = tokens.shape

            # Area identity uses the true function-area id, so the pretrained embedding table is
            # reused unchanged; the spatial slot gets its own small positional embedding.
            tokens = tokens + self.func_area_idx_embedding(areas)[:, None, :, None, :]
            tokens = tokens + self.spatial_token_embedding.weight.view(1, 1, 1, n_per_area, dim)

            pad_masks = torch.ones(
                bsz, n_steps, n_areas, n_per_area, dtype=torch.bool, device=device
            )
            # The CLS slot always stays: "this pad is present and feels nothing" is informative.
            spatial_keep = spatial_valid
            if self.contact_gating:
                spatial_keep = spatial_keep & in_contact
            pad_masks[..., 1:] = spatial_keep[:, None, :, None]

            tokens = tokens.reshape(bsz, n_steps * n_areas * n_per_area, dim)
            pad_masks = pad_masks.reshape(bsz, n_steps * n_areas * n_per_area)
            tokens = self.unified_proj(tokens)
            tokens = tokens * pad_masks.unsqueeze(-1).to(tokens.dtype)
            return tokens, pad_masks

        # Add function-area index embedding only on valid tactile tokens.
        func_area_emb = self.func_area_idx_embedding.weight.view(1, 1, self.total_num_tactile_tokens, self.token_dim)
        output_tokens = output_tokens + func_area_emb * output_pad_masks.unsqueeze(-1).to(output_tokens.dtype)

        # Reshape: (B, T, total_num_tactile_tokens, token_dim) -> (B, T * total_num_tactile_tokens, token_dim)
        # Arrange as [t=0 tokens], [t=1 tokens], ...
        output_tokens = output_tokens.reshape(batch_size, T * self.total_num_tactile_tokens, self.token_dim)
        output_pad_masks = output_pad_masks.reshape(batch_size, T * self.total_num_tactile_tokens)

        # Apply unified projection
        output_tokens = self.unified_proj(output_tokens)  # (B, T * total_num_tactile_tokens, token_dim)
        output_tokens = output_tokens * output_pad_masks.unsqueeze(-1).to(output_tokens.dtype)

        return output_tokens, output_pad_masks
    
    def save_tokenizers(self, checkpoint_dir):
        """
        Save all tokenizers in self.tokenizers to separate checkpoint files.
        Also saves shared_image_chunk_encoder (which encapsulates both shared_chunk and image_proj) if it exists.
        
        Args:
            checkpoint_dir: Directory path where tokenizer checkpoints will be saved.
                           Each tokenizer will be saved as {tokenizer_key}.safetensors
                           shared_image_chunk_encoder will be saved as shared_image_chunk_encoder.safetensors
        """
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all tokenizers
        for tokenizer_key, tokenizer in self.tokenizers.items():
            tokenizer_path = checkpoint_dir / f"{tokenizer_key}.safetensors"
            safetensors.torch.save_model(tokenizer, tokenizer_path)
        
        # Save shared_image_chunk_encoder if it exists (encapsulates both shared_chunk and image_proj)
        if self.shared_image_chunk_encoder is not None:
            shared_chunk_path = checkpoint_dir / "shared_image_chunk_encoder.safetensors"
            safetensors.torch.save_model(self.shared_image_chunk_encoder, shared_chunk_path)

    ################################################################################
    # History-compatible tokenizer checkpoint fallback for old
    # function-area checkpoint names. New checkpoints are keyed by {sensor}_{type}_{*shape},
    # but old checkpoints may still use {sensor}_{*shape} / {sensor}_{func_area} / {sensor}_{area_group}.
    ################################################################################
    def _tokenizer_checkpoint_matches_current_shapes(self, tokenizer: nn.Module, checkpoint_path: pathlib.Path) -> bool:
        expected_state = tokenizer.state_dict()
        try:
            with safetensors.safe_open(str(checkpoint_path), framework="pt") as handle:
                checkpoint_keys = set(handle.keys())
                if checkpoint_keys != set(expected_state.keys()):
                    return False
                for key, tensor in expected_state.items():
                    if tuple(handle.get_slice(key).get_shape()) != tuple(tensor.shape):
                        return False
        except Exception:
            return False
        return True

    def _try_load_history_tokenizer_checkpoint(
        self,
        checkpoint_dir: pathlib.Path,
        tokenizer_key: str,
        tokenizer: nn.Module,
    ) -> pathlib.Path | None:
        sensor_prefix = f"{canonicalize_tactile_sensor_name(tokenizer.sensor_name)}_"
        candidate_paths = sorted(
            path
            for path in checkpoint_dir.glob(f"{sensor_prefix}*.safetensors")
            if path.name != f"{tokenizer_key}.safetensors"
        )
        for candidate_path in candidate_paths:
            if not self._tokenizer_checkpoint_matches_current_shapes(tokenizer, candidate_path):
                continue
            safetensors.torch.load_model(tokenizer, candidate_path)
            return candidate_path
        return None
    ################################################################################

    def load_tokenizers(self, checkpoint_dir, strict=True):
        """
        Load all tokenizers from separate checkpoint files.
        Also loads shared_image_chunk_encoder (which encapsulates both shared_chunk and image_proj) if needed.
        
        Args:
            checkpoint_dir: Directory path where tokenizer checkpoints are stored.
                           Each tokenizer should be saved as {tokenizer_key}.safetensors
                           shared_image_chunk_encoder should be saved as shared_image_chunk_encoder.safetensors
            strict: If True, raise error if tokenizer key is missing. If False, skip missing tokenizers.
        
        Returns:
            Tuple of (num_loaded, num_missing) where:
            - num_loaded: Number of tokenizers successfully loaded
            - num_missing: Number of tokenizers not found in checkpoint directory
        """
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            if strict:
                raise FileNotFoundError(f"Tokenizer checkpoint directory not found: {checkpoint_dir}")
            return 0, len(self.tokenizers)
        
        loaded_count = 0
        missing_keys = []
        loaded_tokenizer_keys = []
        
        # Load all tokenizers
        for tokenizer_key, tokenizer in self.tokenizers.items():
            tokenizer_path = checkpoint_dir / f"{tokenizer_key}.safetensors"
            if tokenizer_path.exists():
                safetensors.torch.load_model(tokenizer, tokenizer_path)
                loaded_count += 1
                loaded_tokenizer_keys.append(tokenizer_key)
            else:
                legacy_tokenizer_path = self._try_load_history_tokenizer_checkpoint(
                    checkpoint_dir,
                    tokenizer_key,
                    tokenizer,
                )
                if legacy_tokenizer_path is not None:
                    loaded_count += 1
                    loaded_tokenizer_keys.append(tokenizer_key)
                    logging.warning(
                        "Loaded legacy tokenizer checkpoint %s for %s via sensor-prefix + module-shape match",
                        legacy_tokenizer_path.name,
                        tokenizer_key,
                    )
                else:
                    missing_keys.append(tokenizer_key)
                    if strict:
                        raise FileNotFoundError(f"Tokenizer checkpoint not found: {tokenizer_path}")

        # Load shared_image_chunk_encoder if needed.
        # This encapsulates both shared_chunk_encoder and image_proj
        if self.has_image_type:
            shared_chunk_path = checkpoint_dir / "shared_image_chunk_encoder.safetensors"
            if shared_chunk_path.exists():
                if self.shared_image_chunk_encoder is not None:
                    safetensors.torch.load_model(self.shared_image_chunk_encoder, shared_chunk_path)
                    logging.info("Loaded shared_image_chunk_encoder checkpoint (includes shared_chunk and image_proj)")                    
                    # Re-apply frozen_shared_chunk setting after loading weights
                    # This ensures that requires_grad is correctly set even if the checkpoint was saved with different settings
                    if self.tokenizer_config.frozen_shared_chunk:
                        for param in self.shared_image_chunk_encoder.shared_chunk_encoder.parameters():
                            param.requires_grad = False
                        logging.info("Re-applied frozen_shared_chunk=True: shared_chunk_encoder parameters are frozen")
                    elif not self.tokenizer_config.frozen_shared_chunk:
                        for param in self.shared_image_chunk_encoder.shared_chunk_encoder.parameters():
                            param.requires_grad = True
                        logging.info("Re-applied frozen_shared_chunk=False: shared_chunk_encoder parameters are not frozen")
                    else:
                        logging.info("Re-applied frozen_shared_chunk=None: shared_chunk_encoder parameters are not frozen")
                else:
                    logging.warning("shared_image_chunk_encoder checkpoint exists but encoder is None")
            else:
                if strict:
                    raise FileNotFoundError(f"shared_image_chunk_encoder checkpoint not found: {shared_chunk_path}")
                else:
                    logging.warning(f"shared_image_chunk_encoder checkpoint not found: {shared_chunk_path}")
        
        if missing_keys:
            logging.warning(f"Missing tokenizer checkpoints: {missing_keys}")
        
        logging.info(f"Loaded tokenizer checkpoints: {loaded_tokenizer_keys}")
        
        return loaded_count, len(missing_keys)


class FakeFTP1HptTactileEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, tactiles, tactile_function_areas, tactile_sensors):
        tactile_keys = list(tactiles.keys())
        n_token = len(tactile_keys)
        batch_size = tactiles[tactile_keys[0]].shape[0]
        token_dim = 1024   # follow gemma_300m config
        device = tactiles[tactile_keys[0]].device
        dtype = tactiles[tactile_keys[0]].dtype
        tokens = torch.zeros(batch_size, n_token, token_dim, device=device, dtype=dtype)
        pad_masks = torch.ones(batch_size, n_token, dtype=torch.bool, device=device)
        return tokens, pad_masks
    
