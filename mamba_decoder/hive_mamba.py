"""
Hivemind Mamba: A multimodal architecture that processes text and images
with a shared collective state space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
import timm
from mambapy.mamba import Mamba, MambaConfig


class SwiGLU(nn.Module):
    """
    SwiGLU activation function for Feed-Forward Networks.

    Implements the Swish-Gated Linear Unit activation: SwiGLU(x) = Swish(W1(x)) * W2(x)
    where Swish(x) = x * sigmoid(x) = x * silu(x)
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """
        Initialize SwiGLU module.

        Args:
            dim: Input and output feature dimension (positive integer)
            hidden_dim: Hidden layer dimension, typically 4x dim (positive integer)
        """
        super().__init__()
        # First linear transformation for Swish activation
        self.w1: nn.Linear = nn.Linear(dim, hidden_dim, bias=False)
        # Second linear transformation for gating
        self.w2: nn.Linear = nn.Linear(dim, hidden_dim, bias=False)
        # Output projection back to original dimension
        self.w3: nn.Linear = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU activation.

        Args:
            x: Input tensor of shape (..., dim) where ... can be any batch dimensions

        Returns:
            Output tensor of shape (..., dim) - same shape as input
        """
        # Compute Swish(W1(x)) * W2(x), then project to output dimension
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class CrossModalMamba(nn.Module):
    """
    Cross-Modal Mamba: Uses Mamba's selective scan to fuse information
    between text and image modalities.

    This module implements cross-modal attention using Mamba's state-space
    selective scan mechanism, allowing text and image features to influence
    each other through a shared state space.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4
    ) -> None:
        """
        Initialize CrossModalMamba module.

        Args:
            d_model: Model dimension / feature size (positive integer, typically 512)
            d_state: State dimension for Mamba's state-space model (positive integer, default 16)
            d_conv: Convolution dimension (unused in this implementation but kept for compatibility)
        """
        super().__init__()
        self.d_model: int = d_model  # Feature dimension
        self.d_state: int = d_state  # State-space dimension

        # Separate projections for text and image queries/keys/values
        # Each projection outputs 2x d_model to split into query and key
        self.text_to_image_proj: nn.Linear = nn.Linear(
            d_model, d_model * 2
        )
        self.image_to_text_proj: nn.Linear = nn.Linear(
            d_model, d_model * 2
        )

        # Mamba-style selective scan parameters
        # dt (delta time) projections for discretization in state-space model
        self.dt_proj_t2i: nn.Linear = nn.Linear(d_model, d_model)
        self.dt_proj_i2t: nn.Linear = nn.Linear(d_model, d_model)

        # Projections to map features to state dimension
        # B and C are input and output matrices in state-space formulation
        self.B_proj_t2i: nn.Linear = nn.Linear(d_model, d_state)
        self.B_proj_i2t: nn.Linear = nn.Linear(d_model, d_state)
        self.C_proj_t2i: nn.Linear = nn.Linear(d_model, d_state)
        self.C_proj_i2t: nn.Linear = nn.Linear(d_model, d_state)

        # State-space parameters
        # A_log: Log-space parameterization of state transition matrix A
        #        Shape: (d_model, d_state) - different A for each feature dimension
        self.A_log: nn.Parameter = nn.Parameter(
            torch.randn(d_model, d_state)
        )
        # D: Skip connection parameter (diagonal matrix)
        #    Shape: (d_model,) - one value per feature dimension
        self.D: nn.Parameter = nn.Parameter(torch.randn(d_model))

        # Output projection: combines original features with cross-modal features
        self.out_proj: nn.Linear = nn.Linear(d_model * 2, d_model)
        # Layer normalization for stable training
        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified selective scan for cross-modal fusion.

        Implements a state-space model scan: h[t+1] = A_discrete * h[t] + B_discrete * x[t]
        where A_discrete and B_discrete are discretized versions of continuous state-space matrices.

        Args:
            x: Input sequence tensor of shape (B, L, D)
               B = batch size, L = sequence length, D = feature dimension
            dt: Time step tensor of shape (B, L, D)
                Controls the discretization step size for each position
            A: State transition matrix of shape (D, d_state)
               Different transition matrix for each feature dimension
            B: Input matrix of shape (B, L, d_state)
               Maps input to state space, varies per sequence position
            C: Output matrix of shape (B, L, d_state)
               Maps state to output, varies per sequence position

        Returns:
            Output tensor of shape (B, L, D) - processed sequence
        """
        B_batch, L, D = x.shape

        # Discretization: Convert continuous state-space to discrete-time
        # Apply softplus to ensure dt > 0 (positive time steps)
        dt = F.softplus(dt)  # (B, L, D) - positive time steps

        # Discretize state transition matrix A
        # A_discrete = exp(dt * A) for each position and feature dimension
        # dt.unsqueeze(-1): (B, L, D, 1)
        # A.unsqueeze(0).unsqueeze(0): (1, 1, D, d_state)
        # Result: (B, L, D, d_state) - discretized A for each position
        dA = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, D, d_state)

        # Discretize input matrix B
        # B_discrete = dt * B for each position
        # dt_proj: (B, L, D, 1)
        # B.unsqueeze(2): (B, L, 1, d_state)
        # Result: (B, L, D, d_state) - discretized B for each position
        dt_proj = dt.unsqueeze(-1)  # (B, L, D, 1)
        dB = dt_proj * B.unsqueeze(
            2
        )  # (B, L, D, d_state) = (B, L, D, 1) * (B, L, 1, d_state)

        # State-space scan: Sequential processing of the sequence
        # Initialize hidden state: (B, D, d_state)
        # Each feature dimension has its own state vector of size d_state
        h: torch.Tensor = torch.zeros(
            B_batch, D, self.d_state, device=x.device, dtype=x.dtype
        )
        outputs: List[torch.Tensor] = []

        # Process each position in the sequence sequentially
        for i in range(L):
            # Update hidden state: h[t+1] = A_discrete * h[t] + B_discrete * x[t]
            # dA[:, i]: (B, D, d_state) - discretized A at position i
            # h: (B, D, d_state) - current hidden state
            # dB[:, i]: (B, D, d_state) - discretized B at position i
            # x[:, i].unsqueeze(-1): (B, D, 1) - input at position i
            h = dA[:, i] * h + dB[:, i] * x[:, i].unsqueeze(-1)
            # h: (B, D, d_state) - updated hidden state

            # Compute output: y[t] = C * h[t]
            # C[:, i].unsqueeze(-1): (B, d_state, 1) - output matrix at position i
            # h @ C[:, i].unsqueeze(-1): (B, D, 1) - matrix multiplication
            # squeeze(-1): (B, D) - output at position i
            y: torch.Tensor = (h @ C[:, i].unsqueeze(-1)).squeeze(
                -1
            )  # (B, D)
            outputs.append(y)

        # Stack all outputs along sequence dimension
        # Result: (B, L, D) - full output sequence
        return torch.stack(outputs, dim=1)  # (B, L, D)

    def forward(
        self, text_feats: torch.Tensor, image_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal Mamba fusion.

        Args:
            text_feats: Text features tensor of shape (B, L_t, D)
                       B = batch size, L_t = text sequence length, D = feature dimension
            image_feats: Image features tensor of shape (B, L_i, D)
                        B = batch size, L_i = image sequence length (patches), D = feature dimension

        Returns:
            Tuple of (text_output, image_output):
            - text_output: Enhanced text features of shape (B, L_t, D)
            - image_output: Enhanced image features of shape (B, L_i, D)
        """
        # Text attending to Image: Project text features to query/key space
        # Output is 2x d_model, split into query and key
        t2i: torch.Tensor = self.text_to_image_proj(
            text_feats
        )  # (B, L_t, 2*D)
        t2i_q: torch.Tensor  # (B, L_t, D) - text queries
        t2i_k: torch.Tensor  # (B, L_t, D) - text keys
        t2i_q, t2i_k = t2i.chunk(2, dim=-1)

        # Image attending to Text: Project image features to query/key space
        i2t: torch.Tensor = self.image_to_text_proj(
            image_feats
        )  # (B, L_i, 2*D)
        i2t_q: torch.Tensor  # (B, L_i, D) - image queries
        i2t_k: torch.Tensor  # (B, L_i, D) - image keys
        i2t_q, i2t_k = i2t.chunk(2, dim=-1)

        # Cross-modal selective parameters: Compute time steps for discretization
        # These control how much information flows from one modality to another
        dt_t2i: torch.Tensor = self.dt_proj_t2i(t2i_q)  # (B, L_t, D)
        dt_i2t: torch.Tensor = self.dt_proj_i2t(i2t_q)  # (B, L_i, D)

        # State transition matrix: Convert from log-space to actual values
        # Negative ensures stability (A is negative definite)
        A: torch.Tensor = -torch.exp(self.A_log)  # (D, d_state)

        # Compute cross-attention style B and C matrices
        # For text-to-image: aggregate image keys to match text sequence length
        # Use cross-attention to project image features to text positions
        # Attention weights: how much each image patch contributes to each text position
        attn_t2i: torch.Tensor = torch.softmax(
            t2i_q @ i2t_k.transpose(-2, -1) / (self.d_model**0.5),
            dim=-1,
        )  # (B, L_t, L_i) - attention weights from text to image
        # Weighted sum of image keys: aggregate image information for each text position
        B_t2i_feat: torch.Tensor = (
            attn_t2i @ i2t_k
        )  # (B, L_t, D) - aggregated image keys for each text position
        # Project to state dimension for state-space model
        B_t2i: torch.Tensor = self.B_proj_t2i(
            B_t2i_feat
        )  # (B, L_t, d_state)
        C_t2i: torch.Tensor = self.C_proj_t2i(
            B_t2i_feat
        )  # (B, L_t, d_state)

        # For image-to-text: aggregate text keys to match image sequence length
        # Attention weights: how much each text token contributes to each image patch
        attn_i2t: torch.Tensor = torch.softmax(
            i2t_q @ t2i_k.transpose(-2, -1) / (self.d_model**0.5),
            dim=-1,
        )  # (B, L_i, L_t) - attention weights from image to text
        # Weighted sum of text keys: aggregate text information for each image position
        B_i2t_feat: torch.Tensor = (
            attn_i2t @ t2i_k
        )  # (B, L_i, D) - aggregated text keys for each image position
        # Project to state dimension for state-space model
        B_i2t: torch.Tensor = self.B_proj_i2t(
            B_i2t_feat
        )  # (B, L_i, d_state)
        C_i2t: torch.Tensor = self.C_proj_i2t(
            B_i2t_feat
        )  # (B, L_i, d_state)

        # Selective scan for cross-modal fusion
        # Process text features with information from images
        text_from_image: torch.Tensor = self.selective_scan(
            t2i_q, dt_t2i, A, B_t2i, C_t2i
        )  # (B, L_t, D)
        # Process image features with information from text
        image_from_text: torch.Tensor = self.selective_scan(
            i2t_q, dt_i2t, A, B_i2t, C_i2t
        )  # (B, L_i, D)

        # Combine original features with cross-modal enhancements
        # Concatenate along feature dimension, then project back to d_model
        text_output: torch.Tensor = self.out_proj(
            torch.cat([text_feats, text_from_image], dim=-1)
        )  # (B, L_t, D)
        image_output: torch.Tensor = self.out_proj(
            torch.cat([image_feats, image_from_text], dim=-1)
        )  # (B, L_i, D)

        # Apply layer normalization for stable training
        return self.norm(text_output), self.norm(image_output)


class HivemindMambaLayer(nn.Module):
    """
    A single Hivemind Mamba layer that processes both modalities
    with a shared collective state space.

    This layer consists of:
    1. Independent Mamba processing for each modality
    2. Cross-modal fusion using CrossModalMamba
    3. Feed-forward network (SwiGLU) for final feature refinement
    """

    def __init__(self, config: MambaConfig) -> None:
        """
        Initialize HivemindMambaLayer.

        Args:
            config: MambaConfig object containing model hyperparameters
                   Must have attributes: d_model, d_state, d_conv, expand_factor
        """
        super().__init__()
        self.d_model: int = config.d_model  # Feature dimension

        # Independent Mamba processing for each modality
        # Each modality is processed separately before cross-modal fusion
        self.text_mamba: Mamba = Mamba(
            MambaConfig(
                d_model=config.d_model,
                n_layers=1,  # Single layer per modality
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand_factor=config.expand_factor,
            )
        )

        self.image_mamba: Mamba = Mamba(
            MambaConfig(
                d_model=config.d_model,
                n_layers=1,  # Single layer per modality
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand_factor=config.expand_factor,
            )
        )

        # Shared hivemind state fusion: Cross-modal attention via Mamba
        self.cross_modal_mamba: CrossModalMamba = CrossModalMamba(
            d_model=config.d_model, d_state=config.d_state
        )

        # SwiGLU FFN for final modality mixing
        # Hidden dimension is 4x the model dimension (standard practice)
        self.ffn: SwiGLU = SwiGLU(config.d_model, config.d_model * 4)
        # Layer normalization for text and image features separately
        self.norm1: nn.LayerNorm = nn.LayerNorm(config.d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(config.d_model)

    def forward(
        self, text_x: torch.Tensor, image_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through HivemindMambaLayer.

        Args:
            text_x: Text features tensor of shape (B, L_t, D)
                   B = batch size, L_t = text sequence length, D = feature dimension
            image_x: Image features tensor of shape (B, L_i, D)
                    B = batch size, L_i = image sequence length, D = feature dimension

        Returns:
            Tuple of (text_out, image_out):
            - text_out: Processed text features of shape (B, L_t, D)
            - image_out: Processed image features of shape (B, L_i, D)
        """
        # Independent Mamba processing: Each modality processes its own features
        # This allows each modality to develop its own internal representations
        text_mamba_out: torch.Tensor = self.text_mamba(
            text_x
        )  # (B, L_t, D)
        image_mamba_out: torch.Tensor = self.image_mamba(
            image_x
        )  # (B, L_i, D)

        # Cross-modal fusion via Mamba: Exchange information between modalities
        # Text features are enhanced with image information and vice versa
        text_cross: torch.Tensor  # (B, L_t, D)
        image_cross: torch.Tensor  # (B, L_i, D)
        text_cross, image_cross = self.cross_modal_mamba(
            text_mamba_out, image_mamba_out
        )

        # Residual connection: Add original features to cross-modal enhancements
        # This preserves original information while adding cross-modal context
        text_fused: torch.Tensor = text_x + text_cross  # (B, L_t, D)
        image_fused: torch.Tensor = (
            image_x + image_cross
        )  # (B, L_i, D)

        # SwiGLU FFN with residual: Final feature refinement
        # Normalize -> FFN -> Add residual for stable training
        text_out: torch.Tensor = text_fused + self.ffn(
            self.norm1(text_fused)
        )  # (B, L_t, D)
        image_out: torch.Tensor = image_fused + self.ffn(
            self.norm2(image_fused)
        )  # (B, L_i, D)

        return text_out, image_out


class HivemindMamba(nn.Module):
    """
    Complete Hivemind Mamba model for multimodal processing.

    This model processes both text and image inputs using a shared state-space
    architecture, enabling cross-modal understanding and generation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        image_encoder: str = "vit_base_patch16_224",
        num_classes: int = 1000,
        max_seq_len: int = 512,
    ) -> None:
        """
        Initialize HivemindMamba model.

        Args:
            vocab_size: Vocabulary size for text tokenizer (positive integer)
            d_model: Model dimension / feature size (positive integer, default 512)
            n_layers: Number of HivemindMambaLayer blocks (positive integer, default 6)
            d_state: State dimension for Mamba's state-space model (positive integer, default 16)
            d_conv: Convolution dimension for Mamba (positive integer, default 4)
            expand_factor: Expansion factor for Mamba's feed-forward (positive integer, default 2)
            image_encoder: TIMM model name for image encoding (string, default "vit_base_patch16_224")
            num_classes: Number of classes for classification tasks (positive integer, default 1000)
            max_seq_len: Maximum sequence length for text (positive integer, default 512)
        """
        super().__init__()

        self.d_model: int = d_model  # Feature dimension
        self.vocab_size: int = vocab_size  # Vocabulary size
        self.max_seq_len: int = max_seq_len  # Maximum sequence length

        # Text embedding: Maps token indices to dense vectors
        # Input: token indices (B, L_t), Output: embeddings (B, L_t, d_model)
        self.text_embedding: nn.Embedding = nn.Embedding(
            vocab_size, d_model
        )
        # Positional encoding: Learnable position embeddings
        # Shape: (1, max_seq_len, d_model) - broadcastable to batch
        self.text_pos_encoding: nn.Parameter = nn.Parameter(
            torch.randn(1, max_seq_len, d_model)
        )

        # Image encoder using TIMM: Vision Transformer or CNN backbone
        # Removes classification head and global pooling to get patch features
        self.image_encoder: nn.Module = timm.create_model(
            image_encoder,
            pretrained=True,  # Use pretrained weights
            num_classes=0,  # Remove classification head
            global_pool="",  # Remove global pooling to keep spatial structure
        )

        # Get the feature dimension from TIMM model dynamically
        # This allows using different image encoders without hardcoding dimensions
        with torch.no_grad():
            dummy_img: torch.Tensor = torch.randn(
                1, 3, 224, 224
            )  # (B, C, H, W)
            img_features: torch.Tensor = self.image_encoder(dummy_img)
            # Extract feature dimension from last axis
            img_feature_dim: int = int(img_features.shape[-1])

        # Project image features to d_model: Align image and text feature spaces
        # Input: (B, N_patches, img_feature_dim), Output: (B, N_patches, d_model)
        self.image_proj: nn.Linear = nn.Linear(
            img_feature_dim, d_model
        )

        # Hivemind Mamba layers configuration
        # Each layer uses single Mamba block per modality (n_layers=1)
        # The stacking happens via n_layers parameter in HivemindMamba
        self.config: MambaConfig = MambaConfig(
            d_model=d_model,
            n_layers=1,  # Single Mamba block per layer
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
        )

        # Stack of HivemindMambaLayer blocks
        # Each layer processes both modalities with cross-modal fusion
        self.layers: nn.ModuleList = nn.ModuleList(
            [HivemindMambaLayer(self.config) for _ in range(n_layers)]
        )

        # Output heads: Layer normalization and projection layers
        # Normalize features before final projection
        self.text_norm: nn.LayerNorm = nn.LayerNorm(d_model)
        self.image_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        # Task-specific output heads
        # Text head: Predicts next token logits (B, L_t, vocab_size)
        self.text_head: nn.Linear = nn.Linear(d_model, vocab_size)
        # Image head: Classifies image features (B, num_classes)
        self.image_head: nn.Linear = nn.Linear(d_model, num_classes)
        # Joint head: Combines text and image for cross-modal tasks (B, num_classes)
        self.joint_head: nn.Linear = nn.Linear(
            d_model * 2, num_classes
        )

    def forward(
        self,
        text_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through HivemindMamba model.

        Args:
            text_input: Optional text token indices tensor of shape (B, L_t)
                       B = batch size, L_t = text sequence length
                       Each element is a token index in range [0, vocab_size)
            image_input: Optional image tensor of shape (B, 3, H, W)
                        B = batch size, 3 = RGB channels, H = height, W = width
                        Typically H=W=224 for standard image encoders
            return_features: If True, return intermediate features instead of logits (bool)

        Returns:
            If return_features=True:
                Dictionary with keys:
                - "text_features": (B, L_t, d_model) - text feature embeddings
                - "image_features": (B, N_patches, d_model) - image feature embeddings

            If return_features=False:
                Dictionary with keys (depending on inputs):
                - "text_logits": (B, L_t, vocab_size) - if text_input provided
                - "image_logits": (B, num_classes) - if image_input provided
                - "joint_logits": (B, num_classes) - if both inputs provided
        """
        outputs: Dict[str, torch.Tensor] = {}

        # Process text: Convert token indices to embeddings
        if text_input is not None:
            B: int
            L_t: int
            B, L_t = text_input.shape
            # Embed tokens: (B, L_t) -> (B, L_t, d_model)
            text_x: torch.Tensor = self.text_embedding(text_input)
            # Add positional encoding: (B, L_t, d_model)
            text_x = text_x + self.text_pos_encoding[:, :L_t, :]
        else:
            text_x: Optional[torch.Tensor] = None

        # Process images: Extract features from image encoder
        if image_input is not None:
            # Encode images: (B, 3, H, W) -> (B, N_patches, D_img)
            # N_patches depends on image encoder (e.g., 196 for ViT-Base with 224x224)
            img_features: torch.Tensor = self.image_encoder(
                image_input
            )
            # Project to model dimension: (B, N_patches, D_img) -> (B, N_patches, d_model)
            image_x: torch.Tensor = self.image_proj(img_features)
        else:
            image_x: Optional[torch.Tensor] = None

        # Handle cases where only one modality is provided
        # Create dummy features for missing modality to maintain batch consistency
        if text_x is None and image_x is not None:
            # Image-only processing: Create dummy text features
            B, L_i, D = image_x.shape
            text_x = torch.zeros(
                B, 1, D, device=image_x.device, dtype=image_x.dtype
            )
        elif image_x is None and text_x is not None:
            # Text-only processing: Create dummy image features
            B, L_t, D = text_x.shape
            image_x = torch.zeros(
                B, 1, D, device=text_x.device, dtype=text_x.dtype
            )

        # Pass through Hivemind layers: Sequential processing with cross-modal fusion
        # Each layer processes both modalities and exchanges information
        for layer in self.layers:
            text_x, image_x = layer(text_x, image_x)

        # Normalize: Apply layer normalization before final projections
        text_x = self.text_norm(text_x)  # (B, L_t, d_model)
        image_x = self.image_norm(image_x)  # (B, N_patches, d_model)

        # Return intermediate features if requested (useful for feature extraction)
        if return_features:
            return {
                "text_features": text_x,  # (B, L_t, d_model)
                "image_features": image_x,  # (B, N_patches, d_model)
            }

        # Generate task-specific outputs
        if text_input is not None:
            # Text logits: Predict next token probabilities
            # (B, L_t, d_model) -> (B, L_t, vocab_size)
            outputs["text_logits"] = self.text_head(text_x)

        if image_input is not None:
            # Image logits: Classify image
            # Global average pooling: (B, N_patches, d_model) -> (B, d_model)
            image_pooled: torch.Tensor = image_x.mean(dim=1)
            # Project to classes: (B, d_model) -> (B, num_classes)
            outputs["image_logits"] = self.image_head(image_pooled)

        # Joint prediction: Cross-modal classification task
        # Combines text and image features for tasks like image captioning
        if text_input is not None and image_input is not None:
            # Pool text features: (B, L_t, d_model) -> (B, d_model)
            text_pooled: torch.Tensor = text_x.mean(dim=1)
            # Pool image features: (B, N_patches, d_model) -> (B, d_model)
            image_pooled: torch.Tensor = image_x.mean(dim=1)
            # Concatenate: (B, d_model) + (B, d_model) -> (B, 2*d_model)
            joint_features: torch.Tensor = torch.cat(
                [text_pooled, image_pooled], dim=-1
            )
            # Project to classes: (B, 2*d_model) -> (B, num_classes)
            outputs["joint_logits"] = self.joint_head(joint_features)

        return outputs

    def generate(
        self,
        text_input: Optional[torch.Tensor] = None,
        image_input: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text tokens autoregressively using the model.

        This method generates text tokens one at a time, conditioning on:
        - Previous tokens (autoregressive)
        - Optional image input (multimodal)

        Supports multiple sampling strategies: temperature sampling, top-k,
        top-p (nucleus), and greedy decoding.

        Args:
            text_input: Optional initial token indices tensor of shape (B, L_t)
                       B = batch size, L_t = initial sequence length
                       If None, generation starts from pad_token_id
            image_input: Optional image tensor of shape (B, 3, H, W)
                        B = batch size, 3 = RGB channels, H = height, W = width
                        Used for multimodal generation (e.g., image captioning)
            max_length: Maximum total length of generated sequence (positive integer)
                       Includes initial tokens if text_input is provided
            temperature: Sampling temperature (positive float)
                        Higher values (e.g., 1.5) = more random/diverse
                        Lower values (e.g., 0.5) = more deterministic/focused
                        Only used when do_sample=True
            top_k: Optional top-k sampling parameter (positive integer or None)
                  If set, only sample from top k most likely tokens
                  Filters out low-probability tokens for more focused generation
            top_p: Optional nucleus sampling parameter (float in [0, 1] or None)
                  If set, use nucleus sampling (top-p)
                  Samples from tokens whose cumulative probability <= top_p
                  More dynamic than top-k as it adapts to distribution shape
            do_sample: Whether to use sampling (True) or greedy decoding (False)
                      If False, always picks most likely token (temperature/top_k/top_p ignored)
            pad_token_id: Token ID to use for padding/initialization (integer)
                         Used when text_input is None
            eos_token_id: Optional end-of-sequence token ID (integer or None)
                         Generation stops when this token is generated
                         If None, generation continues until max_length

        Returns:
            Generated token indices tensor of shape (B, generated_length)
            B = batch size, generated_length <= max_length
            Includes initial tokens if text_input was provided
        """
        # Set model to evaluation mode (disables dropout, batch norm updates)
        self.eval()
        # Get device from model parameters (ensures tensors are on same device)
        device: torch.device = next(self.parameters()).device

        # Initialize generation: Set up starting sequence
        if text_input is not None:
            # Start from provided text input
            generated: torch.Tensor = text_input.clone()  # (B, L_t)
            B: int = text_input.shape[0]
        else:
            # Start with pad token if no initial input
            # Determine batch size from image input or use 1
            B = image_input.shape[0] if image_input is not None else 1
            # Create single-token sequence with pad token
            generated = torch.full(
                (B, 1), pad_token_id, dtype=torch.long, device=device
            )

        # Process image once if provided (more efficient than processing each step)
        # Image features remain constant throughout generation
        image_x: Optional[torch.Tensor] = None
        if image_input is not None:
            with torch.no_grad():  # No gradients needed during generation
                # Encode image: (B, 3, H, W) -> (B, N_patches, D_img)
                img_features: torch.Tensor = self.image_encoder(
                    image_input
                )
                # Project to model dimension: (B, N_patches, D_img) -> (B, N_patches, d_model)
                image_x = self.image_proj(img_features)

        # Generate tokens autoregressively: One token at a time
        # Continue until max_length or EOS token
        for _ in range(max_length - generated.shape[1]):
            # Get logits for next token: Forward pass through model
            with torch.no_grad():  # No gradients needed during generation
                # Process current sequence
                current_text: torch.Tensor = (
                    generated  # (B, current_length)
                )
                B, L_t = current_text.shape

                # Embed text: Convert token indices to dense vectors
                # (B, L_t) -> (B, L_t, d_model)
                text_x: torch.Tensor = self.text_embedding(
                    current_text
                )
                # Add positional encoding: (B, L_t, d_model)
                text_x = text_x + self.text_pos_encoding[:, :L_t, :]

                # Handle image: Use provided image features or create dummy
                if image_x is None:
                    # Create dummy image features if no image
                    # Single dummy patch to maintain batch consistency
                    image_x_dummy: torch.Tensor = torch.zeros(
                        B,
                        1,
                        self.d_model,
                        device=device,
                        dtype=text_x.dtype,
                    )
                else:
                    image_x_dummy: torch.Tensor = (
                        image_x  # (B, N_patches, d_model)
                    )

                # Pass through layers: Process with cross-modal fusion
                for layer in self.layers:
                    text_x, image_x_dummy = layer(
                        text_x, image_x_dummy
                    )

                # Normalize: Apply layer normalization
                text_x = self.text_norm(text_x)  # (B, L_t, d_model)

                # Get logits for last position only: Only need prediction for next token
                # Extract last hidden state: (B, L_t, d_model) -> (B, d_model)
                last_hidden: torch.Tensor = text_x[:, -1, :]
                # Project to vocabulary: (B, d_model) -> (B, vocab_size)
                logits: torch.Tensor = self.text_head(last_hidden)

            # Sample next token: Choose token based on logits
            if do_sample:
                # Sampling mode: Use probability distribution

                # Apply temperature: Scale logits to control randomness
                # Higher temperature = flatter distribution = more random
                # Lower temperature = sharper distribution = more focused
                logits = logits / temperature

                # Apply top-k filtering: Keep only top k most likely tokens
                if top_k is not None and top_k > 0:
                    # Ensure top_k doesn't exceed vocabulary size
                    top_k = min(top_k, logits.size(-1))
                    # Get threshold value (k-th largest logit)
                    # torch.topk returns (values, indices), we need values
                    kth_value: torch.Tensor = torch.topk(
                        logits, top_k
                    )[0][..., -1, None]
                    # Create mask for tokens below threshold
                    indices_to_remove: torch.Tensor = (
                        logits < kth_value
                    )
                    # Set low-probability tokens to -inf (will have 0 probability after softmax)
                    logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering: Dynamic token set based on cumulative probability
                if top_p is not None and top_p < 1.0:
                    # Sort logits in descending order
                    sorted_logits: torch.Tensor  # (B, vocab_size)
                    sorted_indices: (
                        torch.Tensor
                    )  # (B, vocab_size) - original indices
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True
                    )
                    # Compute cumulative probabilities
                    # Softmax: (B, vocab_size) -> probabilities
                    # Cumsum: cumulative sum along vocabulary dimension
                    cumulative_probs: torch.Tensor = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )  # (B, vocab_size)

                    # Remove tokens with cumulative probability above threshold
                    # Find first position where cumulative prob > top_p
                    sorted_indices_to_remove: torch.Tensor = (
                        cumulative_probs > top_p
                    )  # (B, vocab_size) - boolean mask
                    # Shift mask right to keep the first token above threshold
                    # This ensures we always keep at least one token
                    sorted_indices_to_remove[..., 1:] = (
                        sorted_indices_to_remove[..., :-1].clone()
                    )
                    sorted_indices_to_remove[..., 0] = 0

                    # Create mask in original (unsorted) order
                    # Scatter sorted mask back to original positions
                    indices_to_remove = (
                        sorted_indices_to_remove.scatter(
                            1,
                            sorted_indices,
                            sorted_indices_to_remove,
                        )
                    )
                    # Set filtered tokens to -inf
                    logits[indices_to_remove] = float("-inf")

                # Sample from distribution: Multinomial sampling
                # Convert logits to probabilities
                probs: torch.Tensor = F.softmax(
                    logits, dim=-1
                )  # (B, vocab_size)
                # Sample one token per batch element
                next_token: torch.Tensor = torch.multinomial(
                    probs, num_samples=1
                )  # (B, 1)
            else:
                # Greedy decoding: Always pick most likely token
                # Argmax: (B, vocab_size) -> (B, 1)
                next_token = torch.argmax(
                    logits, dim=-1, keepdim=True
                )

            # Append to generated sequence: Add new token to sequence
            # Concatenate along sequence dimension
            generated = torch.cat(
                [generated, next_token], dim=1
            )  # (B, current_length + 1)

            # Check for EOS token: Early stopping if end token generated
            if eos_token_id is not None:
                # Check if all batch elements generated EOS token
                if torch.all(next_token == eos_token_id):
                    break  # Stop generation

            # Check if we've reached max sequence length: Hard limit
            if generated.shape[1] >= self.max_seq_len:
                break  # Stop generation

        return generated  # (B, generated_length)


# # Example usage
# if __name__ == "__main__":
#     # Initialize model
#     model = HivemindMamba(
#         vocab_size=32000, d_model=512, n_layers=6, num_classes=1000
#     )

#     # Example inputs
#     text = torch.randint(0, 32000, (2, 128))  # (batch, seq_len)
#     images = torch.randn(
#         2, 3, 224, 224
#     )  # (batch, channels, height, width)

#     # # Forward pass - multimodal
#     # outputs = model(text_input=text, image_input=images)
#     # print("Text logits shape:", outputs["text_logits"].shape)
#     # print("Image logits shape:", outputs["image_logits"].shape)
#     # print("Joint logits shape:", outputs["joint_logits"].shape)

#     # # # Forward pass - text only
#     # text_only = model(text_input=text)
#     # print("\nText-only logits shape:", text_only["text_logits"].shape)

#     # # Forward pass - image only
#     # image_only = model(image_input=images)
#     # print(
#     #     "Image-only logits shape:", image_only["image_logits"].shape
#     # )

#     # # Get features
#     # features = model(
#     #     text_input=text, image_input=images, return_features=True
#     # )
#     # print("\nText features shape:", features["text_features"].shape)
#     # print("Image features shape:", features["image_features"].shape)

#     # Test multimodal generation
#     print("\n" + "=" * 60)
#     print("Testing Multimodal Generation")
#     print("=" * 60)

#     # Create random initial text tokens (prompt)
#     initial_text = torch.randint(
#         0, 32000, (2, 10)
#     )  # (batch, initial_seq_len)
#     test_images = torch.randn(
#         2, 3, 224, 224
#     )  # (batch, channels, height, width)

#     print(f"Initial text shape: {initial_text.shape}")
#     print(f"Image input shape: {test_images.shape}")
#     print(
#         f"\nInitial prompt tokens (batch 0): {initial_text[0].tolist()}"
#     )
#     print(
#         f"Initial prompt tokens (batch 1): {initial_text[1].tolist()}"
#     )

#     # Helper function to display generated tokens
#     def display_generation(initial_tokens, generated_tokens, title):
#         """Display the generation results."""
#         print(f"\n{title}")
#         print("-" * 60)
#         for batch_idx in range(generated_tokens.shape[0]):
#             initial_len = (
#                 initial_tokens.shape[1]
#                 if initial_tokens is not None
#                 else 0
#             )
#             new_tokens = generated_tokens[
#                 batch_idx, initial_len:
#             ].tolist()
#             full_sequence = generated_tokens[batch_idx].tolist()

#             print(f"\nBatch {batch_idx}:")
#             if initial_tokens is not None:
#                 print(
#                     f"  Initial prompt: {initial_tokens[batch_idx].tolist()}"
#                 )
#             print(f"  New tokens: {new_tokens}")
#             print(
#                 f"  Full sequence ({len(full_sequence)} tokens): {full_sequence}"
#             )
#             print(f"  Sequence length: {len(full_sequence)}")

#     # Test generation with different sampling strategies
#     print("\n" + "=" * 60)
#     print("--- Testing with temperature sampling ---")
#     generated_temp = model.generate(
#         text_input=initial_text,
#         image_input=test_images,
#         max_length=30,
#         temperature=0.8,
#         do_sample=True,
#     )
#     display_generation(
#         initial_text, generated_temp, "Temperature Sampling Results"
#     )

#     print("\n" + "=" * 60)
#     print("--- Testing with top-k sampling ---")
#     generated_topk = model.generate(
#         text_input=initial_text,
#         image_input=test_images,
#         max_length=30,
#         temperature=1.0,
#         top_k=50,
#         do_sample=True,
#     )
#     display_generation(
#         initial_text, generated_topk, "Top-K Sampling Results"
#     )

#     print("\n" + "=" * 60)
#     print("--- Testing with top-p (nucleus) sampling ---")
#     generated_topp = model.generate(
#         text_input=initial_text,
#         image_input=test_images,
#         max_length=30,
#         temperature=0.9,
#         top_p=0.9,
#         do_sample=True,
#     )
#     display_generation(
#         initial_text,
#         generated_topp,
#         "Top-P (Nucleus) Sampling Results",
#     )

#     print("\n" + "=" * 60)
#     print("--- Testing with greedy decoding ---")
#     generated_greedy = model.generate(
#         text_input=initial_text,
#         image_input=test_images,
#         max_length=30,
#         do_sample=False,
#     )
#     display_generation(
#         initial_text, generated_greedy, "Greedy Decoding Results"
#     )

#     print("\n" + "=" * 60)
#     print(
#         "--- Testing image-only generation (starts from pad token) ---"
#     )
#     generated_img_only = model.generate(
#         image_input=test_images,
#         max_length=25,
#         temperature=0.7,
#         pad_token_id=0,
#         do_sample=True,
#     )
#     print("\nImage-only generation (no initial prompt):")
#     print("-" * 60)
#     for batch_idx in range(generated_img_only.shape[0]):
#         tokens = generated_img_only[batch_idx].tolist()
#         print(f"\nBatch {batch_idx}:")
#         print(f"  Generated tokens: {tokens}")
#         print(f"  Sequence length: {len(tokens)}")

#     print("\n" + "=" * 60)
#     print("--- Testing text-only generation ---")
#     generated_text_only = model.generate(
#         text_input=initial_text,
#         max_length=30,
#         temperature=0.8,
#         do_sample=True,
#     )
#     display_generation(
#         initial_text,
#         generated_text_only,
#         "Text-Only Generation Results",
#     )

#     print("\n" + "=" * 60)
#     print("Multimodal Generation Tests Completed Successfully!")
#     print("=" * 60)
