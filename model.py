import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Shared Modules ---
class LinearEmbed(nn.Module):
    """(Internal Module) Linear Projection Module: Projects the high-dimensional input spectrum."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


# =================================================================
# [!!!] Champion Model (SpectralMLP) and its Block
# =================================================================

class MLPBlock(nn.Module):
    """
    (Internal Module) Standard Residual Fully Connected Block (MLP Block).
    Applies LayerNorm -> MLP -> GELU -> Residual Skip Connection.
    """

    def __init__(self, embed_dim, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.drop_path = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Input x shape is expected to be (Batch, embed_dim)
        residual = x
        x_norm = self.norm(x)
        mlp_out = self.mlp(x_norm)
        # Apply residual connection
        x = residual + self.drop_path(mlp_out)
        return x


class SpectralMLP(nn.Module):
    """
    (*** Champion Model ***)
    Standard Multilayer Perceptron (MLP) model with residual skip connections
    (The Champion model proposed in this study).
    """

    def __init__(self, input_dim, output_dim=3, embed_dim=128, depth=4, dropout=0.0, **kwargs):
        # **kwargs absorbs irrelevant parameters (e.g., num_heads) for flexible use in ablation studies.
        super().__init__()
        print(f"Initializing [CHAMPION] SpectralMLP: embed={embed_dim}, depth={depth}, dropout={dropout}")

        # 1. Linear Embedding Layer
        self.embed = LinearEmbed(input_dim, embed_dim)

        # 2. Stacked MLP Blocks
        self.blocks = nn.ModuleList([
            MLPBlock(embed_dim, mlp_ratio=2.0, dropout=dropout)
            for _ in range(depth)
        ])

        # 3. Prediction Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, input_dim)
        x = self.embed(x)
        # x shape: (Batch, embed_dim)
        for blk in self.blocks:
            x = blk(x)
        # Final shape is (Batch, embed_dim) before head
        return self.head(x)


# =================================================================
# [!!!] Ablation Model (SpectralTransformerMLP)
# =================================================================

class TransformerBlock(nn.Module):
    """
    (Internal Module) Transformer-like sequence block with optional Dropout.
    """

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.drop_path_attn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop_path_mlp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Note: Input x is expected to be (Batch, SeqLen, embed_dim)
        residual = x
        x_norm = self.norm1(x)

        # Self-Attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.drop_path_attn(attn_out)

        # MLP/FFN
        residual = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + self.drop_path_mlp(mlp_out)
        return x


class SpectralTransformerMLP(nn.Module):
    """
    (*** Ablation Model ***)
    Model using Transformer blocks designed for sequence processing,
    specifically configured for single-point attention (SeqLen=1).
    """

    def __init__(self, input_dim, output_dim=3, embed_dim=128, depth=4, num_heads=8, dropout=0.0):
        super().__init__()
        print(
            f"Initializing [ABLATION] SpectralTransformerMLP: embed={embed_dim}, depth={depth}, heads={num_heads}, dropout={dropout}")

        self.embed = LinearEmbed(input_dim, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, input_dim)
        x = self.embed(x)
        # x shape: (Batch, embed_dim)

        # [!!!] Key step: Simulate sequence length 1 for attention mechanism
        # Sequence length = 1 → Attention degenerates to self-token mixing
        x = x.unsqueeze(1)  # x shape becomes: (Batch, 1, embed_dim)

        for blk in self.blocks:
            x = blk(x)  # Operation runs on (Batch, 1, embed_dim)

        # Global Average Pooling (or squeeze): Collapse the sequence dimension (SeqLen=1)
        x = x.mean(dim=1)  # x shape returns to: (Batch, embed_dim)
        return self.head(x)