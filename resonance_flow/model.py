from flax import linen as nn


class TransformerCoordinatePredictor(nn.Module):
    """
    A simple Transformer that takes an amino acid sequence (as token IDs)
    and predicts 3D coordinates for each token.
    """

    vocab_size: int = 21
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    max_len: int = 512
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        """
        Predicts 3D coordinates.

        Args:
            x: (batch_size, seq_len) token IDs.
            deterministic: True for eval, False for train.
        """
        batch_size, seq_len = x.shape

        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)

        pos_emb = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_len, self.d_model),
        )
        x = x + pos_emb[:, :seq_len, :]
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.num_layers):
            y = nn.LayerNorm()(x)
            y = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.d_model,
                out_features=self.d_model,
            )(y)
            x = x + nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

            y = nn.LayerNorm()(x)
            y = nn.Dense(features=self.d_model * 4)(y)
            y = nn.gelu(y)
            y = nn.Dense(features=self.d_model)(y)
            x = x + nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        x = nn.LayerNorm()(x)
        coords = nn.Dense(features=3, kernel_init=nn.initializers.normal(stddev=1e-3))(x)
        return coords
