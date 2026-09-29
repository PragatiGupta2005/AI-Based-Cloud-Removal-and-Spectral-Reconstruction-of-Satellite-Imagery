"""
Cross-Attention Fusion Layer for Multi-Modal Geospatial AI in CloudClear AI.
Fuses Optical spectral bands, Historical temporal textures, and Sentinel-1 SAR backscatter.
Uses spatial feature downsampling and latent multi-head attention for high efficiency.
"""

import tensorflow as tf
from tensorflow.keras import layers


class CrossAttentionFusionLayer(layers.Layer):
    """
    Multi-Head Cross-Attention Layer for Optical + SAR + Temporal fusion.
    Queries: Current Optical Features
    Keys & Values: Fused Historical + SAR Context Features
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * 2, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, query_feat, context_feat):
        """
        query_feat: (B, H, W, C) from current optical
        context_feat: (B, H, W, C) from SAR + Historical
        """
        B = tf.shape(query_feat)[0]
        H = tf.shape(query_feat)[1]
        W = tf.shape(query_feat)[2]
        C = query_feat.shape[-1]

        # Reshape to sequence: (B, H*W, C)
        q_seq = tf.reshape(query_feat, [B, H * W, C])
        ctx_seq = tf.reshape(context_feat, [B, H * W, C])

        # Cross attention
        attn_out = self.mha(query=q_seq, key=ctx_seq, value=ctx_seq)
        x = self.norm1(q_seq + attn_out)

        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Reshape back to spatial tensor: (B, H, W, C)
        out = tf.reshape(x, [B, H, W, C])
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config
