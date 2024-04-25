'''
    File: transformer_encoder.py
    Source: Pluralsight blog
        (https://www.pluralsight.com/resources/blog/data/how-build-large-language-model)
    Summary: Transformer encoder layer, processes data

    d_model: The dimensionality of the input (and output) of the layer.
    num_heads: The number of heads in the multi-head attention mechanism.
    dff: The dimensionality of the inner layer in the feed-forward network.
    rate: The dropout rate used for regularization
'''

import tensorflow as tf

# scribes, absorbes information
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.MultiHeadAttention(num_heads, d_model)
        self.ffn = tf.keras.Sequential([
            tf.Dense(dff, activation='relu'),
            tf.Dense(d_model)
        ])
        # Layer Normalization and Dropout
        # Helps in stablizing the output of each layer
        # Helps prevent overfitting
        self.layernorm1 = tf.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training):
        # attention and feed-forward operations
        attn_output = self.mha(x,x,x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
    
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2