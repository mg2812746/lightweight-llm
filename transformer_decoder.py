'''
    File: transformer_decoder.py
    Source: Pluralsight blog
        (https://www.pluralsight.com/resources/blog/data/how-build-large-language-model)
    Summary: Transformer decoder layer

    

    
'''
import tensorflow as tf

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = tf.MultiHeadAttention(num_heads, d_model)
        self.mha2 = tf.MultiHeadAttention(num_heads, d_model)

        self.ffn = tf.keras.Sequential([
            tf.Dense(dff, activation='relu'), 
            tf.Dense(d_model)
        ])

        self.layernorm1 = tf.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2