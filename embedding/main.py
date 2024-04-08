# main.py

import tensorflow as tf
from config import *
from transformers import Transformer

# Example data (random tensors for demonstration)
num_samples = 10000
inputs = tf.random.uniform((num_samples, MAX_SEQ_LENGTH), maxval=INPUT_VOCAB_SIZE, dtype=tf.int32)
outputs = tf.random.uniform((num_samples, MAX_SEQ_LENGTH), maxval=OUTPUT_VOCAB_SIZE, dtype=tf.int32)

# Create an instance of the Transformer model
transformer = Transformer(num_layers=NUM_LAYERS,
                          d
