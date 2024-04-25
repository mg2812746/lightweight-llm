import tensorflow as tf
from transformer import Transformer
from utils import create_masks, loss_function
from dataset import load_dataset

# Set hyperparameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

# Initialize Transformer model
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=10000,  # Example: input vocabulary size
    target_vocab_size=10000,  # Example: target vocabulary size
    pe_input=1000,  # Example: maximum input sequence length
    pe_target=1000,  # Example: maximum target sequence length
    rate=dropout_rate
)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Define loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# Load dataset
train_dataset, val_dataset = load_dataset(batch_size=64)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    # Iterate through batches
    for (batch, (inp, tar)) in enumerate(train_dataset):
        # Create masks
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, tar)

        # Perform gradient tape to calculate gradients
        with tf.GradientTape() as tape:
            # Forward pass through the model
            predictions = transformer(
                inp, tar, True, enc_padding_mask, combined_mask, dec_padding_mask)
            # Compute loss
            loss = loss_function(tar, predictions, loss_object)

        # Calculate gradients
        gradients = tape.gradient(loss, transformer.trainable_variables)
        # Apply gradients
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))

        # Track total loss
        total_loss += loss

        # Print progress
        if batch % 100 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {loss.numpy():.4f}')

    # Print epoch loss
    print(f'Epoch {epoch + 1} Loss {total_loss / (batch + 1):.4f}')
