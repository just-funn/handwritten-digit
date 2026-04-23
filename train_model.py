import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten images from 28x28 to 784
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)

# Build model
print("Building model...")
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
print("Training model...")
model.fit(
    x_train_flat, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save model
model_path = 'models/mnist_model.h5'
model.save(model_path)
print(f"Model saved to {model_path}")