#5k8 98.6%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 1. LOAD AND PREPARE DATA ---

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize image data: scale pixel values from [0, 255] to [0, 1]
# This is a very important step for the model to converge better
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a "channel" dimension to the images (MNIST is grayscale so it only has 1 channel)
# Shape changes from (60000, 28, 28) -> (60000, 28, 28, 1)
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

# Convert labels to one-hot encoding
# Example: digit 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# --- 2. BUILD THE MODEL ---

# Use the Keras Sequential API to build the model layer by layer
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    # Convolutional layer 1
    layers.Conv2D(4, kernel_size=(5, 5), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional layer 2
    layers.Conv2D(8, kernel_size=(5, 5), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten 2D -> 1D
    layers.Flatten(),

    # Dropout to prevent overfitting
    layers.Dropout(0.2),

    # Dense layer 1
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    

    # Dense layer 2 (output layer)
    layers.Dense(num_classes, activation="softmax"),
])

# Print the model structure and parameter count
# This is an extremely useful tool in Keras!
print("Model structure:")
model.summary()

# --- 3. COMPILE THE MODEL ---

# Configure the optimizer, loss function, and metrics for the model
model.compile(
    optimizer="adam", # Popular and effective optimizer
    loss="categorical_crossentropy", # Appropriate loss function for one-hot classification tasks
    metrics=["accuracy"] # Track accuracy
)

# --- 4. TRAIN (FIT) THE MODEL ---

batch_size = 64
epochs = 40 

print("\nStarting model training...")
# The training process will happen here
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1 # Use validation split to evaluate after each epoch
)

# --- 5. EVALUATE THE MODEL ---

print("\nEvaluating model on the test dataset...")
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f} ({score[1]*100:.2f}%)")

model.save("model.h5")
