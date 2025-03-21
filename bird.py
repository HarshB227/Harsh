import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow_datasets as tfds
import random

# Load the Stanford Dogs Dataset
dataset_name = 'stanford_dogs'
(ds_train, ds_test), ds_info = tfds.load(dataset_name, 
                                         split=['train[:80%]', 'train[80%:]'], 
                                         as_supervised=True, 
                                         with_info=True)

# Data Preprocessing
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_train = ds_train.map(preprocess_image)
ds_test = ds_test.map(preprocess_image)

# Data Batching and Prefetching
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

ds_train = ds_train.shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# Extract Class Names
class_names = ds_info.features['label'].int2str
num_classes = ds_info.features['label'].num_classes

# Model with Transfer Learning
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the Model
history = model.fit(ds_train, epochs=10, validation_data=ds_test)

# Fine-Tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_fine_tuned = model.fit(ds_train, epochs=10, validation_data=ds_test)

# Evaluate Model
fine_tuned_test_loss, fine_tuned_test_acc = model.evaluate(ds_test)
print(f"Fine-tuned test accuracy: {fine_tuned_test_acc:.4f}")

# Show Predictions with Improved Visualization
def show_predictions(model, dataset, num_images=10):
    plt.figure(figsize=(15, 10))
    test_images = list(dataset.unbatch().take(num_images))
    random.shuffle(test_images)
    
    for i in range(num_images):
        image, label = test_images[i]
        prediction = model.predict(tf.expand_dims(image, axis=0))
        predicted_label = np.argmax(prediction)
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(f"Pred: {class_names(predicted_label)}\nTrue: {class_names(int(label.numpy()))}", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

show_predictions(model, ds_test)
print(f"Final Accuracy: {fine_tuned_test_acc:.4f}")

# Plot Training and Validation Accuracy and Loss
def plot_history(history, history_fine_tuned):
    plt.figure(figsize=(12, 5))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Acc (Initial)')
    plt.plot(history.history['val_accuracy'], label='Val Acc (Initial)')
    plt.plot(history_fine_tuned.history['accuracy'], label='Training Acc (Fine-Tuned)')
    plt.plot(history_fine_tuned.history['val_accuracy'], label='Val Acc (Fine-Tuned)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss (Initial)')
    plt.plot(history.history['val_loss'], label='Val Loss (Initial)')
    plt.plot(history_fine_tuned.history['loss'], label='Training Loss (Fine-Tuned)')
    plt.plot(history_fine_tuned.history['val_loss'], label='Val Loss (Fine-Tuned)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()

plot_history(history, history_fine_tuned)
