
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------

DATASET_PATH = "dataset/plantvillage dataset/color"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# --------------------------------
# 2. LOAD TRAINING DATASET
# --------------------------------

print("Loading PlantVillage dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# --------------------------------
# 3. LOAD VALIDATION DATASET
# --------------------------------

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.20,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# --------------------------------
# 4. SAVE CLASS NAMES
# --------------------------------

class_names = train_ds.class_names

with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("Total disease categories:", len(class_names))
print("Disease classes:", class_names)

# --------------------------------
# 5. OPTIMIZE DATA LOADING
# --------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=AUTOTUNE
)

# --------------------------------
# 6. IMAGE AUGMENTATION
# --------------------------------

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# --------------------------------
# 7. BUILD CNN MODEL
# --------------------------------

model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    data_augmentation,

    layers.Rescaling(1./255),

    # First convolution block
    layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(2, 2),

    # Second convolution block
    layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(2, 2),

    # Third convolution block
    layers.Conv2D(
        128, (3, 3), activation="relu"
    ),
    layers.MaxPooling2D(2, 2),

    # Classification layers
    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# --------------------------------
# 8. COMPILE CNN MODEL
# --------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN Model Summary:")
model.summary()

# --------------------------------
# 9. CALLBACKS
# --------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "plant_disease_model.keras",
    monitor="val_loss",
    save_best_only=True
)

# --------------------------------
# 10. TRAIN THE MODEL
# --------------------------------

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# --------------------------------
# 11. EVALUATE MODEL
# --------------------------------

loss, accuracy = model.evaluate(val_ds)

print("\nTraining completed!")

print("Validation Loss:", loss)

print(
    "Validation Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# --------------------------------
# 12. SAVE FINAL MODEL
# --------------------------------

model.save("plant_disease_model.keras")

print("Model saved successfully!")

# --------------------------------
# 13. PLOT ACCURACY
# --------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_graph.png")
plt.close()

# --------------------------------
# 14. PLOT LOSS
# --------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_graph.png")
plt.close()

print("Accuracy and loss graphs saved!")
