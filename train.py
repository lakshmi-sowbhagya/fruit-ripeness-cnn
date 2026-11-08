# train.py - Google Colab (MobileNetV2 transfer learning)
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

ROOT = "/content/banana"        # folder containing train/, valid/, test/

def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        f"{ROOT}/{split}", image_size=(224, 224), batch_size=32,
        label_mode="categorical", shuffle=shuffle, seed=42)

train_ds = load("train", True)
val_ds   = load("valid", False)
test_ds  = load("test", False)
class_names = train_ds.class_names

# Model: augmentation + rescaling + frozen MobileNetV2 + new head
inputs = layers.Input((224, 224, 3))
x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.1)(x)
x = layers.RandomZoom(0.2)(x)
x = layers.Rescaling(1 / 127.5, offset=-1)(x)

base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False

x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Training
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds, validation_data=val_ds, epochs=15,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=4, restore_best_weights=True)])

# Save model and class names
model.save("fruit_ripeness.keras")
json.dump(class_names, open("classes.json", "w"))

# Evaluation on the test set
y_true = np.concatenate([y.numpy().argmax(1) for _, y in test_ds])
y_pred = model.predict(test_ds).argmax(1)
print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_true, y_pred))
