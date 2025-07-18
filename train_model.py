import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def load_images(image_dir, label, img_size=(224, 224)):
    images, labels = [], []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
    return images, labels

# Load real and fake logos
real_images, real_labels = load_images("real_logos", 0)
fake_images, fake_labels = load_images("fake_logos", 1)

# Combine
X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

model.fit(
    X_train, y_train, epochs=10, batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=2
)

if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/saved_model.h5")

print("✅ Model saved to model/saved_model.h5")
