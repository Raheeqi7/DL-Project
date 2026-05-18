
# ─────────────────────────────────────────────
# STEP 1: Install & Import Libraries
# ─────────────────────────────────────────────
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                      Dense, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, precision_score,
                              recall_score, f1_score)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# ─────────────────────────────────────────────
# STEP 2: Download Dataset from Kaggle
# ─────────────────────────────────────────────
"""
from google.colab import files
files.upload()   

# Download the dataset:
!kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
!unzip -q brain-mri-images-for-brain-tumor-detection.zip -d brain_mri_data
"""

# ─────────────────────────────────────────────
# STEP 3: Load & Preprocess Data
# ─────────────────────────────────────────────

IMG_SIZE = 224       # Resize all images to 224x224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# Dataset paths — adjust if your folder structure differs
DATA_DIR = "brain_mri_data"
YES_DIR = os.path.join(DATA_DIR, "yes")   # Tumor images
NO_DIR  = os.path.join(DATA_DIR, "no")    # No-tumor images

def load_images(folder, label):
    """Load all images from a folder and assign a binary label."""
    images, labels = [], []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
    return images, labels

print("Loading images...")
tumor_images, tumor_labels = load_images(YES_DIR, label=1)
normal_images, normal_labels = load_images(NO_DIR, label=0)

print(f"  Tumor images:    {len(tumor_images)}")
print(f"  No-tumor images: {len(normal_images)}")

# Combine and convert to numpy arrays
X = np.array(tumor_images + normal_images, dtype=np.float32)
y = np.array(tumor_labels + normal_labels, dtype=np.float32)

# Normalize pixel values to [0, 1]
X = X / 255.0

print(f"\nDataset shape: {X.shape}")
print(f"Labels — Tumor: {np.sum(y==1)}, No Tumor: {np.sum(y==0)}")


# ─────────────────────────────────────────────
# STEP 4: Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────

# Plot sample images
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample MRI Images", fontsize=14, fontweight='bold')

for i, ax in enumerate(axes[0]):
    idx = np.where(y == 1)[0][i]
    ax.imshow(X[idx])
    ax.set_title("Tumor", color='red', fontsize=10)
    ax.axis('off')

for i, ax in enumerate(axes[1]):
    idx = np.where(y == 0)[0][i]
    ax.imshow(X[idx])
    ax.set_title("No Tumor", color='green', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig("sample_images.png", dpi=150, bbox_inches='tight')
plt.show()

# Class distribution bar chart
fig, ax = plt.subplots(figsize=(6, 4))
classes = ['No Tumor', 'Tumor']
counts  = [int(np.sum(y==0)), int(np.sum(y==1))]
colors  = ['#2ecc71', '#e74c3c']
bars = ax.bar(classes, counts, color=colors, width=0.4, edgecolor='white')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(count), ha='center', va='bottom', fontweight='bold')
ax.set_title("Class Distribution", fontweight='bold')
ax.set_ylabel("Number of Images")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────
# STEP 5: Train / Test Split
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # maintain class ratio in both sets
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ─────────────────────────────────────────────
# STEP 6: Data Augmentation
# ─────────────────────────────────────────────

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for test set — only rescaling 
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
test_generator  = test_datagen.flow(X_test,  y_test,  batch_size=BATCH_SIZE, shuffle=False)

print("Data augmentation configured.")


# ─────────────────────────────────────────────
# STEP 7: Build CNN Model Architecture
# ─────────────────────────────────────────────

def build_cnn_model(input_shape=(224, 224, 3)):
    """

    Architecture:
    - 3 convolutional blocks (Conv2D + BatchNorm + MaxPooling)
    - Flatten + Dense layers
    - Dropout for regularization
    - Sigmoid output for binary classification
    """
    model = Sequential([

        # ── Block 1 ──────────────────────────
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=input_shape, name='conv1'),
        BatchNormalization(name='bn1'),
        MaxPooling2D((2, 2), name='pool1'),

        # ── Block 2 ──────────────────────────
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        BatchNormalization(name='bn2'),
        MaxPooling2D((2, 2), name='pool2'),

        # ── Block 3 ──────────────────────────
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        BatchNormalization(name='bn3'),
        MaxPooling2D((2, 2), name='pool3'),

        # ── Classifier ───────────────────────
        Flatten(name='flatten'),
        Dense(256, activation='relu', name='fc1'),
        Dropout(0.5, name='dropout'),
        Dense(1, activation='sigmoid', name='output')   # Binary: 0 or 1
    ], name='BrainTumorCNN')

    return model

model = build_cnn_model()
model.summary()


# ─────────────────────────────────────────────
# STEP 8: Compile the Model
# ─────────────────────────────────────────────

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model compiled successfully.")


# ─────────────────────────────────────────────
# STEP 9: Define Callbacks
# ─────────────────────────────────────────────

callbacks = [
    # Stop training early if validation loss stops improving
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Save the best model checkpoint
    ModelCheckpoint(
        filepath='brain_tumor_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Reduce learning rate when validation loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print("Callbacks configured.")


# ─────────────────────────────────────────────
# STEP 10: Train the Model
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("Starting model training...")
print("="*50 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete!")


# ─────────────────────────────────────────────
# STEP 11: Plot Training Curves
# ─────────────────────────────────────────────

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Training History', fontsize=14, fontweight='bold')

    # Accuracy
    ax1.plot(history.history['accuracy'],     label='Train Accuracy',      color='#2980b9', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#e74c3c', linewidth=2, linestyle='--')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Loss
    ax2.plot(history.history['loss'],     label='Train Loss',      color='#2980b9', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2, linestyle='--')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: training_curves.png")

plot_training_history(history)


# ─────────────────────────────────────────────
# STEP 12: Evaluate the Model
# ─────────────────────────────────────────────

print("\n" + "="*50)
print("Model Evaluation on Test Set")
print("="*50)

# Predict on test set
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# Metrics
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"\n  Test Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Test Loss      : {test_loss:.4f}")
print(f"  Precision      : {precision:.4f}")
print(f"  Recall         : {recall:.4f}")
print(f"  F1-Score       : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))


# ─────────────────────────────────────────────
# STEP 13: Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'],
                linewidths=0.5, ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=13)
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: confusion_matrix.png")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives  (Tumor correctly detected): {tp}")
    print(f"  True Negatives  (No tumor correctly rejected): {tn}")
    print(f"  False Positives (Healthy flagged as tumor): {fp}")
    print(f"  False Negatives (Tumor MISSED — critical!): {fn}")

plot_confusion_matrix(y_test, y_pred)


# ─────────────────────────────────────────────
# STEP 14: Predict on New Images
# ─────────────────────────────────────────────

def predict_image(model, image_path, threshold=0.5):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    prob = model.predict(img_input, verbose=0)[0][0]
    label = "TUMOR DETECTED" if prob >= threshold else "NO TUMOR"
    confidence = prob if prob >= threshold else (1 - prob)

    # Visualize result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(img_rgb)
    ax1.set_title("Input MRI Image", fontweight='bold')
    ax1.axis('off')

    color = '#e74c3c' if prob >= threshold else '#2ecc71'
    ax2.barh(['No Tumor', 'Tumor'], [1 - prob, prob],
             color=['#2ecc71', '#e74c3c'], edgecolor='white')
    ax2.set_xlim([0, 1])
    ax2.set_title(f"Prediction: {label}\nConfidence: {confidence*100:.1f}%",
                  fontweight='bold', color=color)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Threshold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches='tight')
    plt.show()

    return {"label": label, "confidence": f"{confidence*100:.1f}%", "probability": float(prob)}

# Example usage:
# result = predict_image(model, "path/to/your/mri_image.jpg")
# print(result)


# ─────────────────────────────────────────────
# STEP 15: Save Model
# ─────────────────────────────────────────────

model.save("brain_tumor_model.h5")
print("\nModel saved as: brain_tumor_model.h5")


print("\n" + "="*50)
print("ALL DONE! Files saved:")
print("  brain_tumor_model.h5    — trained model")
print("  training_curves.png     — accuracy/loss graphs")
print("  confusion_matrix.png    — confusion matrix")
print("  sample_images.png       — EDA visualization")
print("  class_distribution.png  — class balance chart")
print("="*50)
