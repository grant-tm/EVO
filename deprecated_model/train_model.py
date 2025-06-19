from config import SYMBOL, SEQ_LEN, FEATURES, TP_PCT, SL_PCT, LOOKAHEAD, START_TIME, END_TIME, TRAINING_EPOCHS, MODEL_NAME

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.utils import to_categorical

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

print(f"Loading data from preprocessed_training_data.csv")
df = pd.read_csv("preprocessed_training_data.csv")

# =====================================
# LABEL DATA
# =====================================

print("Generating data labels")

# generate labels
def label_row(i):
    entry_price = df.loc[i, "close"]

    for offset in range(1, LOOKAHEAD + 1):
        if i + offset >= len(df):
            break
        future_price = df.loc[i + offset, "close"]
        future_return = (future_price - entry_price) / entry_price

        if future_return > TP_PCT:
            return 1  # Buy — take profit hit
        elif future_return < -SL_PCT:
            return 2  # Sell — stop loss hit

    return 0  # Hold — no decisive move in 3 bars

df["label"] = [label_row(i) if i + LOOKAHEAD < len(df) else 0 for i in range(len(df))]

# =====================================
# FORMAT DATA (FAST VERSION)
# =====================================

print("Formatting data using fast NumPy slicing")

data = df[FEATURES].values.astype(np.float32)
labels = df["label"].values.astype(np.int32)

num_samples = len(df) - SEQ_LEN - LOOKAHEAD
X = np.lib.stride_tricks.sliding_window_view(data, window_shape=(SEQ_LEN,), axis=0)[:num_samples]
X = X.reshape(-1, SEQ_LEN, len(FEATURES))  # Ensure 3D shape

# y = labels[SEQ_LEN:SEQ_LEN + num_samples]  # Align with X
y = labels[SEQ_LEN + LOOKAHEAD - 1 : SEQ_LEN + LOOKAHEAD - 1 + num_samples]

print(f"Formatted {len(X)} samples")

# =====================================
# SPLIT DATA AND COMPUTE CLASS WEIGHTS
# =====================================

# Split chronologically
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

y_train_cat = to_categorical(y_train, num_classes=3)
y_val_cat = to_categorical(y_val, num_classes=3)

print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
print("Label distribution (train):", np.bincount(y_train))
print("Label distribution (val):", np.bincount(y_val))

# Compute class weights: inverse of class frequencies
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weights}")

# =====================================
# DEFINE AND COMPILE MODEL
# =====================================

# Build a simple LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, len(FEATURES))),
    tf.keras.layers.LayerNormalization(),  # Stabilize input range
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# =====================================
# TRAIN MODEL
# =====================================

# Train
history = model.fit(
    X_train[:10000], y_train_cat[:10000],
    validation_data=(X_val, y_val_cat),
    epochs=TRAINING_EPOCHS,
    batch_size=128,
    class_weight=class_weights
)

# Save model
model.save(f"{MODEL_NAME}")
print(f"Model saved to {MODEL_NAME}")

# =====================================
# EVALUATE MODEL
# =====================================

y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
print(f"Prediction distribution: {np.bincount(y_pred)}")
print(f"Label distribution:{np.bincount(y_val)}")

print("Sample prediction probabilities (first 10):")
print(np.round(y_pred_probs[:10], 3))