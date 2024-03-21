import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

# Load preprocessed data
X_train = np.load('X_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)
X_test = np.load('X_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)

# Print out data types and shapes
print("Data types and shapes:")
print("X_train:", X_train.dtype, X_train.shape)
print("y_train:", y_train.dtype, y_train.shape)
print("X_test:", X_test.dtype, X_test.shape)
print("y_test:", y_test.dtype, y_test.shape)

# Print out a sample of the data
print("\nSample of the data:")
print("X_train sample:", X_train[:5])
print("y_train sample:", y_train[:5])
print("X_test sample:", X_test[:5])
print("y_test sample:", y_test[:5])

# Convert lists to numerical values
"""
X_train[:, 3] = [item[0] for item in X_train[:, 3]]
X_train[:, 4] = [item[0] for item in X_train[:, 4]]
X_test[:, 3] = [item[0] for item in X_test[:, 3]]
X_test[:, 4] = [item[0] for item in X_test[:, 4]]
"""
# Convert X_train and X_test to float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Define the DNN model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use 'categorical_crossentropy' for multi-class classification
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
]

# Train the model with tqdm progress bar
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=0)
print("Training completed.")
print(f"plotting curves")
# Plot loss over time
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"loss_function.pdf")


# Load the best model saved by the ModelCheckpoint
print("Loading the best model...")
model = tf.keras.models.load_model('best_model.keras')
print("Model loaded successfully.")

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc*100:.2f}%')

# Save the final model
print("Saving the final model...")
model.save('DNN_HNLs.keras')
print("Final model saved successfully.")

# Plot ROC curve
y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(f"ROC.pdf")

# Get predicted scores for signal and background events
y_pred_signal = model.predict(X_test[y_test == 1]).ravel()
y_pred_background = model.predict(X_test[y_test == 0]).ravel()

# Plot histogram of predicted scores for signal and background events
plt.figure()
plt.hist(y_pred_signal, bins=50, alpha=0.5, color='b', label='Signal')
plt.hist(y_pred_background, bins=50, alpha=0.5, color='r', label='Background')
plt.xlabel('Predicted Score')
plt.ylabel('Frequency')
plt.title('Predicted Scores for Signal and Background Events')
plt.legend(loc='upper center')
plt.grid(True)
plt.savefig(f"dnn_classification.pdf")
