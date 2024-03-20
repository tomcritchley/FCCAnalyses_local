import tensorflow as tf
import numpy as np

# Load the test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load the model
model = tf.keras.models.load_model('DNN_HNLs.keras')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# If you want to make predictions on new data
# predictions = model.predict(X_new)

