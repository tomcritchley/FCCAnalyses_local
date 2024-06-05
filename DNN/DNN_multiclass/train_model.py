import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, Callback
from tqdm import tqdm
import os
import seaborn as sns
from sklearn.utils import class_weight
import argparse
from imblearn.over_sampling import SMOTE
import pandas as pd
import shap

base_HNL = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/"

class DynamicWeightsCallback(Callback):
    def __init__(self, validation_data, initial_weights, increase_factor=1.5, decrease_factor=0.75, patience=3, min_delta=0.01):
        super().__init__()
        self.validation_data = validation_data
        self.weights = initial_weights
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_recall = 0
        self.best_precision = 0

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        predictions = self.model.predict(X_val)
        val_recall = recall_score(y_val, predictions.round())
        val_precision = precision_score(y_val, predictions.round())

        # Check for improvements
        if (val_recall - self.best_recall) < self.min_delta and (val_precision - self.best_precision) < self.min_delta:
            self.wait += 1
        else:
            self.wait = 0
            self.best_recall = max(val_recall, self.best_recall)
            self.best_precision = max(val_precision, self.best_precision)

        # If no improvement for 'patience' epochs, adjust weights
        if self.wait >= self.patience:
            if val_recall < 0.96:
                self.weights[1] *= self.increase_factor
            if val_precision < 0.96:
                self.weights[1] *= self.decrease_factor

            # Reset wait counter after adjusting
            self.wait = 0

        # Log the updated weights
        print(f"Updated weights: {self.weights}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNN Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    file = args.label

    for i in range(3):
        print(f"Training on dataset: train_{i}")

        X_train = np.load(f'/eos/user/t/tcritchl/DNN/training20/X_train_{i}_{file}.npy', allow_pickle=True)
        y_train = np.load(f'/eos/user/t/tcritchl/DNN/training20/y_train_{i}_{file}.npy', allow_pickle=True)
        X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing20/X_test_{file}.npy', allow_pickle=True)
        y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing20/y_test_{file}.npy', allow_pickle=True)
        weights_train = np.load(f'/eos/user/t/tcritchl/DNN/training20/weights_train_{i}_{file}.npy', allow_pickle=True)
        
        print("Data types and shapes:")
        print("X_train:", X_train.dtype, X_train.shape)
        print("y_train:", y_train.dtype, y_train.shape)
        print("X_test:", X_test.dtype, X_test.shape)
        print("y_test:", y_test.dtype, y_test.shape)

        print("\nSample of the data:")
        print("X_train sample:", X_train[:5])
        print("y_train sample:", y_train[:5])
        print("X_test sample:", X_test[:5])
        print("y_test sample:", y_test[:5])

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        signal_weight_factor = 3
        background_weight_factor = 1

        adjusted_weights = np.where(y_train == 1, 
                                    weights_train * signal_weight_factor, 
                                    weights_train * background_weight_factor)

        class_counts = np.bincount(y_train.astype(int))
        bkg = class_counts[0]
        sig = class_counts[1]
        total = bkg + sig

        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

        X_val = X_train[:int(len(X_train) * 0.2)] 
        y_val = y_train[:int(len(y_train) * 0.2)]

        initial_weights = {0: 1, 1: 1}
        dynamic_weights_cb = DynamicWeightsCallback(validation_data=(X_val, y_val), initial_weights=initial_weights)

        print("X_validation shape:", X_val.shape)
        print("y_validation shape:", y_val.shape)  

        print('Training background distribution:\n    Total: {}\n    Background: {} ({:.5f}% of total)\n'.format(
            total, bkg, 100 * bkg / total))

        print('Training signal distribution:\n    Total: {}\n    Signal: {} ({:.5f}% of total)\n'.format(
            total, sig, 100 * sig / total))

        class_counts = np.bincount(y_test.astype(int))
        bkg_test = class_counts[0]
        sig_test = class_counts[1]
        total_test = bkg_test + sig_test

        print('Testing distribution:\n    Total: {}\n    Positive: {} ({:.5f}% of total)\n'.format(
            total_test, bkg_test, 100 * sig_test / total_test))

        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)), 
            Dropout(0.1),
            Dense(64, activation='relu'),  
            Dropout(0.1),  
            Dense(1, activation='sigmoid')  
        ])

        optimizer = Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc'), AUC(name='prc', curve='PR')])

        def scheduler(epoch, lr):
            if epoch < 10:
                return float(lr)
            else:
                return float(lr * tf.math.exp(-0.1))

        print(f'Average class probability in training set:   {y_train.mean():.4f}')
        print(f'Average class probability in validation set: {y_val.mean():.4f}')
        print(f'Average class probability in test set:       {y_test.mean():.4f}')

        callbacks = [
            ModelCheckpoint(f'/eos/user/t/tcritchl/DNN/trained_models20/best_model_{file}_train_{i}.keras', save_best_only=True, monitor='val_prc', mode='max'),
            LearningRateScheduler(scheduler)
        ]

        history = model.fit(X_train, y_train, epochs=100, sample_weight=adjusted_weights, batch_size=256, validation_data=(X_val, y_val), callbacks=callbacks)
        print("Training completed.")
        print(f"plotting curves")

        for metric in ['loss', 'accuracy', 'precision', 'recall', 'prc']:
            plt.figure()
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Training and Validation {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots20/{metric}_{file}_train_{i}.pdf')
            plt.close()

        print("Loading the best model...")
        model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models20/best_model_{file}_train_{i}.keras')
        print("Model loaded successfully.")
        model.save(f'/eos/user/t/tcritchl/DNN/trained_models20/DNN_HNLs_{file}_train_{i}.keras')
        print(f"model saved successfully for {file} train_{i}")
