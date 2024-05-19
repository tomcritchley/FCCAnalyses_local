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
from tensorflow.keras.metrics import AUC
from imblearn.over_sampling import SMOTE
import pandas as pd
import shap

base_HNL = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/"

masses = [
    "10GeV",
    "20GeV",
    "30GeV",
    "40GeV",
    "50GeV",
    "60GeV",
    "70GeV",
    "80GeV",   
]

couplings = [
    "1e-2", 
    "1e-2p5", 
    "1e-3", 
    "1e-3p5", 
    "1e-4", 
    "1e-4p5", 
    "1e-5"
]

signal_filenames = []

for mass in masses:
    for coupling in couplings:
        base_file = f"HNL_Dirac_ejj_{mass}_{coupling}Ve.root"
        signal_file = os.path.join(base_HNL, base_file)
        if os.path.exists(signal_file):
            signal_filenames.append(signal_file)
        else:
            print(f"file {signal_file} does not exist, moving to next file")

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
        
    parser = argparse.ArgumentParser(description='BDT Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    file = args.label

    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training16/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training16/y_train_{file}.npy', allow_pickle=True)
    X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing16/X_test_{file}.npy', allow_pickle=True)
    y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing16/y_test_{file}.npy', allow_pickle=True)
    weights_train = np.load(f'/eos/user/t/tcritchl/DNN/testing16/weights_train_{file}.npy', allow_pickle=True)
    
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

    #print(f"first few entries of weights for training: {weights_train[:10]}")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    #adjust weights from just cross section
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

    #defining validation data for more control
    X_val = X_train[:int(len(X_train) * 0.2)] 
    y_val = y_train[:int(len(y_train) * 0.2)]
    """ total_validation_samples = int(len(X_train) * 0.2)  # 20% of the training data for validation
        num_positive_val = int(total_validation_samples * 0.0042798)  # 0.42798% are positive
        num_negative_val = total_validation_samples - num_positive_val  # Remaining are negative
        positive_indices = np.where(y_train == 1)[0]
        negative_indices = np.where(y_train == 0)[0]
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)

        val_indices = np.concatenate([
            np.random.choice(positive_indices, num_positive_val, replace=False),
            np.random.choice(negative_indices, num_negative_val, replace=False)
        ])
        np.random.shuffle(val_indices)
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        train_indices = np.setdiff1d(np.arange(len(X_train)), val_indices)
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        """
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
    ##model for 12,
    """model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.1),  
    Dense(1, activation='sigmoid') 
    ])"""
    ## model for 13 ##

    model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)), 
    Dropout(0.1),
    Dense(64, activation='relu'),  
    Dropout(0.1),  
    Dense(1, activation='sigmoid')  
    ])


    ## model for 7,8,9,10 ###
    """    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Reduced from 500 to 256 neurons
        Dropout(0.2),  # Slightly lower dropout for less complex model
        Dense(128, activation='relu'),  # Reduced layer size
        Dropout(0.2),  # Adjusted dropout
        Dense(1, activation='sigmoid')  # Output layer remains the same
        ])
    """
    ##model for test 4,5,6, 11###
    """    model = Sequential([
            Dense(500, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2), 
            Dense(500,activation='relu'),
            Dropout(0.5),
            Dense(250,activation='relu'),
            Dropout(0.5),
            Dense(100,activation='relu'),
            Dropout(0.5),
            Dense(50,activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])"""
    ### model for test 1, 2, 3###
    """        
        model = Sequential([
            Dense(512, input_shape=(X_train.shape[1],)),
            LeakyReLU(alpha=0.01),
            BatchNormalization(),
            Dropout(0.3), 
            
            Dense(512),
            LeakyReLU(alpha=0.01),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256),
            LeakyReLU(alpha=0.01),
            Dropout(0.3),
            
            Dense(128),
            LeakyReLU(alpha=0.01),
            Dropout(0.3),
            
            Dense(64),
            LeakyReLU(alpha=0.01),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])"""
        
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
        EarlyStopping(monitor='val_loss', mode='max', patience=15, restore_best_weights=True),
        ModelCheckpoint(f'/eos/user/t/tcritchl/DNN/trained_models16/best_model_{file}.keras', save_best_only=True, monitor='val_prc', mode='max'),
        LearningRateScheduler(scheduler),
        dynamic_weights_cb
    ]
   
    #weights = {0: 1, 1: 2}
    #sample_weight=weights_train
    history = model.fit(X_train, y_train, epochs=100, batch_size=256, validation_data=(X_val, y_val), callbacks=callbacks) #sample_weight=adjusted_weights
    print("Training completed.")
    print(f"plotting curves")
    """
    def permutation_feature_importance(model, X, y, metric=accuracy_score):
       
        original_score = metric(y, model.predict(X).round())
        feature_importance = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            permuted_score = metric(y, model.predict(X_permuted).round())
            
            feature_importance[i] = original_score - permuted_score

        return feature_importance
    importances = permutation_feature_importance(model, X_test, y_test)
    print("Feature importances:", importances)

    ### plotting feature importance ###
    plt.figure()
    plt.bar(range(X_test.shape[1]), importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots16/feature_importance_{file}.pdf')
    plt.close()
    """
    for metric in ['loss', 'accuracy', 'precision', 'recall', 'prc']:
        plt.figure()
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots16/{metric}_{file}.pdf')
        plt.close()

    print("Loading the best model...")
    model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models16/best_model_{file}.keras')
    print("Model loaded successfully.")
    model.save(f'/eos/user/t/tcritchl/DNN/trained_models16/DNN_HNLs_{file}.keras')
    print(f"model saved successfully for {file}")
