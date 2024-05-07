import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
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

def simple_oversample(X_train, y_train, scale_factor):
    
    minority_indices = np.where(y_train == 1)[0]
    repeat_count = int(scale_factor)  # Ensure each instance is repeated equally
    
    # Extend minority indices by repeating each exactly `repeat_count` times
    repeated_minority_indices = np.repeat(minority_indices, repeat_count)
    
    # Shuffle to mix them up for training
    np.random.shuffle(repeated_minority_indices)
    
    X_oversampled = np.vstack([X_train, X_train[repeated_minority_indices]])
    y_oversampled = np.hstack([y_train, y_train[repeated_minority_indices]])
    
    return X_oversampled, y_oversampled

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
            if val_recall < 0.98:
                self.weights[1] *= self.increase_factor
            if val_precision < 0.98:
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

    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training10/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training10/y_train_{file}.npy', allow_pickle=True)
    X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing10/X_test_{file}.npy', allow_pickle=True)
    y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing10/y_test_{file}.npy', allow_pickle=True)
    weights_train = np.load(f'/eos/user/t/tcritchl/DNN/testing10/weights_train_{file}.npy', allow_pickle=True)
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

    print(f"first few entries of weights for training: {weights_train[:10]}")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    class_counts = np.bincount(y_train.astype(int))
    bkg = class_counts[0]
    sig = class_counts[1]
    total = bkg + sig
    
    print("Initial X_train shape:", X_train.shape)
    print("Initial y_train shape:", y_train.shape)
    #defining validation data for more control
    X_val = X_train[:int(len(X_train) * 0.2)] 
    y_val = y_train[:int(len(y_train) * 0.2)]
    dynamic_weights_cb = DynamicWeightsCallback(validation_data=(X_val, y_val), initial_weights=weights_train)

    print("Initial X_train shape:", X_val.shape)
    print("Initial y_train shape:", y_val.shape)  
    
    print('Training background distribution:\n    Total: {}\n    Background: {} ({:.5f}% of total)\n'.format(
        total, bkg, 100 * bkg / total))
    
    print('Training signal distribution:\n    Total: {}\n    Signal: {} ({:.5f}% of total)\n'.format(
        total, sig, 100 * sig / total))
    
    weight_for_0 = (1 / bkg) * (total / 2.0)
    weight_for_1 = (1 / sig) * (total / 2.0)

    weights = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0 (tensorflow tutorial): {:.2f}'.format(weight_for_0))
    print('Weight for class 1: (tensorflow tutorial) {:.2f}'.format(weight_for_1))
        
    class_counts = np.bincount(y_test.astype(int))
    bkg_test = class_counts[0]
    sig_test = class_counts[1]
    total_test = bkg_test + sig_test

    print('Testing distribution:\n    Total: {}\n    Positive: {} ({:.5f}% of total)\n'.format(
        total_test, bkg_test, 100 * sig_test / total_test))
    
    print(f"initialising SMOTE...")

    smote = SMOTE()
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    #how does smote affect distributions?

    signal_indices_original = np.where(y_train == 1)[0]
    signal_indices_smote = np.where(y_train_smote == 1)[0]

    X_train_signal = X_train[signal_indices_original, :]
    X_train_smote_signal = X_train_smote[signal_indices_smote, :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(X_train_signal[:, 0], ax=axes[0], kde=True, color='blue', label='Original')  # Replace 0 with the feature index you are interested in
    sns.histplot(X_train_smote_signal[:, 0], ax=axes[1], kde=True, color='red', label='SMOTE')
    axes[0].set_title('Original Distribution of Feature 1')
    axes[1].set_title('Distribution of Feature 1 After SMOTE')
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots11/smote_effect_{file}.pdf")
    plt.close()

    class_counts = np.bincount(y_train_smote.astype(int))
    bkg_smote = class_counts[0]
    sig_smote = class_counts[1]
    total = bkg_smote + sig_smote
    
    ratio = bkg / sig
    print(f"YOUR RATIO! ratio of background to signal = {ratio}")
    X_train_oversampled, y_train_oversampled = simple_oversample(X_train, y_train, scale_factor=ratio)

    class_counts = np.bincount(y_train_oversampled.astype(int))
    bkg_oversampled = class_counts[0]
    sig_oversampled = class_counts[1]
    total = bkg_oversampled + sig_oversampled

    print(f"""
    Some statistics after oversampling:
    - Oversampled background: {bkg_oversampled}
    - Oversampled signal: {sig_oversampled}
    - Total events before oversampling: {total_test}
    - Total oversampled events in training: {total}
    """)

    signal_indices_oversampled = np.where(y_train_oversampled == 1)[0]
    X_train_oversampled_signal = X_train_oversampled[signal_indices_oversampled, :]

    #how does random scale factor affect distrubtions?

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(X_train_signal[:, 0], ax=axes[0], kde=True, color='blue', label='Original')  # Replace 0 with the feature index you are interested in
    sns.histplot(X_train_oversampled_signal[:, 0], ax=axes[1], kde=True, color='red', label='Scale Factor')
    axes[0].set_title('Original Distribution of Feature 1')
    axes[1].set_title('Distribution of Feature 1 After Scale factor')
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots11/scale_factor_effect_{file}.pdf")
    plt.close()


    print('Training background distribution:\n    Total: {}\n    Background: {} ({:.5f}% of total)\n'.format(
        total, bkg_smote, 100 * bkg_smote / total))
    
    print('Training signal distribution:\n    Total: {}\n    Signal: {} ({:.5f}% of total)\n'.format(
        total, sig_smote, 100 * sig_smote / total))

    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
    ])
    model.compile(optimizer='adam',
                loss='binary_crossentropy',  # Use 'categorical_crossentropy' for multi-class classification
                metrics=['accuracy'])
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True),
        ModelCheckpoint(f'/eos/user/t/tcritchl/DNN/trained_models5/best_model_{file}.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    ]
    """
    initial_bias = np.log([sig/bkg])

    model = Sequential([
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
    ])

    optimizer = Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall', AUC(name='prc', curve='PR')])

    def scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))
    
    # Update callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(f'/eos/user/t/tcritchl/DNN/trained_models11/best_model_{file}.keras', save_best_only=True, monitor='val_loss', mode='min'),
        LearningRateScheduler(scheduler),
        dynamic_weights_cb
    ]

    #class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    #class_weight_dict = dict(enumerate(class_weights))
    #class_weight_dict = {0: 0.5, 1: 12.5}
    #print(f"class weights (sklearn automatic): {class_weight_dict}")
    #val_start_index = int(len(y_train) * (1 - 0.3))
    #y_val = y_train[val_start_index:]

    print(f'Average class probability in training set:   {y_train.mean():.4f}')
    print(f'Average class probability in validation set: {y_val.mean():.4f}')
    print(f'Average class probability in test set:       {y_test.mean():.4f}')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(f'/eos/user/t/tcritchl/DNN/trained_models11/best_model_{file}.keras', save_best_only=True, monitor='val_loss', mode='min'),
        LearningRateScheduler(scheduler),
        dynamic_weights_cb
    ]

    signal_weight_factor = 10
    background_weight_factor = 1 

    adjusted_weights = np.where(y_train == 1, 
                                weights_train * signal_weight_factor, 
                                weights_train * background_weight_factor)


    #history = model.fit(X_train, y_train,sample_weight=adjusted_weights, epochs=100, batch_size=156, validation_split=0.2, callbacks=callbacks) #,class_weight=class_weight_dict)
    history = model.fit(X_train, y_train,sample_weight=weights_train, epochs=100, batch_size=156, validation_data=(X_val, y_val), callbacks=callbacks)
    print("Training completed.")
    print(f"plotting curves")
    
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
    plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots11/feature_importance_{file}.pdf')
    plt.close()

    for metric in ['loss', 'accuracy', 'precision', 'recall', 'prc']:
        plt.figure()
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots11/{metric}_{file}.pdf')
        plt.close()

    print("Loading the best model...")
    model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models11/best_model_{file}.keras')
    print("Model loaded successfully.")
    model.save(f'/eos/user/t/tcritchl/DNN/trained_models11/DNN_HNLs_{file}.keras')
    print(f"model saved successfully for {file}")
