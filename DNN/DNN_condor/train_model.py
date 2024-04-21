import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tqdm import tqdm
import os
from sklearn.utils import class_weight
import argparse

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

print(signal_filenames)

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='BDT Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    file = args.label

    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training2/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training2/y_train_{file}.npy', allow_pickle=True)
    X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing2/X_test_{file}.npy', allow_pickle=True)
    y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing2/y_test_{file}.npy', allow_pickle=True)

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

    class_counts = np.bincount(y_train.astype(int))
    bkg = class_counts[0]
    sig = class_counts[1]
    total = bkg + sig
    
    print('Training distribution:\n    Total: {}\n    Positive: {} ({:.5f}% of total)\n'.format(
        total, bkg, 100 * sig / total))
    
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
    model = Sequential([
    Dense(128, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(64, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'), 
    Dropout(0.3),
    Dense(32, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') 
])

    optimizer = Adam(learning_rate=0.0001)

    metrics = ['accuracy','loss', 'prc', 'precision', 'recall']

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)

    def scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))

    # Update callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(f'/eos/user/t/tcritchl/DNN/trained_models5/best_model_{file}.keras', save_best_only=True, monitor='val_loss', mode='min'),
        LearningRateScheduler(scheduler)
    ]

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    print(f"class weights (sklearn automatic): {class_weight_dict}")

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks,class_weight=class_weight_dict) #change batch size to contain background slices

    #weight up the minority signal class
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
   # history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=0) #class_weight=class_weight_dict) #20% of the training data will be used as validation
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
    plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots5/feature_importance_{file}.pdf')
    plt.close()

    for metric in ['loss', 'accuracy', 'Precision', 'Recall', 'prc']:
        plt.figure()
        plt.plot(history.history[metric], label=f'Training {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots5/{metric}_{file}.pdf')
        plt.close()

    print("Loading the best model...")
    model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models4/best_model_{file}.keras')
    print("Model loaded successfully.")
    model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.keras')
    print(f"model saved successfully for {file}")
