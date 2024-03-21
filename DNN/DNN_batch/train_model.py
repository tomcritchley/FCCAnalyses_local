import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import os

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

for filename in signal_filenames:
    
    file_parts = filename.split('/')
    final_part = file_parts[-1]
    info_parts = final_part.split('_')

    file = '_'.join(info_parts[3:5]).replace('Ve.root', '')

    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training1/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training1/y_train_{file}.npy', allow_pickle=True)
    X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing1/X_test_{file}.npy', allow_pickle=True)
    y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing1/y_test_{file}.npy', allow_pickle=True)

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
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min', restore_best_weights=True),
        ModelCheckpoint(f'/eos/user/t/tcritchl/DNN/trained_models1/best_model_{file}.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    ]

    # Train the model with tqdm progress bar
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=0) #20% of the training data will be used as validation
    print("Training completed.")
    print(f"plotting curves")
    # Plot loss over time
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots1/loss_function_{file}.pdf")

    # Load the best model saved by the ModelCheckpoint
    print("Loading the best model...")
    model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models1/best_model_{file}.keras')
    print("Model loaded successfully.")
    model.save(f'/eos/user/t/tcritchl/DNN/trained_models1/DNN_HNLs_{file}.keras')
    print(f"model saved successfully for {file}")