import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.regularizers import l2
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

    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/y_train_{file}.npy', allow_pickle=True)
    X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing5/X_test_{file}.npy', allow_pickle=True)
    y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing5/y_test_{file}.npy', allow_pickle=True)

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
    """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
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
    
    #weight up the minority signal class
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks, verbose=0,class_weight=class_weight_dict) #20% of the training data will be used as validation
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
    
    ### plot loss function ###
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/loss_function_{file}.pdf")
    plt.close()
    
    print("Loading the best model...")
    model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models4/best_model_{file}.keras')
    print("Model loaded successfully.")
    model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.keras')
    print(f"model saved successfully for {file}")
