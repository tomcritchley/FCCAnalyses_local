import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from keras_tuner import RandomSearch, HyperParameters
import argparse

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                    input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                        activation='relu'))
        model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=keras.optimizers.Adam(
                  hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', AUC(name='prc', curve='PR')])
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNN Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label', required=True)
    args = parser.parse_args()

    # Assume the loading code for data and other setups are already here

    tuner = RandomSearch(
        build_model,
        objective='val_prc',  # Assuming you're focusing on precision-recall curve
        max_trials=10,  # Number of variations on hyperparameters
        executions_per_trial=1,  # Number of models to train for each trial
        directory='model_tuning',
        project_name=f'tuning_{args.label}'
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=callbacks)

    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Save the best model
    best_model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/best_model_{args.label}.keras')
    
    print("Best model saved successfully.")
    # Evaluate the best model on the test set, plot results, etc.
