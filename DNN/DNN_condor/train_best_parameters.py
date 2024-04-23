import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
import argparse

def simple_oversample(X_train, y_train, scale_factor):
    minority_indices = np.where(y_train == 1)[0]
    repeated_minority_indices = np.repeat(minority_indices, int(scale_factor))
    np.random.shuffle(repeated_minority_indices)
    X_oversampled = np.vstack([X_train, X_train[repeated_minority_indices]])
    y_oversampled = np.hstack([y_train, y_train[repeated_minority_indices]])
    return X_oversampled, y_oversampled

def create_model(input_dim, layers=[500, 500, 250, 100, 50], dropout_rate=0.21, learning_rate=0.001):
    model = Sequential()
    for index, layer in enumerate(layers):
        if index == 0:
            model.add(Dense(layer, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
        else:
            model.add(Dense(layer, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser(description='DNN Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    # Load data
    file = args.label
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/y_train_{file}.npy', allow_pickle=True)

    # Data preprocessing
    X_train = X_train.astype(np.float32)
    bkg, sig = np.bincount(y_train.astype(int))
    ratio = bkg / sig
    X_train_oversampled, y_train_oversampled = simple_oversample(X_train, y_train, ratio)

    # Model setup for GridSearchCV
    input_dim = X_train.shape[1]
    model = KerasClassifier(build_fn=lambda: create_model(input_dim=input_dim), epochs=50, batch_size=100, verbose=1)
    param_grid = {
        'layers': [[500, 500, 250, 100, 50], [300, 300, 150]],
        'dropout_rate': [0.2, 0.3],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [50, 100],
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=1)
    grid_result = grid.fit(X_train_oversampled, y_train_oversampled)
    
    best_params = grid.best_params_
    print("Best parameters found:", best_params)

    # Results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for mean, stdev, param in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params']):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    # Final training with best parameters
    final_model = create_model(input_dim=input_dim, **grid_result.best_params_)
    final_model.fit(X_train_oversampled, y_train_oversampled, epochs=best_params.get('epochs', 100), batch_size=best_params['batch_size'], validation_split=0.2)

    # Save the final trained model
    final_model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.h5')
    print("Model training completed and saved.")

if __name__ == "__main__":
    main()
