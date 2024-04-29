import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC
from keras_tuner import RandomSearch, HyperParameters, Objective
import argparse
import shap

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                    input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                        activation='relu'))
        model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=keras.optimizers.Adam(
                  hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', AUC(name='prc', curve='PR')])
    
    return model


def shap_feature_importance(file, model, X_train):

    feature_names = [
        r'$\Delta R_{jj}$',
        r'$\Delta R_{ejj}$',
        r'$D_0$',
        r'Dijet $\phi$',
        r'E_{\text{miss}} $\theta$',
        r'$E_{\text{miss}}$',
        r'$E_{e}$',
        r'Vertex $\chi^2$',
        r'$n_{\text{Primary Tracks}}$',
        r'$n_{\text{Tracks}}$'
    ]

    background = X_train[:1000].astype(np.float32)
    
    explainer = shap.GradientExplainer(model, background)
    
    shap_values = explainer.shap_values(background)
    
    plt.figure()
    shap.summary_plot(shap_values, background, feature_names=feature_names)
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/shapley_{file}.pdf")
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DNN Training Script')
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

    # Assume the loading code for data and other setups are already here
    tuner = RandomSearch(
        build_model,
        objective=Objective("val_prc", direction="max"),  # Specify the objective explicitly
        max_trials=15,  # Number of different hyperparameter combinations to try
        executions_per_trial=1,  # Number of models that should be constructed for each trial
        directory='model_tuning',  # Directory where the hyperparameters will be stored
        project_name='tuning_results'
    )
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    val_start_index = int(len(y_train) * (1 - 0.3))
    y_val = y_train[val_start_index:]

    print(f'Average class probability in training set:   {y_train.mean():.4f}')
    print(f'Average class probability in validation set: {y_val.mean():.4f}')
    print(f'Average class probability in test set:       {y_test.mean():.4f}')

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=callbacks)

    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Save the best model
    best_model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/gridsearch_best_model_{args.label}.keras')
    
    print("Best model saved successfully.")

    shap_feature_importance(file, best_model, X_train) 
