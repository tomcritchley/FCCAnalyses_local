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

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=512, step=32),
                    input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                        activation='relu'))
        model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=keras.optimizers.Adam(
                  hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'precision', 'recall', AUC(name='prc', curve='PR')])
    
    return model

def plot_metrics(history,label):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['prc'], label='Precision-Recall Curve')
    plt.plot(history.history['val_prc'], label='Validation PR')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots7/prec_recall_{label}.pdf")

def permutation_feature_importance(model, X, y, feature_names,label):
    from sklearn.metrics import accuracy_score
    original_score = accuracy_score(y, model.predict(X).round())
    feature_importance = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        X_permuted = X.copy()
        np.random.shuffle(X_permuted[:, i])
        
        permuted_score = accuracy_score(y, model.predict(X_permuted).round())
        feature_importance[i] = original_score - permuted_score

    plt.figure(figsize=(10, 5))
    plt.bar(feature_names, feature_importance, color='lightblue')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha="right")
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots7/feature_importance_{label}.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNN Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    file = args.label
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training7/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training7/y_train_{file}.npy', allow_pickle=True)
    X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing7/X_test_{file}.npy', allow_pickle=True)
    y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing7/y_test_{file}.npy', allow_pickle=True)

    feature_names = [
        r'$\Delta R_{jj}$', r'$\Delta R_{ejj}$', r'$D_0$', r'Dijet $\phi$', r'E_{\text{miss}} $\theta$',
        r'$E_{\text{miss}}$', r'$E_{e}$', r'Vertex $\chi^2$', r'$n_{\text{Primary Tracks}}$', r'$n_{\text{Tracks}}$'
    ]
    
    tuner = RandomSearch(build_model, objective=Objective("val_prc", direction="max"), max_trials=15,
                         executions_per_trial=1, directory='model_tuning', project_name='tuning_results')
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]


    val_start_index = int(len(y_train) * (1 - 0.3))
    y_val = y_train[val_start_index:]

    print(f'Average class probability in training set:   {y_train.mean():.4f}')
    print(f'Average class probability in validation set: {y_val.mean():.4f}')
    print(f'Average class probability in test set:       {y_test.mean():.4f}')

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=callbacks)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(f'/eos/user/t/tcritchl/DNN/trained_models7/DNN_HNLs_{args.label}.keras')
    print("Best model saved successfully.")

    history = best_model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    label = file
    plot_metrics(history,label)
    permutation_feature_importance(best_model, X_test, y_test, feature_names,label)