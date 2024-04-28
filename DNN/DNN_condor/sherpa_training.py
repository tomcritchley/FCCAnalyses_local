"""import numpy as np
import sherpa
import shap
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import argparse

def create_model(input_dim, num_layers, learning_rate):
    model = Sequential()
    metrics=['accuracy', 'precision', 'recall', AUC(name='prc', curve='PR')]
    # Initialize layers based on the number of layers
    for i in range(num_layers):
        # Add Dense layers
        if i == 0:
            model.add(Dense(50, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))  # Using a fixed number of nodes
        else:
            model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))  # Using a fixed number of nodes
        model.add(Dropout(0.2))  # Using a fixed dropout rate
        model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model

def shap_feature_importance(file,model, X_train):
    explainer = shap.DeepExplainer(model, X_train[:1000])  # Using 100 samples as the background dataset for approximation
    shap_values = explainer.shap_values(X_train[:1000])
    # Using matplotlib to save the figure
    plt.figure()
    shap.summary_plot(shap_values, X_train[:1000], feature_names=["Feature_" + str(i) for i in range(X_train.shape[1])])
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/shapley_{file}.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='DNN Training Script')
    parser.add_argument('--label', required=True, help='Label for the data', metavar='label')
    args = parser.parse_args()

    # Load data
    file = args.label
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/X_train_{file}.npy')
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/y_train_{file}.npy')
    input_dim = X_train.shape[1]

    parameters = [
        sherpa.Discrete('num_layers', range=[2, 3, 4]),  # Number of layers
        sherpa.Continuous('learning_rate', range=[0.0001, 0.001]),  # Learning rate
        sherpa.Discrete('batch_size', range=[32, 64, 128])  # Batch size
    ]

    algorithm = sherpa.algorithms.GridSearch()
    study = sherpa.Study(parameters=parameters, algorithm=algorithm, lower_is_better=False)

    best_prc_auc = 0  # Initialize the best performance metric
    best_model = None  # Placeholder for the best model

    for trial in study:
        print(f"Testing parameters: {trial.parameters}")
        model = create_model(input_dim, trial.parameters['num_layers'], trial.parameters['learning_rate'])
        history = model.fit(X_train, y_train, epochs=12, batch_size=trial.parameters['batch_size'], validation_split=0.2, verbose=1)
        results = model.evaluate(X_train, y_train, verbose=1)
        prc_auc = results[model.metrics_names.index('prc')]  # Get PRC AUC score

        study.add_observation(trial, objective=prc_auc)
        if prc_auc > best_prc_auc:  # Check if this model is the best one
            best_prc_auc = prc_auc
            best_model = model
            model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.keras')  # Save the best model
            print("Best model updated and saved.")

        if study.should_trial_stop(trial):
            y_pred = (model.predict(X_train) > 0.5).astype("int32")
            cm = confusion_matrix(y_train, y_pred)
            plt.clf()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/conf_training_{file}.pdf")
            plt.close()
            study.finalize(trial)

    study.finalize_study()  # Finalize the study after all trials

    best_trial = study.get_best_result()
    print(f"Best trial parameters: {best_trial.parameters}")
    print(f"Best trial objective value: {best_trial.objective}")

    shap_feature_importance(file, best_model, X_train)  # Calculate SHAP values for the best model

if __name__ == "__main__":
    main()


import numpy as np
import sherpa
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix
import argparse

# Define the model creation function
def create_model(input_dim, num_layers, learning_rate):
    model = Sequential()
    metrics=['accuracy', 'precision', 'recall', AUC(name='prc', curve='PR')]
    for i in range(num_layers):
        model.add(Dense(50, input_dim=input_dim if i == 0 else None, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=metrics)
    return model

# Define the SHAP feature importance function
def shap_feature_importance(file, model, X_train):
    explainer = shap.DeepExplainer(model, X_train[:1000])
    shap_values = explainer.shap_values(X_train[:1000])
    plt.figure()
    shap.summary_plot(shap_values, X_train[:1000], feature_names=["Feature_" + str(i) for i in range(X_train.shape[1])])
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/shapley_{file}.pdf")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='DNN Training Script')
    parser.add_argument('--label', required=True, help='Label for the data', metavar='label')
    args = parser.parse_args()

    # Load data
    file = args.label
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/X_train_{file}.npy')
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/y_train_{file}.npy')
    input_dim = X_train.shape[1]

    # Define hyperparameter space using the sherpa.Parameter.grid() method
    hp_space = {'num_layers': [2, 3, 4],
                'learning_rate': [0.0001, 0.001],
                'batch_size': [32, 64, 128]}
    parameters = sherpa.Parameter.grid(hp_space)
    
    # Create the GridSearch algorithm instance
    alg = sherpa.algorithms.GridSearch()

    # Initialize the Sherpa study
    study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=False)

    best_prc_auc = 0
    best_model = None

    for trial in study:
        print(f"Testing parameters: {trial.parameters}")
        model = create_model(input_dim, trial.parameters['num_layers'], trial.parameters['learning_rate'])
        history = model.fit(X_train, y_train, epochs=12, batch_size=trial.parameters['batch_size'], validation_split=0.2, verbose=1)
        results = model.evaluate(X_train, y_train, verbose=1)
        prc_auc = results[model.metrics_names.index('prc')]

        study.add_observation(trial, objective=prc_auc)
        if prc_auc > best_prc_auc:
            best_prc_auc = prc_auc
            best_model = model
            model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.keras')
            print("Best model updated and saved.")

        study.finalize_trial(trial)

    study.finalize_study()

    best_trial = study.get_best_result()
    print(f"Best trial parameters: {best_trial.parameters}")
    print(f"Best trial objective value: {best_trial.objective}")

    shap_feature_importance(file, best_model, X_train)

if __name__ == "__main__":
    main()

"""

import numpy as np
import sherpa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix
import argparse

def create_model(input_dim, num_layers, learning_rate):
    model = Sequential()
    metrics=['accuracy', 'precision', 'recall', AUC(name='prc', curve='PR')]
    for i in range(num_layers):
        model.add(Dense(50, input_dim=input_dim if i == 0 else None, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=metrics)
    return model

def main():
    parser = argparse.ArgumentParser(description='DNN Training Script')
    parser.add_argument('--label', required=True, help='Label for the data', metavar='label')
    args = parser.parse_args()

    file = args.label
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/X_train_{file}.npy')
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/y_train_{file}.npy')
    input_dim = X_train.shape[1]

    parameters = [
        sherpa.Continuous('learning_rate', [0.0001, 0.001]),
        sherpa.Discrete('num_layers', [2, 3, 4]),
        sherpa.Discrete('batch_size', [32, 64, 128])
    ]

    algorithm = sherpa.algorithms.GridSearch()
    study = sherpa.Study(parameters=parameters, algorithm=algorithm, lower_is_better=False)

    for trial in study:
        model = create_model(input_dim, trial.parameters['num_layers'], trial.parameters['learning_rate'])
        model.fit(X_train, y_train, epochs=12, batch_size=trial.parameters['batch_size'], validation_split=0.2, verbose=1)
        prc_auc = model.evaluate(X_train, y_train, verbose=0)[model.metrics_names.index('prc')]

        study.add_observation(trial, objective=prc_auc)
        if prc_auc > study.best_result:
            model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.h5')
        study.finalize(trial)

    print(f"Best trial parameters: {study.best_trial.parameters}")
    print(f"Best trial objective value: {study.best_result}")

if __name__ == "__main__":
    main()
