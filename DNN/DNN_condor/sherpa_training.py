import numpy as np
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
        sherpa.Discrete('num_layers', [2, 3, 4]),  # Number of layers
        sherpa.Continuous('learning_rate', [0.0001, 0.001]),  # Learning rate
        sherpa.Discrete('batch_size', [32, 64, 128])  # Batch size
    ]

    algorithm = sherpa.algorithms.GridSearch()
    study = sherpa.Study(parameters=parameters, algorithm=algorithm, lower_is_better=False)

    for trial in study:
        print(f"Testing parameters: {trial.parameters}")
        model = create_model(input_dim, trial.parameters['num_layers'], trial.parameters['learning_rate'])
        history = model.fit(X_train, y_train, epochs=12, batch_size=trial.parameters['batch_size'], validation_split=0.2, verbose=1)

        # Evaluate model to find the best one
        results = model.evaluate(X_train, y_train, verbose=1)
        metric_index = model.metrics_names.index('prc')  # Assuming 'prc' (Precision-Recall curve AUC) is used as the primary metric
        prc_auc = results[metric_index]
        study.add_observation(trial, objective=prc_auc)

        if study.should_trial_stop(trial):
            study.finalize(trial)

            y_pred = model.predict_classes(X_train)
            cm = confusion_matrix(y_train, y_pred)
            plt.clf()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/conf_training_{file}.pdf")
            plt.close()
            model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.keras')
            break

        study.finalize(trial)

    best_trial = study.get_best_result()
    print(f"Best trial parameters: {best_trial.parameters}")
    print(f"Best trial objective value: {best_trial.objective}")

    print(f"shapley feature importance...")

    shap_feature_importance(file, model, X_train)

if __name__ == "__main__":
    main()



def create_model(input_dim, layers, dropout_rate, learning_rate):
    
    metrics=['accuracy', 'precision', 'recall', AUC(name='prc', curve='PR')]
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
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    # Load data
    file = args.label
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/y_train_{file}.npy', allow_pickle=True)
    X_train = X_train.astype(np.float32)

    input_dim = X_train.shape[1]
    parameters = [
        sherpa.Discrete('num_layers', [2, 3, 4, 5]),  # Number of layers
        sherpa.Discrete('nodes_per_layer', [100, 200, 300]),  # Nodes per layer
        sherpa.Continuous('dropout_rate', [0.1, 0.5]),
        sherpa.Continuous('learning_rate', [0.0001, 0.001]),
        sherpa.Discrete('batch_size', [32, 64, 128, 256])  # Batch size
    ]

    algorithm = sherpa.algorithms.GridSearch()
    study = sherpa.Study(parameters=parameters, algorithm=algorithm, lower_is_better=False)

    best_model = None
    best_balanced_metric = -1

    for trial in study:

        print(f"Testing parameters: {trial.parameters}")

        # Extracting parameters from the trial
        layers = [trial.parameters['nodes_per_layer']] * trial.parameters['num_layers']
        batch_size = trial.parameters['batch_size']
        
        model = create_model(input_dim, layers, trial.parameters['dropout_rate'], trial.parameters['learning_rate'])
        history = model.fit(X_train, y_train, epochs=12, batch_size=batch_size, validation_split=0.2, verbose=1)
        
        # Evaluate the model
        results = model.evaluate(X_train, y_train, verbose=1)
        metrics_names = model.metrics_names
        recall_index = metrics_names.index('recall') if 'recall' in metrics_names else None
        precision_index = metrics_names.index('precision') if 'precision' in metrics_names else None
        
        if recall_index is not None and precision_index is not None:
            recall = results[recall_index]
            precision = results[precision_index]
            balanced_metric = 2 * (precision * recall) / (precision + recall + 0.0001)  # F1 score formula
            print(f"Trial balanced metric: {balanced_metric}")

            ### Plot ROC Curve ###
            y_pred = model.predict(X_train).ravel()
            fpr, tpr, thresholds = roc_curve(y_train, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/ROC_{file}.pdf")
            plt.close()

            # Confusion Matrix
            y_pred = model.predict_classes(X_train)
            cm = confusion_matrix(y_train, y_pred)
            plt.clf()
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/conf_training_{file}.pdf")
            plt.close()

            study.add_observation(trial, objective=balanced_metric)
        else:
            print("Warning: Recall or precision metric not found in the model evaluation results.")
        
        if balanced_metric > best_balanced_metric:
            best_balanced_metric = balanced_metric
            best_model = model
            model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.h5')
            print("Best model updated and saved.")

        if study.should_trial_stop(trial):
            study.finalize(trial)
            break

        study.finalize(trial)
    
    best_trial = study.get_best_result()
    print(f"Best trial parameters: {best_trial.parameters}")
    print(f"Best trial objective value: {best_trial.objective}")

    print(f"shapley feature importance...")

    shap_feature_importance(file, model, X_train)

if __name__ == "__main__":
    main()