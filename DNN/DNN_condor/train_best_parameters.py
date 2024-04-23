import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
import argparse
import shap
import pandas as pd

def simple_oversample(X_train, y_train, scale_factor):
    minority_indices = np.where(y_train == 1)[0]
    repeated_minority_indices = np.repeat(minority_indices, int(scale_factor))
    np.random.shuffle(repeated_minority_indices)
    X_oversampled = np.vstack([X_train, X_train[repeated_minority_indices]])
    y_oversampled = np.hstack([y_train, y_train[repeated_minority_indices]])
    return X_oversampled, y_oversampled

def create_model(input_dim, layers=[500, 500, 250, 100, 50], dropout_rate=[0.1, 0.5], learning_rate=0.001):
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

def plot_grid_search(file, cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots5/best_parameters_{file}.pdf")
    plt.close()

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

    # Data preprocessing
    X_train = X_train.astype(np.float32)
    #bkg, sig = np.bincount(y_train.astype(int))
    #ratio = bkg / sig
    #X_train_oversampled, y_train_oversampled = simple_oversample(X_train, y_train, ratio)

    # Model setup for GridSearchCV
    input_dim = X_train.shape[1]
    print(f"making keras classifier...")
    model = KerasClassifier(build_fn=lambda: create_model(input_dim=input_dim), verbose=1)
    param_grid = {
        'layers': [[500, 500, 250, 100, 50], [300, 300, 150]],
        'dropout_rate': [0.1, 0.5],
        'learning_rate': [0.001, 0.0001],
    }
    print(f"performing grid search...")
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=1)
    grid_result = grid.fit(X_train, y_train)
    print(f"grid search complete...")
    best_params = grid.best_params_
    print("Best parameters found:", best_params)

    # Results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for mean, stdev, param in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params']):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    dropout_rate_range = [0.1, 0.5]
    learning_rate_range = [0.001, 0.0001]

    plot_grid_search(grid_result.cv_results_, dropout_rate_range, learning_rate_range, 'Dropout rate', 'Learning Rate')

    # Final training with best parameters
    print(f"final model being trained...")
    final_model = create_model(input_dim=input_dim, **grid_result.best_params_)
    final_model.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.2)

    # Save the final trained model
    final_model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.h5')
    print("Model training completed and saved.")

    print(f"shapley feature importance...")
    # Call this function with your trained final model and the oversampled training set
    shap_feature_importance(file, final_model, X_train)

if __name__ == "__main__":
    main()
