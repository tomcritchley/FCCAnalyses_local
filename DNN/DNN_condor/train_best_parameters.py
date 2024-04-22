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

if __name__ == "__main__":

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
    X_train_oversampled, y_train_oversampled = simple_oversample(X_train, y_train, scale_factor=ratio)

    # Model setup for GridSearchCV
    model = KerasClassifier(build_fn=lambda: create_model(input_dim=X_train.shape[1]), verbose=1, epochs=50, batch_size=100)
    param_grid = {
        'layers': [[500, 500, 250, 100, 50], [300, 300, 150]],
        'dropout_rate': [0.2, 0.3],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [50, 100],
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=1)
    grid_result = grid.fit(X_train_oversampled, y_train_oversampled)

    # Results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    for mean, stdev, param in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params']):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    # Optionally save the best model
    grid.best_estimator_.model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/best_model_{file}.h5')
    
    def scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)
        else:
            return float(lr * tf.math.exp(-0.1))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(f'/eos/user/t/tcritchl/DNN/trained_models5/best_model_{file}.keras', save_best_only=True, monitor='val_loss', mode='min'),
        LearningRateScheduler(scheduler)
    ]
    
    history = model.fit(X_train_oversampled, y_train_oversampled, epochs=100, batch_size=256, validation_split=0.2, callbacks=callbacks) #,class_weight=class_weight_dict)
    
    def permutation_feature_importance(model, X, y, metric=accuracy_score):
       
        original_score = metric(y, model.predict(X).round())
        feature_importance = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            permuted_score = metric(y, model.predict(X_permuted).round())
            
            feature_importance[i] = original_score - permuted_score

        return feature_importance
    importances = permutation_feature_importance(model, X_train, y_train)
    print("Feature importances:", importances)

    ### plotting feature importance ###
    plt.figure()
    plt.bar(range(X_train.shape[1]), importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots5/feature_importance_{file}.pdf')
    plt.close()

    for metric in ['loss', 'accuracy', 'precision', 'recall', 'prc']:
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
    model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models5/best_model_{file}.keras')
    print("Model loaded successfully.")
    model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.keras')
    print(f"model saved successfully for {file}")
