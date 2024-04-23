import numpy as np
import sherpa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import argparse

def f1_score(y_true, y_pred):
    """Define the F1 score for model evaluation."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def create_model(input_dim, layers, dropout_rate, learning_rate):
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
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[f1_score])
    return model

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
        sherpa.Discrete('layers', [300, 500]),
        sherpa.Continuous('dropout_rate', [0.1, 0.5]),
        sherpa.Continuous('learning_rate', [0.0001, 0.001])
    ]

    algorithm = sherpa.algorithms.GridSearch()
    study = sherpa.Study(parameters=parameters, algorithm=algorithm, lower_is_better=False)

    best_model = None
    best_f1 = -1

    for trial in study:
        print(f"Testing parameters: {trial.parameters}")
        model = create_model(input_dim, [trial.parameters['layers']] * 5, trial.parameters['dropout_rate'], trial.parameters['learning_rate'])
        history = model.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, verbose=1)
        
        # Evaluate the model
        loss, f1 = model.evaluate(X_train, y_train, verbose=1)
        print(f"Trial F1 score: {f1}")
        study.add_observation(trial, objective=f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            model.save(f'/eos/user/t/tcritchl/DNN/trained_models5/best_model_{file}.h5')
            print("Best model updated and saved.")

        if study.should_trial_stop(trial):
            study.finalize(trial)
            break

        study.finalize(trial)
    
    best_trial = study.get_best_result()
    print(f"Best trial parameters: {best_trial.parameters}")
    print(f"Best trial objective value: {best_trial.objective}")

if __name__ == "__main__":
    main()