import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, make_scorer, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import AUC
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.over_sampling import SMOTE
import argparse

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

def create_model():
    model = Sequential([
        Dense(128, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(128, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),
        Dense(64, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'), 
        Dropout(0.3),
        Dense(32, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1Score()])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDT Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/X_train_{args.label}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training5/y_train_{args.label}.npy', allow_pickle=True)

    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=256, verbose=1)
    param_grid = {'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 10, 10)]}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer(f1_score), n_jobs=1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    print("Best parameters found: ", grid_result.best_params_)
    print("Best F1 score found: ", grid_result.best_score_)

