import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tqdm import tqdm
import os

base_HNL = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/"

masses = [
    "10GeV",
    "20GeV",
    "30GeV",
    "40GeV",
    "50GeV",
    "60GeV",
    "70GeV",
    "80GeV",   
]

couplings = [
    "1e-2", 
    "1e-2p5", 
    "1e-3", 
    "1e-3p5", 
    "1e-4", 
    "1e-4p5", 
    "1e-5"
]

signal_filenames = []

for mass in masses:
    for coupling in couplings:
        base_file = f"HNL_Dirac_ejj_{mass}_{coupling}Ve.root"
        signal_file = os.path.join(base_HNL, base_file)
        if os.path.exists(signal_file):
            signal_filenames.append(signal_file)
        else:
            print(f"file {signal_file} does not exist, moving to next file")

print(signal_filenames)

for filename in signal_filenames:
    
    file = filename.split('/')
    final_part = file[-1]
    info_parts = final_part.split('_')
    file = '_'.join(info_parts[4:])
    
    print(f"loading data...")
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training1/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training1/y_train_{file}.npy', allow_pickle=True)
    X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing1/X_test_{file}.npy', allow_pickle=True)
    y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing1/y_test_{file}.npy', allow_pickle=True)
    print(f"data loaded for {file}!")
    print(f"loading model....")
    model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models1/DNN_HNLs_{file}.keras')
    print(f"model loaded for {file}!")

    ### testing the model ###

    # Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc*100:.2f}%')

    ### ROC curve ###
    y_pred = model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots1/ROC_{file}.pdf")

    # Get predicted scores for signal and background events
    y_pred_signal = model.predict(X_test[y_test == 1]).ravel()
    y_pred_background = model.predict(X_test[y_test == 0]).ravel()

    # Plot histogram of predicted scores for signal and background events
    plt.figure()
    plt.hist(y_pred_signal, bins=50, alpha=0.5, color='b', label='Signal')
    plt.hist(y_pred_background, bins=50, alpha=0.5, color='r', label='Background')
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    plt.title('Predicted Scores for Signal and Background Events')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots1/dnn_classification_{file}.pdf")
