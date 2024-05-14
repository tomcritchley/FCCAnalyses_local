import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from tensorflow.keras.metrics import Precision, Recall, AUC
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
import os
import argparse
import json
import math

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

significance_directions = ["RL", "LR"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BDT Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    file = args.label

    def plot_confusion_matrix(y_true, y_pred, threshold=0.5): #threshold for confusion!

        y_pred_labels = (y_pred > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Background', 'Signal'], yticklabels=['Background', 'Signal'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots14/CM_{file}.pdf")

    significance_direction = significance_directions[1]

    results_dict = {}

    print(f"loading data...")
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training14/X_train_{file}.npy', allow_pickle=True)
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training14/y_train_{file}.npy', allow_pickle=True)
    X_test = np.load(f'/eos/user/t/tcritchl/DNN/testing14/X_test_{file}.npy', allow_pickle=True)
    y_test = np.load(f'/eos/user/t/tcritchl/DNN/testing14/y_test_{file}.npy', allow_pickle=True)
    weights_test = np.load(f'/eos/user/t/tcritchl/DNN/testing14/weights_test_{file}.npy', allow_pickle=True)
    print(f"data loaded for {file}!")
    print(f"loading model....")
    #model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models5/DNN_HNLs_{file}.keras')
    model = tf.keras.models.load_model(f'/eos/user/t/tcritchl/DNN/trained_models14/DNN_HNLs_{file}.keras')
    print(f"model loaded for {file}!")

    print(model.metrics_names)

    ### testing the model ###

    # Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    
    #test_loss, test_acc, test_prc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=2)
    test_loss, test_accuracy, test_auc, test_prc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=2)

    y_pred = model.predict(X_test).ravel()

    #confusion matrix to see ROC in detail
    plot_confusion_matrix(y_test, y_pred, threshold=0.5)
    
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
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots14/ROC_{file}.pdf")

    ### Precision ROC ###

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 0.2])  # Adjust x-limits to zoom into the left part of the curve
    plt.ylim([0.8, 1.05])  # Adjust y-limits to focus on the higher TPRs
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Zoomed)')
    plt.legend(loc="lower right")

    # Plot Precision-Recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots14/Precise_ROC_{file}.pdf")
    plt.close()

    ##################################################################################################################
    ###################################### DNN PREDICTIONS ###########################################################
    ##################################################################################################################


    # Get predicted scores for signal and background events
    y_pred_signal = model.predict(X_test[y_test == 1]).ravel()
    y_pred_background = model.predict(X_test[y_test == 0]).ravel()

    # Plot histogram of predicted scores for signal and background events
    plt.figure()
    plt.hist(y_pred_signal, bins=50, alpha=0.5, color='b', label='Signal')
    plt.hist(y_pred_background, bins=50, alpha=0.5, color='r', label='Background')
    plt.xlabel('Predicted Score')
    plt.ylabel('MC events')
    plt.title('Raw Predicted Scores for Signal and Background Events')
    plt.yscale('log')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots14/raw_dnn_classification_{file}.pdf")

    background_weights = weights_test[y_test == 0]
    unique_weights = np.unique(background_weights)

    # Plot histogram of predicted scores for signal and each background type
    plt.clf()
    plt.figure(figsize=(10, 6))
    # Plot for signal
    n_signal, bins, _ = plt.hist(y_pred_signal, bins=50, alpha=0.7, color='blue', label='Signal (n={})'.format(len(y_pred_signal)), histtype='step', linewidth=2)

    # Colors for each background type
    colors = ['red', 'green', 'purple']
    labels = ['Background Type 1', 'Background Type 2', 'Background Type 3']

    # Iterate over each unique weight (background type)
    for weight, color, label in zip(unique_weights, colors, labels):
        mask = background_weights == weight
        y_pred_background_type = y_pred_background[mask]
        n_background, _, _ = plt.hist(y_pred_background_type, bins=bins, alpha=0.5, color=color, label='{} (n={})'.format(label, len(y_pred_background_type)), histtype='step', linewidth=2)

    plt.xlabel('Predicted Score')
    plt.ylabel('Number of Events')
    plt.title('Raw Predicted Scores for Signal and Background Events')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots14/raw_bkg_separated_{file}.pdf")
   
    weightsSIG = weights_test[y_test == 1] 
    weightsBKG = weights_test[y_test == 0]
    # Plot histogram of predicted scores for signal and background events WEIGHTED 10fb^-1 ##
    plt.figure()
    plt.clf()
    plt.hist(y_pred_signal, bins=50, alpha=0.5, color='blue', label='Signal', weights=weightsSIG)
    plt.hist(y_pred_background, bins=50, alpha=0.5, color='red', label='Background', weights=weightsBKG)
    plt.xlabel('Predicted Score')
    plt.ylabel('Weighted MC events')
    plt.title('Weighted Predicted Scores for Signal and Background Events at 10 fb')
    plt.yscale('log')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots14/scaled_dnn10fb_{file}.pdf")
    plt.close()

    # Plot histogram of predicted scores for signal and background events WEIGHTED 150 ab^-1 ##
    plt.figure()
    plt.clf()
    plt.hist(y_pred_signal, bins=50, alpha=0.5, color='blue', label='Signal', weights=(weightsSIG*15000))
    plt.hist(y_pred_background, bins=50, alpha=0.5, color='red', label='Background', weights=(weightsBKG*15000))
    plt.xlabel('Predicted Score')
    plt.ylabel('Weighted MC events')
    plt.title('Weighted Predicted Scores for Signal and Background Events at 150 ab')
    plt.yscale('log')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots14/scaled_dnn_150ab_{file}.pdf")
    plt.close()

    bin_width = 0.0001 #in the region of interest, binning resolution
    
    full_range_bins = np.linspace(np.min(y_pred), np.max(y_pred), 2000) #scan over full range with 1000 bins
    
    print(f"scanning over all bins for mass point {file}")

    # Calculate histogram for signal events over the full range

    S = y_pred_signal
    B = y_pred_background

    target_luminosity = 10000

    signal_hist, _ = np.histogram(S, bins=full_range_bins, weights=weightsSIG)
    peak_bin = np.argmax(signal_hist)
    peak_value = full_range_bins[peak_bin]

    range_width = 0.1  # Width of the range around the peak
    if peak_value + range_width > 1.0:
        #If the peak is close to the maximum possible output of 1.0, set max_bin to 1.0
        max_bin = 1.0 + bin_width #e.g 1.0001
        min_bin = peak_value - range_width #e.g 0.9000
    else:
        #Otherwise, set the range around the peak bin as usual
        min_bin = peak_value - range_width
        max_bin = peak_value + range_width + bin_width
    
    #have to ensure there at least something in the background
    bins = np.arange(min_bin, max_bin, bin_width)
    bkg_hist, _ = np.histogram(B, bins=bins, weights=weightsBKG)
    
    while np.argmax(bkg_hist) == 0:
        print(f"mass point {file} has 0 background events for a minimum bin of {min_bin}")
        # Decrement min_bin
        min_bin = min_bin - 0.5
        print(f"new minimum bin {min_bin}")
        # Update bins and compute background histogram
        bins = np.arange(min_bin, max_bin, bin_width)
        bkg_hist, _ = np.histogram(B, bins=bins, weights=weightsBKG)
        # Check if the background histogram is empty
        if np.argmax(bkg_hist) == 0:
            print(f"No background events found for mass point {file}. Exiting loop.")
            min_bin = 0
            max_bin = 1
            bin_width = 0.001
            break

    fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [5, 2], 'hspace': 0.05})

    for axis in ax:
        axis.tick_params(axis='both', direction='in', which='both', top=True, right=True)

    bins_a = np.arange(min_bin, max_bin, bin_width)

    hB, bins = np.histogram(B, bins=bins_a, weights=weightsBKG)
    hS, bins = np.histogram(S, bins=bins_a, weights=weightsSIG)

    ax[0].hist(B, bins_a, weights=weightsBKG, alpha=0.5, label="Background", color="lightcoral", edgecolor='darkred', hatch='///', histtype='stepfilled', linewidth=2.0, density=False)
    ax[0].hist(S, bins_a, weights=weightsSIG, alpha=0.5, label="Signal", color="skyblue", edgecolor='darkblue', hatch='', linewidth=2.0, histtype='stepfilled', density=False)
    density = False
    # Add labels and a title
    ax[0].set_yscale('log')
    if density:
        ax[0].set_ylabel('(1/N)dN/dx')
    else:
        ax[0].set_ylabel('Log Normalised Events')
    ax[0].set_title(f"{file} vs Total Background")
    ax[0].legend(loc='upper right', fontsize='small', frameon=False)

    fig.text(0.175, 0.85, "FCCee Simulation (DELPHES)", ha='left', va='center', fontsize=8, weight='bold')
    fig.text(0.175, 0.81, "Exactly one recontructed electron, E > 15 GeV", ha='left', va='center', fontsize=8)
    fig.text(0.175, 0.77, r"$\sqrt{s} = 91$ GeV, $\int L \, dt = 10 \, \text{fb}^{-1}$", ha='left', va='center', fontsize=8)

    def make_cumulative_significance_matplotlib(signal_hist, background_hist, significance_direction, uncertainty_count_factor=0.1):
        
        sig_list = []
        s_cumulative = 0
        b_cumulative = 0
        sigma_cumulative = b_cumulative * uncertainty_count_factor

        n_bins = len(signal_hist)
        
        if significance_direction == "RL":
            bin_range = range(1, n_bins + 1)
        elif significance_direction == "LR":
            bin_range = range(n_bins, 0, -1)
        else:
            raise ValueError("Invalid significance_direction. Use 'LR' for left to right or 'RL' for right to left.")

        bin_edges = np.linspace(min_bin, max_bin, n_bins + 1)

        for bin_idx in bin_range:
            s = signal_hist[bin_idx - 1]
            s_cumulative += s
            b = background_hist[bin_idx - 1]
            b_cumulative += b
            sigma_cumulative = b_cumulative * uncertainty_count_factor
            significance = 0

            if b_cumulative + sigma_cumulative > 0 and b_cumulative >= 0 and sigma_cumulative != 0:
                n = s_cumulative + b_cumulative
                significance = math.sqrt(abs(
                    2 * (n * math.log((n * (b_cumulative + sigma_cumulative**2)) / (b_cumulative**2 + n * sigma_cumulative**2)) - (b_cumulative**2 / sigma_cumulative**2) * math.log((1 + (sigma_cumulative**2 * (n - b_cumulative)) / (b_cumulative * (b_cumulative + sigma_cumulative**2))))
                )))
            left_edge = bin_edges[bin_idx - 1]
            print(f"significance {significance} for bin {bin_idx} with DNN threshold {left_edge}, number of signal events {s}, bkg{b}")

            sig_list.append((significance, bin_idx, left_edge))

        return sig_list

    # Plot cumulative significance on the second subplot
    sig_list = make_cumulative_significance_matplotlib(hS, hB, significance_direction, uncertainty_count_factor=0.1)
    sig_list.sort(key=lambda x: x[1])
    significance_values, bin_index, bdt_output = zip(*sig_list)

    results_dict[file] = {
        "significance_list": sig_list
    }

    ax[1].step(bins_a[:-1], significance_values, where='post', color='green', linewidth=1.5)
    ax[1].set_xlabel('DNN response')
    ax[1].set_ylabel(f'Z Significance ({significance_direction})')
    ax[1].grid(True)

    max_significance_index = np.argmax(significance_values)
    max_significance_bin = bin_index[max_significance_index]
    max_significance_value = significance_values[max_significance_index]
    ax[1].axvline(x=bins_a[int(max_significance_bin)], linestyle='--', color='red', label=f'Max Significance: {max_significance_value:.2f}')
    ax[1].legend()


    plt.savefig(f"/eos/user/t/tcritchl/DNN/DNN_plots14/DNN_output_{file}_10fb.pdf")

    ##################################################################################################################
    ###################################### SAVING MODEL OUTPUTS ######################################################
    ##################################################################################################################

json_file_path = f"/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/DNN_Run14_10fb_{file}.json"

print(f"attempting to save results to {json_file_path}....!")
try:
    # Saving results to the unique file
    with open(json_file_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=2)
    print(f"Results saved to {json_file_path} successfully!")
except Exception as e:
    print(f'something went wrong saving the file: {e}')
