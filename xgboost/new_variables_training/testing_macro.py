import ROOT
from training_macro import load_data
from DataPreparation_macro import masses, couplings
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import roc_curve, auc
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns

##################################################################################################################
###################################### INPUTS ####################################################################
##################################################################################################################

with open('/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/new_variables_training/configuration.json') as config_file:
    config = json.load(config_file)

run = config["run_number"]
#bkg_norm = config["bkg_normalisation_factor"]
#sgl_norm = config["signal_normalisation_factor"]

labels = []
base_path = f"/eos/user/t/tcritchl/xgBOOST/testing{run}/"

for mass in masses:
    for coupling in couplings:
        print(f"getting label for mass: {mass}, coupling {coupling}")
        base_file = f"test_signal_{mass}_{coupling}.root"
        signal_file = os.path.join(base_path, base_file)
        if os.path.exists(signal_file):
            labels.append(f"signal_{mass}_{coupling}") #label will be of the form "signal_10GeV_1e-2"
        else:
            print(f"file {base_file} does not exist, moving to next file")

print(labels) #list of labels for the data prepared thing..

"""
RL signficance is used to count the significance from left to right and keep everything to the left of the cut,
LR significance is used to count from right most to left most bin, and values to the left of the cut are kept
"""

significance_directions = ["RL", "LR"]
bdt_thr = 0.9


##################################################################################################################
###################################### BDT PREDICTIONS ###########################################################
##################################################################################################################

if __name__ == "__main__":

    results_dict = {}

    parser = argparse.ArgumentParser(description='BDT Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    label = args.label

    significance_direction = significance_directions[1]

    x, y_true, w, w_training = load_data(f"/eos/user/t/tcritchl/xgBOOST/testing{run}/test_{label}.root", f"/eos/user/t/tcritchl/xgBOOST/testing{run}/test_background_total.root")
    print(f"weights {w}")

    # Load trained model
    File = f"/eos/user/t/tcritchl/xgBOOST/trained_models{run}/tmva_{label}.root"
    print(f"BDT file is: {File}")
    if ROOT.gSystem.AccessPathName(File):
        ROOT.Info("testing_macro.py", File + "does not exist")
        exit()

    bdt = ROOT.TMVA.Experimental.RBDT[""]("myBDT", File)

    # Convert to object which can parse to compute
    v = ROOT.std.vector('float')()
    for row in x:
        for element in row:
            v.push_back(element)

    # For some reason was only making one prediction from v, so iterate
    y_pred_list = []
    for row in x:
        v = ROOT.std.vector('float')()
        for element in row:
            v.push_back(element)
        pred = bdt.Compute(v)
        y_pred_list.append(pred[0])        

    # Convert list to NumPy array
    y_pred_np = np.array(y_pred_list)

    # Reshape the NumPy array
    y_pred_np = y_pred_np.reshape(-1)

    maskS = (y_true == 1) & (y_pred_np > bdt_thr)
    maskB = (y_true == 0) & (y_pred_np > bdt_thr)

    print(y_true.shape)  # Check the shape of y_true
    print(y_pred_np.shape)  # Check the shape of y_pred
    print(x.shape)
    
    B = y_pred_np[y_true == 0] #the predictions for the background
    S = y_pred_np[y_true == 1] #the predictions for the signal

    resultsSIG = y_pred_np[maskS] #the true signal in the region beyond the threshold
    resultsBKG = y_pred_np[maskB] #the false positives (bkg) in the region beyond the threshold

    w_signal = w[y_true == 1] #take the weights where the data is signal
    w_background = w[y_true == 0] #take the weights where the data is background

    B_weighted = np.sum(B * w_background) #now do np.sum rather than np.sum becuase it is no longer an integer
    S_weighted = np.sum(S * w_signal) 

    # Number of signal and background events that satisfy the condition
    print(f"Number of signal events that satisfy the threshold of {bdt_thr}: {np.size(resultsSIG)}") #number of correctly predicted signal at threshold
    print(f"Number of background events that satisfy the threshold of {bdt_thr}: {np.size(resultsBKG)}") #number of correctly predicted background at threshold

    print(f"weighted number of signal event in total: {S_weighted}")
    print(f"weighted number of background events in total: {B_weighted}")

    # weight the signals with the normalisation factors x 2 since we have half of the data set

    weightsSIG = np.ones_like(S) * w_signal * 2 #sgl_norm #used half of the background
    weightsBKG = np.ones_like(B) * w_background * 3/2 #bkg_norm # using 2/3 for the test here so since w = x_sec x lumi / N_gen, then you multiply by [N_samp / fraction used = N_gen]...  

    print(f"weights of the signal first 10: {weightsSIG[:10]}")
    print(f"weights of the background first 15: {weightsBKG[:15]}")

    ##################################################################################################################
    ###################################### ROC CURVE #################################################################
    ##################################################################################################################
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_np, sample_weight=w)
    area = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.plot(fpr, tpr, label=f'ROC curve (area = {area:.2f})')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/ROC_xgboost_{label}_5.pdf")

    ##################################################################################################################
    ###################################### BDT OUTPUT PLOT RAW #######################################################
    ##################################################################################################################

    B = y_pred_np[y_true == 0] #the predictions for the background
    S = y_pred_np[y_true == 1] #the predictions for the signal
    # Plot histogram of predicted scores for signal and background events
    plt.figure()
    plt.hist(S, bins=50, alpha=0.5, color='b', label='Signal')
    plt.hist(B, bins=50, alpha=0.5, color='r', label='Background')
    plt.xlabel('Predicted Score')
    plt.ylabel('Log MC events')
    plt.title('Predicted Scores for Signal and Background Events')
    plt.yscale('log')
    plt.legend(loc='upper center')
    plt.grid(True)
    plt.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/raw_bdt_classification_{label}.pdf")
    plt.close()

    ##################################################################################################################
    ###################################### CONFUSION MATRIX ##########################################################
    ##################################################################################################################

    # Assuming y_pred_np is the predicted labels and y_true is the actual labels
    cm = confusion_matrix(y_true, (y_pred_np > 0.5).astype(int))
    plt.clf()
    plt.figure(figsize=(8, 6))  # Assuming threshold of 0.5 for binary classification
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Background', 'Signal'], yticklabels=['Background', 'Signal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/raw_confusion_matrix_{label}.pdf")
    plt.close()

    ##################################################################################################################
    ###################################### BDT OUTPUT PLOTS ##########################################################
    ##################################################################################################################
    

    #### logic for calculating the ideal bin range ###

    ## need to calculate the bdt score for each signal event between 0 and 1.0 and say break it into 1000 so between 0.001 and 1.000 ##

    ## then find the peak and do +/- 0.10 for the peak number of events in the bin and set them equal to min and max bin, with 2000 bins.
    
    bin_width = 0.0001 #in the region of interest, binning resolution
    
    full_range_bins = np.linspace(np.min(y_pred_np), np.max(y_pred_np), 1000) #scan over full range with 1000 bins
    
    print(f"scanning over all bins for mass point {label}")

    # Calculate histogram for signal events over the full range
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
        print(f"mass point {label} has 0 background events for a minimum bin of {min_bin}")
        # Decrement min_bin
        min_bin = min_bin - 0.5
        print(f"new minimum bin {min_bin}")
        # Update bins and compute background histogram
        bins = np.arange(min_bin, max_bin, bin_width)
        bkg_hist, _ = np.histogram(B, bins=bins, weights=weightsBKG)
        # Check if the background histogram is empty
        if np.argmax(bkg_hist) == 0:
            print(f"No background events found for mass point {label}. Exiting loop.")
            min_bin = 0
            max_bin = 1
            bin_width = 0.001
            break

    fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [5, 2], 'hspace': 0.05})

    for axis in ax:
        axis.tick_params(axis='both', direction='in', which='both', top=True, right=True)

    ##optimise this to scan over the full range of significances?
    
    #bins_a = np.arange(0.8000, 1.0001, bin_width) ##change to 0.8000 -> 1.0001 for most recent plot

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
    ax[0].set_title(f"{label} vs Total Background")
    ax[0].legend(loc='upper right', fontsize='small', frameon=False)

    fig.text(0.175, 0.85, "FCCee Simulation (DELPHES)", ha='left', va='center', fontsize=8, weight='bold')
    fig.text(0.175, 0.81, "Exactly one recontructed electron, E > 20 GeV", ha='left', va='center', fontsize=8)
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
            print(f"significance {significance} for bin {bin_idx} with BDT threshold {left_edge}, number of signal events {s}, bkg{b}")

            sig_list.append((significance, bin_idx, left_edge))

        return sig_list

    # Plot cumulative significance on the second subplot
    sig_list = make_cumulative_significance_matplotlib(hS, hB, significance_direction, uncertainty_count_factor=0.1)
    sig_list.sort(key=lambda x: x[1])
    significance_values, bin_index, bdt_output = zip(*sig_list)

    results_dict[label] = {
        "weighted_background_events": B_weighted,
        "weighted_signal_events": S_weighted,
        "significance_list": sig_list
    }

    ax[1].step(bins_a[:-1], significance_values, where='post', color='green', linewidth=1.5)
    ax[1].set_xlabel('BDT response')
    ax[1].set_ylabel(f'Z Significance ({significance_direction})')
    ax[1].grid(True)

    max_significance_index = np.argmax(significance_values)
    max_significance_bin = bin_index[max_significance_index]
    max_significance_value = significance_values[max_significance_index]
    ax[1].axvline(x=bins_a[int(max_significance_bin)], linestyle='--', color='red', label=f'Max Significance: {max_significance_value:.2f}')
    ax[1].legend()


    plt.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/BDT_output_{label}_10fb.pdf")

    ##################################################################################################################
    ###################################### SAVING MODEL OUTPUTS ######################################################
    ##################################################################################################################

    print(f"starting file saving loop")
        
    signal = [f"/eos/user/t/tcritchl/xgBOOST/testing{run}/test_{label}.root", "signal"] #second entry was preivously just {label} but the name was acting in a strange way when looking at the bdt output
    background = [f"/eos/user/t/tcritchl/xgBOOST/testing{run}/test_background_total.root", "background_total"]
    try:
        for filepath, label in [signal, background]:
            print(">>> Extract the training and testing events for {} from the {} dataset.".format(label, filepath))

            if os.path.exists(filepath):
                print("Opened " + filepath)

                file = ROOT.TFile(filepath, "UPDATE")
                tree = file.Get("events")

                new_tree = tree.CloneTree(0)
                new_tree.SetName(f"events_modified_{label}")
                
                if label.startswith("signal"):
                    bdt_output_branch = np.zeros_like(S, dtype=np.float32)
                    new_tree.Branch(f"bdt_output_{label}", bdt_output_branch, 'bdt_output/F')
                elif label.startswith("background"):
                    bdt_output_branch = np.zeros_like(B, dtype=np.float32)
                    new_tree.Branch(f"bdt_output_{label}", bdt_output_branch, 'bdt_output/F')

                print(f"the shape of S is {S.shape}, and it contains the elements {S}")
                print(f"the shape of B is {B.shape}, and it contains the elements {B}")
                print(f"tree has entries: {tree.GetEntries()}")
                for idx in range(tree.GetEntries()):
                    tree.GetEntry(idx)
                    if label.startswith("signal"):
                        bdt_output_branch[0]  = S[idx] # changes for not filling all the 
                        print(f"bdt branch: {idx}, with value {S[idx]}")
                    elif label.startswith("background"):
                        bdt_output_branch[0]  = B[idx]
                    new_tree.Fill()
                print(f"writing file")
                file.Write()
                file.Close()
                print(f"Successfully done")

            else:
                print(f"File {filepath} does not exist, skipping.")
    except Exception as e:
        print(f"An error occurred: {e}")

    json_file_path = f"/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/new_variables_training/LLP_study/xgboost_run{run}_{args.label}.json"

    try:
        with open(json_file_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=2)
        print(f"Results saved to {json_file_path} successfully!")
    except Exception as e:
        print(f"An error occurred while saving results: {e}")