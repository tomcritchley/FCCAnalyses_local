import ROOT
from training_macro import load_data
from DataPreparation_macro import masses, couplings
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
import json

##################################################################################################################
###################################### INPUTS ####################################################################
##################################################################################################################

with open('/afs/cern.ch/user/t/tcritchl/xgboost/configuration.json') as config_file:
    config = json.load(config_file)

run = config["run_number"]
bkg_norm = config["bkg_normalisation_factor"]
sgl_norm = config["signal_normalisation_factor"]


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

results_dict = {}

for label in labels:

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
   
    weightsSIG = np.ones_like(S) * w_signal * sgl_norm #used half of the background
    weightsBKG = np.ones_like(B) * w_background * bkg_norm # using 2/3 for the test here so since w = x_sec x lumi / N_gen, then you multiply by [N_samp / fraction used = N_gen]...  

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
    plt.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/ROC_xgboost_{label}.pdf")

    ##################################################################################################################
    ###################################### BDT OUTPUT PLOTS ##########################################################
    ##################################################################################################################

    fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [5, 2], 'hspace': 0.05})

    for axis in ax:
        axis.tick_params(axis='both', direction='in', which='both', top=True, right=True)

    bin_width = 0.0001 #change to 0.0001 for the same results as for the most recent plot
    ##optimise this to scan over the full range of significances?
    bins_a = np.arange(0.8000, 1.0001, bin_width) ##change to 0.8000 -> 1.0001 for most recent plot

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
    fig.text(0.175, 0.81, "Exactly one recontructed electron, E > 30 GeV", ha='left', va='center', fontsize=8)
    fig.text(0.175, 0.77, r"$\sqrt{s} = 91$ GeV, $\int L \, dt = 150 \, \text{ab}^{-1}$", ha='left', va='center', fontsize=8)

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
            print(f"significance {significance} for bin {bin_idx}, number of signal events {s}, bkg{b}")
            sig_list.append((significance, bin_idx))

        return sig_list

    # Plot cumulative significance on the second subplot
    sig_list = make_cumulative_significance_matplotlib(hS, hB, significance_direction, uncertainty_count_factor=0.1)
    sig_list.sort(key=lambda x: x[1])
    significance_values, bin_indices = zip(*sig_list)

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
    max_significance_bin = bin_indices[max_significance_index]
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

            print(f"the shape of S is {S.shape}")
            print(f"the shape of B is {B.shape}")
            print(f"tree has entries: {tree.GetEntries()}")
            for idx in range(tree.GetEntries()):
                tree.GetEntry(idx)
                if label.startswith("signal"):
                    bdt_output_branch[idx] = S[idx]
                elif label.startswith("background"):
                    bdt_output_branch[idx] = B[idx]
                new_tree.Fill()
            print(f"writing file")
            file.Write()
            file.Close()
            print(f"Successfully done")

        else:
            print(f"File {filepath} does not exist, skipping.")

json_file_path = f"/afs/cern.ch/user/t/tcritchl/xgboost_batch/test_xgboost_results{run}_10fb.json"

with open(json_file_path, "w") as json_file:
    json.dump(results_dict, json_file, indent=2)

print(f"Results saved to {json_file_path}")