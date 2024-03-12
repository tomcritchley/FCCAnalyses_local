##CHANGE TRAIN TO TEST, DIRECTORIES AND LUMI###

import ROOT
import numpy as np
import os
import json
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import GridSearchCV
from  matplotlib import  pyplot
from DataPreparation_macro import masses, couplings, variables
import ROOT
import argparse

with open('/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/xgboost_batch/configuration.json') as config_file:
    config = json.load(config_file)

run = config["run_number"]
train_or_test = config["train_or_test"].strip()
print(f"are we training or testing: {train_or_test}")
labels = []
base_path = f"/eos/user/t/tcritchl/xgBOOST/training{run}/"
json_file = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"


with open(json_file, 'r') as f:
    json_data = json.load(f)

cross_section_dict = {}

for key, value in json_data.items():
    parts = key.split("_")
    mass = parts[-2]
    coupling = parts[-1]

    root_file_name = f"{train_or_test}_signal_{mass}_{coupling.replace('Ve', '')}.root"

    cross_section_dict[root_file_name] = value["cross_section_pb"]

print(cross_section_dict)

#have to edit if it is training or testing e.g. test_signal vs train_signal
cross_section_dict = {f"{train_or_test}_signal_{mass}_{coupling.replace('Ve', '')}.root": cross_section for coupling, mass, cross_section, _ in json_data}
#print(cross_section_dict)

for mass in masses:
    for coupling in couplings:
        print(f"getting label for mass: {mass}, coupling {coupling}")
        base_file = f"train_signal_{mass}_{coupling}.root"
        signal_file = os.path.join(base_path, base_file)
        if os.path.exists(signal_file):
            labels.append(f"signal_{mass}_{coupling}") #label will be of the form "signal_10GeV_1e-2"
        else:
            print(f"file {signal_file} does not exist, moving to next file")

print(labels) #list of labels for the data prepared thing..

# in order to start TMVA                                                                                                          
ROOT.TMVA.Tools.Instance()

def load_data(signal_filename, background_filename):
    
    # Read data from ROOT files
    data_sig = ROOT.RDataFrame("events", signal_filename).AsNumpy()
    data_bkg = ROOT.RDataFrame("events", background_filename).AsNumpy()

    # Convert inputs to format readable by machine learning tools
    x_sig = np.vstack([data_sig[var] for var in variables]).T
    x_bkg = np.vstack([data_bkg[var] for var in variables]).T

    x = np.vstack([x_sig, x_bkg])

    # Convert RVec<float> to shape (1,) numpy arrays; probably could do just scalars but the arrays seemed to work
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if isinstance(x[i, j], ROOT.VecOps.RVec('float')):
                x[i, j] = np.array([element for element in x[i, j]])
        
    # Create labels
    num_sig = x_sig.shape[0]
    num_bkg = x_bkg.shape[0]
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
 
    cross_sections = data_bkg["cross_section"]

    # Compute weights based on cross-sections
    weights = []
    num_bb = 0
    num_cc = 0
    num_4body = 0
    signal = os.path.basename(signal_filename)
    print(signal)
    
    cross_section_signal = cross_section_dict.get(signal, 1.0)
    
    if cross_section_signal == 1.0:
        print(f"something has gone wrong with the cross section retrieval")
        exit()

    print(f"cross section for {signal} is {cross_section_signal}")
    print(f"number of background = {num_bkg}")

    for cross_section in cross_sections:
        if cross_section == 6654.46:
            num_bb += 1
            #weights.append(6654.46*10000/438738637)
            weights.append(6654.46*10000/438738637)
        elif cross_section == 5215.46:
            num_cc += 1
            #weights.append(5215.46*10000/499786495)
            weights.append(6654.46*10000/438738637)
        elif cross_section == 0.014:
            num_4body += 1
            #weights.append(0.014*10000/100000)
            weights.append(0.014*10000/100000)
            
    print(f"number of bb: {num_bb}; number of cc: {num_cc}; number of 4body: {num_4body}")
    print(f"total background {num_bb+num_cc+num_4body}")
    print(f"weights background = {weights[:10]}")

    if signal == "test_signal_40Gev_1e-5.root":
        print(f"using the signal with less events {signal} :(")
        w = np.hstack([np.ones(num_sig)*(cross_section_signal*10000/7196), #scale factor = target lumi * x-sec / number of generated events
                   np.ones(num_bkg)*weights])
    elif signal == "test_signal_20Gev_1e-5.root":
        print(f"using the signal with less events {signal} :(")
        w = np.hstack([np.ones(num_sig)*(cross_section_signal*10000/45948), #scale factor = target lumi * x-sec / number of generated events
                   np.ones(num_bkg)*weights])
    elif signal == "test_signal_20GeV_1e-4p5.root":
        print(f"using the signal with less events {signal} :(")
        w = np.hstack([np.ones(num_sig)*(cross_section_signal*10000/9707), #scale factor = target lumi * x-sec / number of generated events
                   np.ones(num_bkg)*weights])
    elif signal == "test_signal_10GeV_1e-3p5.root":
        print(f"using the signal with less events {signal} :(")
        w = np.hstack([np.ones(num_sig)*(cross_section_signal*10000/9707), #scale factor = target lumi * x-sec / number of generated events
                   np.ones(num_bkg)*weights])
    else:
        w = np.hstack([np.ones(num_sig)*(cross_section_signal*10000/100000), #scale factor = target lumi * x-sec / number of generated events
                   np.ones(num_bkg)*weights])

    #Compute weights balancing both classes so you have the same number for each
    num_all = num_sig + num_bkg
    w_training = np.hstack([np.ones(num_sig) * num_all / num_sig, np.ones(num_bkg) * num_all / num_bkg])
    
    return x, y, w, w_training

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='BDT Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    label = args.label
    
    if not os.path.exists(f"/eos/user/t/tcritchl/xgBOOST/trained_models{run}/tmva_{label}.root"):
        print(f"Loading data: train_{label}.root")
        sample = "_20gev"
        x, y, w, w_training = load_data(f"/eos/user/t/tcritchl/xgBOOST/training{run}/train_{label}.root", f"/eos/user/t/tcritchl/xgBOOST/training{run}/train_background_total.root")
        print("x type:", type(x), "x shape:", x.shape)
        print("y type:", type(y), "y shape:", y.shape)
        print("w type:", type(w), "w shape:", w.shape)
        print("Sample elements in x:", x[:5])
        print("Sample elements in y:", y[:5])
        print("Sample elements in w:", w[:5])
        print("Done Loading")

        param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 500],
    }

        grid_search = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, scoring='accuracy', cv=3)
        grid_search.fit(x, y, sample_weight=w_training)
        bdt = XGBClassifier(max_depth=3, n_estimators=500)

        best_params = grid_search.best_params_
        bdt = XGBClassifier(**best_params)
        bdt.fit(x, y, sample_weight=w_training)
        sorted_idx = np.argsort(bdt.feature_importances_)[::-1]
        plot_importance(bdt, max_num_features = 15)
        pyplot.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/feature_importance_plot_{label}.pdf")
        pyplot.show()
        # Save model in TMVA format
        print("Training done on ",x.shape[0],f"events. Saving model in tmva_{label}.root")
        ROOT.TMVA.Experimental.SaveXGBoost(bdt, "myBDT", f"/eos/user/t/tcritchl/xgBOOST/trained_models{run}/tmva_{label}.root", num_inputs=x.shape[1])
        
        plot_tree(bdt, num_trees=5, rankdir='LR')  # Adjust num_trees as needed
        pyplot.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/decision_tree_plot_{label}.pdf")
    else:
        print(f"The trained model for {label} already exists! No need to re-run :)")
