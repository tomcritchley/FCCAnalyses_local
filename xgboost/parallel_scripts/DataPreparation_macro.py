## changes to process 5000 chunks ... ##

import ROOT
import pandas as pd
import seaborn as sns
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.stats import zscore

# Set display options to show all columns
pd.set_option('display.max_columns', None)

with open('/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/xgboost_batch/configuration.json') as config_file:
    config = json.load(config_file)

run = config["run_number"]
train_or_test = config["train_or_test"]
input_HNLs = config["preparation_input_path_HNLs"]
input_bkg = config["preparation_input_path_bkg"]
bkg_split = config["signal_split"]
sgl_split = config["background_split"]

##################################################################################################################
###################################### RUN OVER ALL MASS POINTS ##################################################
##################################################################################################################

#base_path where the signals are located
base_path = input_HNLs

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

#initialise list to store the files

signal_files = []

for mass in masses:
    for coupling in couplings:
        print(f"using mass: {mass}, coupling {coupling}")
        base_file = f"HNL_Dirac_ejj_{mass}_{coupling}Ve.root"
        signal_file = os.path.join(base_path, base_file)
        if os.path.exists(signal_file):
            signal_files.append([signal_file, f"signal_{mass}_{coupling}"]) #label will be of the form "signal_10GeV_1e-2"
        else:
            print(f"file {signal_file} does not exist, moving to next file")

print(signal_files)

def filter_events(df):

    #df = df.Filter(f"n_RecoElectrons==1 && RecoElectron_lead_e > 35 && RecoDiJet_angle < {np.pi} && RecoElectron_DiJet_delta_R < 5 && RecoDiJet_phi < {np.pi} && RecoDiJet_delta_R < 5 && ROOT::VecOps::All(RecoElectronTrack_absD0sig < 5)", "Exactly one electron final state with lead electron energy E > 35 GeV and D0_sig < 5 (prompt decay)") FOR LLPS
    df = df.Filter(f"n_RecoElectrons==1 && RecoElectron_lead_e > 20 && RecoDiJet_angle < {np.pi} && RecoElectron_DiJet_delta_R < 5 && RecoDiJet_phi < {np.pi} && RecoDiJet_delta_R < 5", "Exactly one electron final state with lead electron energy E > 20 GeV")
    
    return df


#bdt features

variables = [
    "RecoDiJet_delta_R",
    "RecoDiJet_angle", 
    "RecoElectron_DiJet_delta_R",
    "RecoElectronTrack_absD0sig", 
    "RecoElectronTrack_absD0",
    "RecoDiJet_phi",
    "RecoMissingEnergy_theta",
    "RecoMissingEnergy_e",
    "RecoElectron_lead_e"
]

#logic for one combined soup of background --> this needs to change

raw_background = [(f"{input_bkg}p8_ee_Zbb_ecm91/p8_ee_Zbb_ecm91/p8_ee_Zbb_ecm91.root", "6654.46"),(f"{input_bkg}p8_ee_Zcc_ecm91/p8_ee_Zcc_ecm91/p8_ee_Zcc_ecm91.root", "5215.46"), (f"{input_bkg}ejjnu/ejjnu.root", "0.014")]

bkg = ["placeholder", "background_total"]

data = []

data.append(bkg)

for list in signal_files:
    data.append(list)

print(f"data is: {data}")

if __name__ == "__main__":
    
    #dealing with background first, then iterate over the signals:

    for filepath, label in data:
        print(">>> Extract the training and testing events for {} from the {} dataset.".format(
            label, filepath))
        
        if label.startswith("signal"):

            df = ROOT.RDataFrame("events", filepath)
        
        elif label.startswith("background"):
           
            df = ROOT.RDataFrame("events", {raw_background[0][0],raw_background[1][0],raw_background[2][0]})
        
        generated_events = df.Count().GetValue()

        print(f"generated events {generated_events}")
        df = filter_events(df) #call the filter
        
        df = df.Define("event_index", "rdfentry_")

        # Book cutflow report
        report = df.Report()

        columns = ROOT.std.vector["string"](variables)
        
        if label.startswith("background"):
            columns.push_back("cross_section")
        if label.startswith("background"):
        #filter 10/90 split
            df.Filter(f"event_index % {bkg_split}  == 0", "Select events with even event number for training").Snapshot("events", f"/eos/user/t/tcritchl/xgBOOST/training{run}/train_{label}.root", columns)
            df.Filter(f"event_index % {bkg_split} != 0", "Select events with odd event number for testing").Snapshot("events", f"/eos/user/t/tcritchl/xgBOOST/testing{run}/test_{label}.root", columns)
        
        elif label.startswith("signal"):
            
            df.Filter(f"event_index % {sgl_split} == 0", "Select events with even event number for training").Snapshot("events", f"/eos/user/t/tcritchl/xgBOOST/training{run}/train_{label}.root", columns)
            df.Filter(f"event_index % {sgl_split} != 0", "Select events with odd event number for testing").Snapshot("events", f"/eos/user/t/tcritchl/xgBOOST/testing{run}/test_{label}.root", columns)

        report.Print()

##################################################################################################################
###################################### PLOTTING FOR VARIABLE RELATIONSHIPS #######################################
##################################################################################################################
    
    for column_name in df.GetColumnNames():
        column_type = df.GetColumnType(column_name)
        print(f"{column_name}: {column_type}")

    x = df.AsNumpy()

    for key, array in x.items():
        for i, obj in enumerate(array):
            if isinstance(obj, ROOT.VecOps.RVec('float')):
                array[i] = float(obj[0])

    df_pd = pd.DataFrame(x)

    print(f"printing the df_pd contents {df_pd.describe()}")

    df_selected = df_pd[variables].copy()

    print(df_selected.dtypes)
    print(df_selected.isnull().sum())

    nan_check = df_selected.isna().any()

    # Print columns with NaN values
    columns_with_nan = nan_check[nan_check].index
    print(f"Columns with NaN values: {columns_with_nan}")

    # Remove rows with NaN values
    df_selected = df_selected.dropna()

    # Verify that NaN values are removed
    print("NaN values after removal:")
    print(df_selected.isna().sum())

    for column in df_selected:
        if df_selected[column].dtype != 'float32':
            print(f"converting column {column} from {df_selected[column].dtype} to float32")
            df_selected[column] = pd.to_numeric(df_selected[column], errors='raise').astype('float32')

    df_selected = df_selected[(np.abs(zscore(df_selected)) < 3).all(axis=1)]
    
    print(df_selected.describe())
    correlation_matrix = df_selected.corr()
    plt.figure(figsize=(12, 10))
    #can't add annotations to every square because of bug with matplotlib
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix Heatmap")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/correlation_heatmap.pdf")

    scatter_matrix(df_selected, alpha=0.8, figsize=(15, 15), diagonal='kde')
    plt.suptitle("Pair Plot", y=1.02)
    plt.yticks(rotation=90)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"/eos/user/t/tcritchl/xgboost_plots{run}/pair_plot.pdf")
