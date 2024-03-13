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

with open('/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/new_variables_training/configuration.json') as config_file:
    config = json.load(config_file)

run = config["run_number"]
train_or_test = config["train_or_test"]
input_HNLs = config["preparation_input_path_HNLs"]
input_bkg_cc = config["preparation_input_path_bkg_cc"]
input_bkg_bb = config["preparation_input_path_bkg_bb"]
input_bkg_4body = config["preparation_input_path_bkg_4body"]
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
    df = df.Filter(f"n_RecoElectrons==1 && RecoElectron_lead_e > 35 && RecoDiJet_angle < {np.pi} && RecoElectron_DiJet_delta_R < 5 && RecoDiJet_phi < {np.pi} && RecoDiJet_delta_R < 5", "Exactly one electron final state with lead electron energy E > 35 GeV")  
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
    "RecoElectron_lead_e",
    "Vertex_chi2", #new variable
    "n_primt", #new variable -> we want (ntracks - n_primt)
    "ntracks",
]

raw_background = [
    
    (f"{input_bkg_bb}p8_ee_Zbb_ecm91/", "6654.46"), #dir for the bb chunks
    (f"{input_bkg_cc}p8_ee_Zcc_ecm91/", "5215.46"), #dir for the cc chunks
    (f"{input_bkg_4body}ejjnu/", "0.014") #dir for ejjnu
]

data = []

background = []
for bg_dir, bg_xs in raw_background:
    # Loop over each chunk file in the directory
    for chunk_file in os.listdir(bg_dir):
        if chunk_file.endswith('.root'):
            filepath = os.path.join(bg_dir, chunk_file)
            
            if bg_xs == "6654.46":
                label = f"background_{os.path.splitext(chunk_file)[0]}_bb"
                
                if chunk_file == "chunk_1985.root":
                    print(f"skipping problematic chunk 1985")
                    continue
                elif chunk_file == "chunk_3322.root":
                    print(f"skipping problematic chunk 3322")
                    continue
                else:
                    try:
                    # Try to open the file and read its contents
                        with ROOT.TFile(filepath, "READ") as file:
                            # Check if the file is readable
                            if file.IsZombie() or file.TestBit(ROOT.TFile.kRecovered):
                                print(f"Skipping problematic file {filepath}")
                                continue
                            else:
                                background.append((filepath, label))
                    except Exception as e:
                        print(f"Error processing file {filepath}: {e}")
                        print(f"Skipping problematic file {filepath}")
                        continue        
            elif bg_xs == "5215.46":
                label = f"background_{os.path.splitext(chunk_file)[0]}_cc"
                try:
                # Try to open the file and read its contents
                    with ROOT.TFile(filepath, "READ") as file:
                        # Check if the file is readable
                        if file.IsZombie() or file.TestBit(ROOT.TFile.kRecovered):
                            print(f"Skipping problematic file {filepath}")
                            continue
                        else:
                            background.append((filepath, label))
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
                    print(f"Skipping problematic file {filepath}")
                    continue
            elif bg_xs == "0.014":
                label = f"background_{os.path.splitext(chunk_file)[0]}"
                background.append((filepath, label))

bkg_placeholder = ("bkg_placeholder_name","background_total")
data.append(bkg_placeholder)

#print(f"background list: {background}") #check what this first loop is doing
 
for list in signal_files:
    data.append(list) #add the signal files

#print(f"data is: {data}")

if __name__ == "__main__":
    
    #dealing with background first, then iterate over the signals:
    
    for list in data:
        
        label = list[1]
        filepath = list[0]
        
        print(">>> Extract the training and testing events for {} from the {} dataset.".format(
            label, filepath))
        
        if label.startswith("signal"):

            df = ROOT.RDataFrame("events", filepath) #three seperate df for the signals
        
        elif label.startswith("background"):
            
            print(f"background loop for label {label}")
           
            background_files = []
            
            for bkg_file, bkg_label in background:
                background_files.append(bkg_file)
                ##this wont work because every time it enters a background it will try to create a new dataframe, we only want the dataframe with all the processed backrgounds
                ##placeholder might work
            #print(f"we're in the background files loop, here are the background files -- {background_files}")
            print(f"generating a combined dataframe object ...")
            df = ROOT.RDataFrame("events", background_files)
            print(f"finished generating dataframe!")
        
        #print(f"counting events in df...")
        #generated_events = df.Count().GetValue()

        #print(f"generated events {generated_events}")
        print(f"filtering events..")
        df = filter_events(df) #call the filter
        print(f"defining index....")
        df = df.Define("event_index", "rdfentry_")
        print(f"compiling report...")
        report = df.Report()
	
        columns = ROOT.std.vector["string"](variables)
        
        if label.startswith("background"):
            columns.push_back("cross_section")

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

    df_pd = pd.DataFrame(x) ##cant take the cov matrix ones so need to exclude it somehow

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
