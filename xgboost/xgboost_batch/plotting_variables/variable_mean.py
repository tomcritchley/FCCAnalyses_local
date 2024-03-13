import ROOT
import pandas as pd
import seaborn as sns
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy.stats import zscore

pd.set_option('display.max_columns', None)

signal_means = {}

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

base_path = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/"

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

for file, label in signal_files:
    if os.path.exists(file):
        df = ROOT.RDataFrame("events", file) #generate the rdf object
        
        generated_events = df.Count().GetValue() #count num of events
        print(f"generated events {generated_events}")

        columns = ROOT.std.vector["string"](variables) #define the variables as the df columns
    else:
        print(f"signal file {file} does not exist, moving on")
        continue
    column_names = df.GetColumnNames()
    print(f"the column names are: {column_names}")
    filtered_columns = [column for column in column_names if column in variables]
    print(f"the filtered column names are: {filtered_columns}")

    for column_name in filtered_columns:
        try:
            column_type = df.GetColumnType(column_name)
            print(f"{column_name}: {column_type}")
        except Exception as e:
            print(f"something is wrong with the column: {column_name}, error was {e}")
            continue

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

    columns_with_nan = nan_check[nan_check].index
    print(f"Columns with NaN values: {columns_with_nan}")

    df_selected = df_selected.dropna()

    print("NaN values after removal:")
    print(df_selected.isna().sum())

    for column in df_selected:
        if df_selected[column].dtype != 'float32':
            print(f"converting column {column} from {df_selected[column].dtype} to float32")
            df_selected[column] = pd.to_numeric(df_selected[column], errors='raise').astype('float32')

    df_selected = df_selected[(np.abs(zscore(df_selected)) < 3).all(axis=1)]
    mean_values = df_selected.mean().round(3)
    print(f"description of the means of the df for label {label}")
    print(f":{mean_values}")
    print(f"description of the pandas df for label {label}:")
    print(f"{df_selected.describe()}")
    print(f"saving to json for the label...")
    signal_means[label] = mean_values.to_dict()

json_file_path = "/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/xgboost_batch/plotting_variables/signal_means.json"
try:
    with open(json_file_path, "w") as json_file:
        json.dump(signal_means, json_file, indent=4)
    print("Mean values saved to JSON file:", json_file_path)
except Exception as e:
    print("Error saving mean values to JSON file:", e)
