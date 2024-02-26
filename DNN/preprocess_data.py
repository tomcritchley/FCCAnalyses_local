import uproot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

#name of the tree --> (case sensitive!)
tree_name = "events"

#base_path where the signals are located
base_HNL = "/eos/user/t/tcritchl/HNLs_Feb24/"

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

background_filenames = ["/eos/user/t/tcritchl/xgBOOST/fullstats/p8_ee_Zbb_ecm91/p8_ee_Zbb_ecm91/p8_ee_Zbb_ecm91.root", "/eos/user/t/tcritchl/xgBOOST/fullstats/p8_ee_Zcc_ecm91/p8_ee_Zcc_ecm91/p8_ee_Zcc_ecm91.root", "eos/user/t/tcritchl/xgBOOST/fullstats/ejjnu/ejjnu.root"]

dfs = []

#signal w score of 1
for filename in signal_filenames:
    with uproot.open(f"{filename}:{tree_name}") as tree:
        df_signal = tree.arrays(library="pd")
        df_signal['label'] = 1
        dfs.append(df_signal)

#background w score of 0
for filename in background_filenames:
    with uproot.open(filename) as tree:
        df_background = tree.arrays(library="pd")
        df_background['label'] = 0
        dfs.append(df_background)

df = pd.concat(dfs, ignore_index=True)

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

df = df[variables + ['label']]

# Optionally, preprocess your dataframe (e.g., normalization, feature engineering)
# For example, split your data into features and labels
X = df.iloc[:, :-1]  # assuming the last column is the label
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#normalising/scaling to what?
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

