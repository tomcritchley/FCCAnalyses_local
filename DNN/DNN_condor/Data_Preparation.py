import uproot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import awkward as ak
import os
import argparse
import json

tree_name = "events"
base_HNL = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/"
json_file = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"
target_luminosity = 10000

variables = [
    "n_RecoElectrons", "RecoDiJet_delta_R", "RecoDiJet_angle",
    "RecoElectron_DiJet_delta_R", "RecoElectronTrack_absD0sig",
    "RecoElectronTrack_absD0", "RecoDiJet_phi", "RecoMissingEnergy_theta",
    "RecoMissingEnergy_e", "RecoElectron_lead_e"
]

masses = ["10GeV", "20GeV", "30GeV", "40GeV", "50GeV", "60GeV", "70GeV", "80GeV"]
couplings = ["1e-2", "1e-2p5", "1e-3", "1e-3p5", "1e-4", "1e-4p5", "1e-5"]

with open(json_file, 'r') as f:
    cross_section_dict = {f"{key.split('_')[-2]}_{key.split('_')[-1].replace('Ve', '')}": value["cross_section_pb"]
                          for key, value in json.load(f).items()}

background_dirs = [
    ("/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/p8_ee_Zcc_ecm91", "5215.46"),
    ("/eos/user/p/pakontax/FCC_8March2024/p8_ee_Zbb_ecm91/", "6654.46"),
    ("/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/ejjnu/", "0.014")
]

def load_and_preprocess_data(filepaths, x_sec, filter_func, label):
    dfs = []
    for filepath in filepaths:
        try:
            with uproot.open(f"{filepath}:{tree_name}") as tree:
                df = tree.arrays(variables, library="pd")
                df['cross_section'] = float(x_sec)
                df = filter_func(df)
                df['weight'] = (df['cross_section'] * target_luminosity) / len(df)
                df['label'] = label
                dfs.append(df)
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue
    return pd.concat(dfs, ignore_index=True)

def load_and_preprocess_bkg(filepaths_and_xsecs, filter_func, label):
    dfs = []
    for filepath, x_sec in filepaths_and_xsecs:
        try:
            with uproot.open(f"{filepath}:{tree_name}") as tree:
                df = tree.arrays(variables, library="pd")
                df['cross_section'] = float(x_sec)
                df = filter_func(df)
                if x_sec == "5215.46":
                    df['weight'] = (df['cross_section'] * target_luminosity) / 499786495
                elif x_sec == "6654.46":
                    df['weight'] = (df['cross_section'] * target_luminosity) / 438538637
                elif x_sec == "0.014":
                    df['weight'] = (df['cross_section'] * target_luminosity) / 100000
                df['label'] = label
                dfs.append(df)
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue
    return pd.concat(dfs, ignore_index=True)


def signal_filter(df):
    return df[
        (df["n_RecoElectrons"] == 1) & 
        (df["RecoElectron_lead_e"] > 15) &
        (df["RecoDiJet_angle"] < np.pi) & 
        (df["RecoElectron_DiJet_delta_R"] < 5) &
        (df["RecoDiJet_phi"] < np.pi) & 
        (df["RecoDiJet_delta_R"] < 5)
    ]

def main(label):
    signal_file = f"{base_HNL}/HNL_Dirac_ejj_{label}Ve.root"
    signal_df = load_and_preprocess_data([signal_file], cross_section_dict.get(label, 1.0), signal_filter, 1)
    
    background_files = [(os.path.join(dir, file), x_sec) for dir, x_sec in background_dirs for file in os.listdir(dir) if file.endswith('.root')]
    background_df = load_and_preprocess_bkg(background_files, signal_filter, 0)  # Use 1.0 as a placeholder for x_sec
    
    df = pd.concat([signal_df, background_df], ignore_index=True)
    
    print(f"converting missing D0 to a numpy array...")
    D0 = ak.to_numpy(df['RecoElectronTrack_absD0'])
    df['RecoElectronTrack_absD0'] = D0

    print(f"converting missing D0 sig to a numpy array...")
    D0sig = ak.to_numpy(df['RecoElectronTrack_absD0sig'])
    df['RecoElectronTrack_absD0sig'] = D0sig

    print(f"converting missing energy theta to a numpy array...")
    miss_e_theta = ak.to_numpy(df['RecoMissingEnergy_theta'])
    df['RecoMissingEnergy_theta'] = miss_e_theta

    print(f"converting missing energy to a numpy array...")
    miss_e = ak.to_numpy(df['RecoMissingEnergy_e'])
    df['RecoMissingEnergy_e'] = miss_e

    X = df[variables]
    y = df['label']
    weights = df['weight']

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

    # Scale the flattened data
    print("Scaling the data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaling completed.")

    print("Saving the preprocessed data...")
    np.save(f'/eos/user/t/tcritchl/DNN/training2/X_train_{label}.npy', X_train_scaled)
    np.save(f'/eos/user/t/tcritchl/DNN/testing2/X_test_{label}.npy', X_test_scaled)
    np.save(f'/eos/user/t/tcritchl/DNN/training2/y_train_{label}.npy', y_train)
    np.save(f'/eos/user/t/tcritchl/DNN/testing2/y_test_{label}.npy', y_test)
    np.save(f'/eos/user/t/tcritchl/DNN/testing2/weights_test_{label}.npy', weights_test)
    print(f"Preprocessed data saved successfully for {label}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDT Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()
    main(args.label)
