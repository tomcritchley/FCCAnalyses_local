import uproot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import os
import argparse
import json
import time

tree_name = "events"
base_HNL = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/"
json_file = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"
target_luminosity = 10000  # pb^-1 ---> =10fb^-1

variables = [
    "n_RecoElectrons", "RecoDiJet_delta_R", "RecoDiJet_angle",
    "RecoElectron_DiJet_delta_R", "RecoElectronTrack_absD0sig",
    "RecoElectronTrack_absD0", "RecoDiJet_phi", "RecoMissingEnergy_theta",
    "RecoMissingEnergy_e", "RecoElectron_lead_e", "Vertex_chi2",
    "n_primt", "ntracks"
]

with open(json_file, 'r') as f:
    cross_section_dict = json.load(f)

background_dirs = [
    ("/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/p8_ee_Zcc_ecm91", "5215.46"),
    ("/eos/user/p/pakontax/FCC_8March2024/p8_ee_Zbb_ecm91/", "6654.46"),
    ("/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/ejjnu/", "0.014")
]

# For signal
def load_and_filter_data(filepath, x_sec, tree_name, variables, filter_func):
    with uproot.open(filepath) as file:
        df = file[tree_name].arrays(variables, library="pd")
        df['weight'] = (float(x_sec) * target_luminosity) / len(df)
        return filter_func(df)

# Chunks of background with a retry mechanism
def load_and_preprocess_bkg(filepath, x_sec, filter_func, label):
    dfs = []
    attempt = 0
    max_attempts = 3
    success = False
    
    while attempt < max_attempts and not success:
        try:
            with uproot.open(f"{filepath}:{tree_name}") as tree:
                df = tree.arrays(variables, library="pd")
                df['cross_section'] = float(x_sec)
                df = filter_func(df)
                if x_sec == "5215.46":
                    df['weight'] = (df['cross_section'] * target_luminosity) / 498091935
                elif x_sec == "6654.46":
                    df['weight'] = (df['cross_section'] * target_luminosity) / 438538637
                elif x_sec == "0.014":
                    df['weight'] = (df['cross_section'] * target_luminosity) / 100000
                df['label'] = label
                dfs.append(df)
                success = True
        except Exception as e:
            attempt += 1
            print(f"Error processing file {filepath} on attempt {attempt}: {e}")
            if attempt < max_attempts:
                time.sleep(2)  # wait for 2 seconds before retrying
            else:
                print(f"Failed to process file {filepath} after {max_attempts} attempts.")
    return pd.concat(dfs, ignore_index=True)

def basic_filter(df):
    return df[
        (df["n_RecoElectrons"] == 1) & 
        (df["RecoElectron_lead_e"] > 20) &
        (df["RecoDiJet_angle"] < np.pi) & 
        (df["RecoElectron_DiJet_delta_R"] < 5) &
        (df["RecoDiJet_phi"] < np.pi) & 
        (df["RecoDiJet_delta_R"] < 5)
    ]

def convert_to_numpy(df, fields):
    for field in fields:
        print(f"Converting {field} to a numpy array...")
        df[field] = ak.to_numpy(df[field])

def prepare_datasets():
    parser = argparse.ArgumentParser(description='Prepare datasets for DNN Training')
    parser.add_argument('--label', required=True, help='Signal label, e.g., 10GeV_1e-2')
    args = parser.parse_args()

    signal_file = f"{base_HNL}/HNL_Dirac_ejj_{args.label}Ve.root"
    signal_x_sec = cross_section_dict[f"HNL_Dirac_ejj_{args.label}Ve"]['cross_section_pb']
    signal_df = load_and_filter_data(signal_file, signal_x_sec, tree_name, variables, basic_filter)
    signal_df['label'] = 1

    training_sets = {}
    for i, (background_dir, x_sec) in enumerate(background_dirs):
        print(f"Processing background type {i} with cross section {x_sec} pb")

        background_files = [os.path.join(background_dir, file) for file in os.listdir(background_dir) if file.endswith('00.root') or file.endswith('ejjnu.root')]
        background_df = pd.concat([load_and_preprocess_bkg(filepath, x_sec, basic_filter, 0) for filepath in background_files], ignore_index=True)
        
        signal_train, signal_test = train_test_split(signal_df, test_size=0.5, random_state=42)
        background_train, background_test = train_test_split(background_df, test_size=0.5, random_state=42)

        df_train = pd.concat([signal_train, background_train], ignore_index=True).sample(frac=1).reset_index(drop=True)
        df_test = pd.concat([signal_test, background_test], ignore_index=True).sample(frac=1).reset_index(drop=True)

        training_sets[f"train_{i}"] = df_train

        print(f"Training set for background type {i}:")
        print(f"  Signal events: {len(signal_train)}")
        print(f"  Background events: {len(background_train)}")

        print(f"Test set for background type {i}:")
        print(f"  Signal events: {len(signal_test)}")
        print(f"  Background events: {len(background_test)}")

        if i == 0:
            combined_test = df_test
        else:
            combined_test = pd.concat([combined_test, df_test], ignore_index=True)

    # Convert to numpy arrays
    def convert_to_numpy(df, fields):
        for field in fields:
            df[field] = ak.to_numpy(df[field])

    for df in training_sets.values():
        convert_to_numpy(df, variables)

    convert_to_numpy(combined_test, variables)

    # Standardize and save datasets
    scaler = StandardScaler()

    for key, df_train in training_sets.items():
        X_train, y_train, weights_train = df_train[variables], df_train['label'], df_train['weight']
        X_train_scaled = scaler.fit_transform(X_train)

        np.save(f'/eos/user/t/tcritchl/DNN/training20/X_{key}_{args.label}.npy', X_train_scaled)
        np.save(f'/eos/user/t/tcritchl/DNN/training20/y_{key}_{args.label}.npy', y_train)
        np.save(f'/eos/user/t/tcritchl/DNN/training20/weights_{key}_{args.label}.npy', weights_train)

    X_test, y_test, weights_test = combined_test[variables], combined_test['label'], combined_test['weight']
    X_test_scaled = scaler.transform(X_test)

    np.save(f'/eos/user/t/tcritchl/DNN/testing20/X_test_{args.label}.npy', X_test_scaled)
    np.save(f'/eos/user/t/tcritchl/DNN/testing20/y_test_{args.label}.npy', y_test)
    np.save(f'/eos/user/t/tcritchl/DNN/testing20/weights_test_{args.label}.npy', weights_test)

    print(f"Data preparation complete for label: {args.label}")

if __name__ == "__main__":
    prepare_datasets()
