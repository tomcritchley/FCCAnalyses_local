import uproot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import awkward as ak
import os
import argparse
import json

#name of the tree --> (case sensitive!)
tree_name = "events"
#base_path where the signals are located
base_HNL = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/"

json_file = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json" #cross section dictionary for the HNLs

target_luminosity = 10000  # Just an example value, adjust according to your analysis

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

with open(json_file, 'r') as f:
    json_data = json.load(f)

cross_section_dict = {}

for key, value in json_data.items():
    parts = key.split("_")
    mass = parts[-2]
    coupling = parts[-1]

    file_name = f"{mass}_{coupling.replace('Ve', '')}"

    cross_section_dict[file_name] = value["cross_section_pb"]

print(cross_section_dict)

cc_basedir = "/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/p8_ee_Zcc_ecm91/"
bb_basedir = "/eos/user/p/pakontax/FCC_8March2024/p8_ee_Zbb_ecm91/"
ejjnu_basedir = "/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/"

background_dirs = [(cc_basedir,"5215.46"),(bb_basedir,"6654.46"),(ejjnu_basedir,"0.014")]

background_filenames = []

for dir, x_sec in background_dirs:
    for chunk_file in os.listdir(dir):
        if chunk_file.endswith('.root'):
            filepath = os.path.join(dir, chunk_file)
            
            if x_sec == "6654.46":
                label = f"background_{os.path.splitext(chunk_file)[0]}_bb"
                
                if chunk_file == "chunk_1985.root" or chunk_file == "chunk_3322.root":
                    print(f"Skipping problematic chunk {chunk_file}")
                    continue
                else:
                    try:
                        with uproot.open(filepath) as file:
                            if not file:
                                print(f"Skipping problematic file {filepath}")
                                continue
                            else:
                                background_filenames.append((filepath, x_sec))
                    except Exception as e:
                        print(f"Error processing file {filepath}: {e}")
                        print(f"Skipping problematic file {filepath}")
                        continue        
            elif x_sec == "5215.46":
                label = f"background_{os.path.splitext(chunk_file)[0]}_cc"
                try:
                    with uproot.open(filepath) as file:
                        if not file:
                            print(f"Skipping problematic file {filepath}")
                            continue
                        else:
                            background_filenames.append((filepath, x_sec))
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
                    print(f"Skipping problematic file {filepath}")
                    continue
            elif x_sec == "0.014":
                label = f"background_{os.path.splitext(chunk_file)[0]}"
                background_filenames.append((filepath, x_sec))

dfs_signal = []
dfs_background = []

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BDT Training Script')
    parser.add_argument('--label', help='Label for the data', metavar='label')
    args = parser.parse_args()

    file = args.label

    signal_filenames = (f"{base_HNL}/HNL_Dirac_ejj_{file}Ve.root",)
    x_sec = cross_section_dict.get(file, 1.0)
    # signal w score of 1
    for filename, x_sec in signal_filenames:
        print(f"attempting to open {filename} for the tree {tree_name}....")
        with uproot.open(f"{filename}:{tree_name}") as tree:
            print(f"file is open...")
            # Select only the variables of interest
            df_signal = tree.arrays(variables, library="pd")
            print(f"cross section is {x_sec}, adding to the df")
            df_signal['cross-section'] = x_sec
            print(f"cross section added to the dataframe")

            print(f"adding weights to the df...")
            if filename.endswith("40Gev_1e-5.root"):
                df_signal['weight'] = (df_signal['cross_section'] * target_luminosity) / 7196
            if filename.endswith("20Gev_1e-4.root"):
                df_signal['weight'] = (df_signal['cross_section'] * target_luminosity) / 76400
            if filename.endswith("20Gev_1e-5.root"):
                df_signal['weight'] = (df_signal['cross_section'] * target_luminosity) / 45948
            if filename.endswith("20Gev_1e-4p5.root"):
                df_signal['weight'] = (df_signal['cross_section'] * target_luminosity) / 9707
            if filename.endswith("10Gev_1e-3p5.root"):
                df_signal['weight'] = (df_signal['cross_section'] * target_luminosity) / 38667
            else:
                df_signal['weight'] = (df_signal['cross_section'] * target_luminosity) / 100000
            
            print(f"labelling signal file with 1...")
            df_signal['label'] = 1
            print(f"successfully labelled signal, adding to dfs.")
            print("First few rows of df_signal:")
            print(df_signal.head())  # Print the first few rows
            print("Number of events in the signal dataframe:", len(df_signal))
            dfs_signal.append((df_signal,filename))

    # background w score of 0
    for filename, x_sec in background_filenames:
        print(f"attempting to open {filename} for the tree {tree_name}....")
        try:
            with uproot.open(f"{filename}:{tree_name}") as tree:
                print(f"file is open...")
                # Select only the variables of interest
                df_background = tree.arrays(variables, library="pd")
                print(f"cross section is {x_sec}, adding to the df")
                df_background['cross-section'] = x_sec
                print(f"cross section added to the dataframe")
                if x_sec == "5215.46":
                    df_background['weight'] = (df_background['cross_section'] * target_luminosity) / 499786495 # needs to change for missing chunks!!
                elif x_sec == "6654.46":
                    df_background['weight'] = (df_background['cross_section'] * target_luminosity) / 438538637
                elif x_sec == "0.014":
                    df_background['weight'] = (df_background['cross_section'] * target_luminosity) / 100000
                print(f"labelling signal file with 0...")
                df_background['label'] = 0
                print(f"successfully labelled background, adding to dfs.")
                print("First few rows of df_background:")
                print(df_background.head())  # Print the first few rows
                print("Number of events in the background dataframe:", len(df_background))
                dfs_background.append(df_background)
        except Exception as e:
                print(f"For some reason, {filename} failed with message {e}")
                continue

    for signal_df, filename in dfs_signal:
        file_parts = filename.split('/')
        final_part = file_parts[-1]
        info_parts = final_part.split('_')

        file = '_'.join(info_parts[3:5]).replace('Ve.root', '')
        print(f"concatenating df for signal {file}")
        dfs = []
        dfs.append(signal_df)
        for df_background in dfs_background:
            dfs.append(df_background)
        df = pd.concat(dfs, ignore_index=True)

        print("Number of events in the combined dataframe:", len(df))
            
        print(f"unprocessed df concenated... printing header!")

        print(df.head())  # Print the first few rows
        print(f"filtering events....")
        df = df[df['RecoElectron_lead_e'] > 35] #attempt to filter
        print(df.head())
        print("Shape of the DataFrame:", df.shape)

        print(f"converting missing D0 to a numpy array...")
        D0 = ak.to_numpy(df['RecoElectronTrack_absD0'])
        df['RecoElectronTrack_absD0'] = D0

        print(f"converting missing D0 sig to a numpy array...")
        D0 = ak.to_numpy(df['RecoElectronTrack_absD0sig'])
        df['RecoElectronTrack_absD0sig'] = D0

        print(f"converting missing energy theta to a numpy array...")
        miss_e_theta = ak.to_numpy(df['RecoMissingEnergy_theta'])
        df['RecoMissingEnergy_theta'] = miss_e_theta

        print(f"converting missing energy to a numpy array...")
        miss_e = ak.to_numpy(df['RecoMissingEnergy_e'])
        df['RecoMissingEnergy_e'] = miss_e

        # Optionally, preprocess your dataframe (e.g., normalization, feature engineering)
        # For example, split your data into features and labels
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the flattened data
        print("Scaling the data...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Scaling completed.")

        print("Saving the preprocessed data...")
        np.save(f'/eos/user/t/tcritchl/DNN/training1/X_train_{file}.npy', X_train_scaled)
        np.save(f'/eos/user/t/tcritchl/DNN/testing1/X_test_{file}.npy', X_test_scaled)
        np.save(f'/eos/user/t/tcritchl/DNN/training1/y_train_{file}.npy', y_train)
        np.save(f'/eos/user/t/tcritchl/DNN/testing1/y_test_{file}.npy', y_test)
        print(f"Preprocessed data saved successfully for {file}.")
