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

tree_name = "events"
base_HNL = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/"
json_file = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"
target_luminosity = 10000  #pb^-1 ---> =10fb^-1

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

#for signal
def load_and_filter_data(filepath, x_sec, tree_name, variables, filter_func):
    with uproot.open(filepath) as file:
        df = file[tree_name].arrays(variables, library="pd")
        df['weight'] = (float(x_sec) * target_luminosity) / len(df)
        return filter_func(df)
    
#for chunks of background   
def load_and_preprocess_bkg(filepaths_and_xsecs, filter_func, label):
    dfs = []
    for filepath, x_sec in filepaths_and_xsecs:
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
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue
    return pd.concat(dfs, ignore_index=True)


def basic_filter(df):
    return df[
        (df["n_RecoElectrons"] == 1) & 
        (df["RecoElectron_lead_e"] > 15) &
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

    background_files = [(os.path.join(dir, file), x_sec) for dir, x_sec in background_dirs for file in os.listdir(dir) if file.endswith('.root')]
    background_df = load_and_preprocess_bkg(background_files, basic_filter, 0)

    #Balancing the datasets
    n_signal = len(signal_df)
    n_background = len(background_df)

    #50/50 split for training and testing the signal
    df_train_signal, df_test_signal = train_test_split(signal_df, test_size=0.5, random_state=42)
    
    #downsampling for the background to reduce class inequality 
    df_train_background = background_df.sample(n=min(n_signal, n_background // 2), random_state=42)
    df_test_background = background_df.drop(df_train_background.index)

    #adjust the weights based on the fraction which is used during testing
    background_weight_scale = len(background_df) / len(df_test_background)
    signal_weight_scale = len(signal_df) / len(df_test_signal)
    df_test_background['weight'] *= background_weight_scale
    df_test_signal['weight'] *= signal_weight_scale

    n_background_train = len(df_train_background)
    fraction_background_used_in_training = n_background_train / n_background
    print(f"Fraction of background used in training: {fraction_background_used_in_training:.2f}") #needed for understanding the split to normalise correctly

    testing_fraction_background = len(df_test_background) / len(background_df)
    testing_fraction_signal = len(df_test_signal) / len(signal_df)

    #json file can be used for normalising during testing
    stats = {
        'label': args.label,
        'training_fraction_background': testing_fraction_background,
        'training_fraction_signal': testing_fraction_signal
    }

    background_efficiency_json = f'/eos/user/t/tcritchl/DNN/background_stats_{args.label}.json'
    with open(background_efficiency_json, 'w') as jf:
        json.dump(stats, jf, indent=4)

    df_train = pd.concat([df_train_signal, df_train_background], ignore_index=True)

    df_test = pd.concat([df_test_signal, df_test_background], ignore_index=True)

    convert_to_numpy(df_train, ['RecoElectronTrack_absD0', 'RecoElectronTrack_absD0sig', 'RecoMissingEnergy_theta', 'RecoMissingEnergy_e'])
    convert_to_numpy(df_test, ['RecoElectronTrack_absD0', 'RecoElectronTrack_absD0sig', 'RecoMissingEnergy_theta', 'RecoMissingEnergy_e'])

    #shuffling datasets
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    try:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df_train.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    cbar_kws={'label': 'Correlation coefficient'},
                    xticklabels=correlation_matrix.columns,
                    yticklabels=correlation_matrix.columns)

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title(f"Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f'/eos/user/t/tcritchl/DNN/correlation_matrix_{args.label}.pdf')
        plt.show()
    except Exception as e:
        print(f"something went wrong with the correlation matrix...: {e}")

    X_train, y_train, weights_train = df_train[variables], df_train['label'], df_train['weight']
    X_test, y_test, weights_test = df_test[variables], df_test['label'], df_test['weight']

    #make the variables less wild, centred around 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save(f'/eos/user/t/tcritchl/DNN/training4/X_train_{args.label}.npy', X_train_scaled)
    np.save(f'/eos/user/t/tcritchl/DNN/testing4/X_test_{args.label}.npy', X_test_scaled)
    np.save(f'/eos/user/t/tcritchl/DNN/training4/y_train_{args.label}.npy', y_train)
    np.save(f'/eos/user/t/tcritchl/DNN/testing4/y_test_{args.label}.npy', y_test)
    np.save(f'/eos/user/t/tcritchl/DNN/testing4/weights_test_{args.label}.npy', weights_test)

    print(f"Data preparation complete for label: {args.label}")

if __name__ == "__main__":
    prepare_datasets()
