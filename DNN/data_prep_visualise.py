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

def load_and_filter_data(filepath, x_sec, tree_name, variables, filter_func):
    with uproot.open(filepath) as file:
        df = file[tree_name].arrays(variables, library="pd")
        df['weight'] = (float(x_sec) * target_luminosity) / len(df)
        return filter_func(df)

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

def plot_feature_distributions(X_train, y_train, X_train_scaled, feature_names):
    signal_indices = (y_train == 1)
    background_indices = (y_train == 0)

    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        plt.hist(X_train[signal_indices, i], bins=50, alpha=0.5, label='Signal', density=True)
        plt.hist(X_train[background_indices, i], bins=50, alpha=0.5, label='Background', density=True)
        plt.xlabel(feature)
        plt.ylabel('Normalized Frequency')
        plt.title(f'Original {feature}')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(X_train_scaled[signal_indices, i], bins=50, alpha=0.5, label='Signal', density=True)
        plt.hist(X_train_scaled[background_indices, i], bins=50, alpha=0.5, label='Background', density=True)
        plt.xlabel(f'Scaled {feature}')
        plt.ylabel('Normalized Frequency')
        plt.title(f'Scaled {feature}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots20/{feature}_signal_vs_background_distributions.pdf')
        plt.close()

def prepare_datasets():
    parser = argparse.ArgumentParser(description='Prepare datasets for DNN Training')
    parser.add_argument('--label', required=True, help='Signal label, e.g., 10GeV_1e-2')
    args = parser.parse_args()

    signal_file = f"{base_HNL}/HNL_Dirac_ejj_{args.label}Ve.root"
    signal_x_sec = cross_section_dict[f"HNL_Dirac_ejj_{args.label}Ve"]['cross_section_pb']
    signal_df = load_and_filter_data(signal_file, signal_x_sec, tree_name, variables, basic_filter)
    signal_df['label'] = 1

    background_files = [(os.path.join(dir, file), x_sec) for dir, x_sec in background_dirs for file in os.listdir(dir) if file.endswith('00.root') or file.endswith('ejjnu.root')]
    background_df = load_and_preprocess_bkg(background_files, basic_filter, 0)

    bg_df_groups = {x_sec: df for x_sec, df in background_df.groupby('cross_section')}

    print("Number of events per cross section:")
    for x_sec, df in bg_df_groups.items():
        print(f"Cross section {x_sec}: {len(df)} events (training+testing)")

    training_bg_dfs = []
    training_mask = pd.Series(False, index=background_df.index)

    for x_sec, df in bg_df_groups.items():
        print(f"x-sec is {x_sec} with type {type(x_sec)}")
        if x_sec == 0.014:
            sample_size = int(len(df) * 0.9)
        else:
            sample_size = int(len(df) * 0.3)

        sampled_indices = df.sample(sample_size, random_state=42).index
        training_bg_dfs.append(df.loc[sampled_indices])
        training_mask.loc[sampled_indices] = True

    training_bg_df = pd.concat(training_bg_dfs, ignore_index=True)
    testing_bg_df = background_df.loc[~training_mask]

    print(f"Total training events: {len(training_bg_df)}")
    print(f"Total testing events: {len(testing_bg_df)}")

    df_train_signal, df_test_signal = train_test_split(signal_df, test_size=0.5, random_state=42)

    df_train = pd.concat([df_train_signal, training_bg_df], ignore_index=True).sample(frac=1).reset_index(drop=True)
    df_test = pd.concat([df_test_signal, testing_bg_df], ignore_index=True).sample(frac=1).reset_index(drop=True)

    print(f"Training set: Signal: {len(df_train_signal)}, Background: {len(training_bg_df)}")
    print(f"Testing set: Signal: {len(df_test_signal)}, Background: {len(testing_bg_df)}")

    convert_to_str = lambda x: f"{x:.5f}"
    background_df['cross_section'] = background_df['cross_section'].apply(convert_to_str)
    training_bg_df['cross_section'] = training_bg_df['cross_section'].apply(convert_to_str)
    testing_bg_df['cross_section'] = testing_bg_df['cross_section'].apply(convert_to_str)
    bg_df_groups = {x_sec: df for x_sec, df in background_df.groupby('cross_section')}

    background_weight_scales = {}
    for x_sec, df in bg_df_groups.items():
        total_count = len(df)
        training_count = len(training_bg_df[training_bg_df['cross_section'] == x_sec])
        testing_count = len(testing_bg_df[testing_bg_df['cross_section'] == x_sec])

        if testing_count > 0:
            print(f"for x_sec = {x_sec}, the proportion used in testing is {testing_count / total_count}")
            background_weight_scales[x_sec] = total_count / testing_count
        else:
            background_weight_scales[x_sec] = 0

    for index, row in testing_bg_df.iterrows():
        cross_section = row['cross_section']
        if cross_section in background_weight_scales:
            testing_bg_df.at[index, 'weight'] *= background_weight_scales[cross_section]

    signal_weight_scale = len(signal_df) / len(df_test_signal)
    df_test_signal['weight'] *= signal_weight_scale

    print(f"Train signal count: {len(df_train_signal)}, Test signal count: {len(df_test_signal)}")
    print(f"Train background count: {len(training_bg_df)}, Test background count: {len(testing_bg_df)}")

    convert_to_numpy(df_train, ['RecoElectronTrack_absD0', 'RecoElectronTrack_absD0sig', 'RecoMissingEnergy_theta', 'RecoMissingEnergy_e'])
    convert_to_numpy(df_test, ['RecoElectronTrack_absD0', 'RecoElectronTrack_absD0sig', 'RecoMissingEnergy_theta', 'RecoMissingEnergy_e'])

    training_variables = [
        "RecoDiJet_delta_R",
        "RecoElectron_DiJet_delta_R",
        "RecoElectronTrack_absD0", "RecoDiJet_phi", "RecoMissingEnergy_theta",
        "RecoMissingEnergy_e", "RecoElectron_lead_e", "Vertex_chi2",
        "n_primt", "ntracks"
    ]

    try:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df_train[training_variables].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                    cbar_kws={'label': 'Correlation coefficient'},
                    xticklabels=correlation_matrix.columns,
                    yticklabels=correlation_matrix.columns)

        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title(f"Correlation Matrix")
        plt.tight_layout()
        plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots20/correlation_matrix_{args.label}.pdf')
    except Exception as e:
        print(f"something went wrong with the correlation matrix...: {e}")

    X_train, y_train, weights_train = df_train[training_variables], df_train['label'], df_train['weight']
    X_test, y_test, weights_test = df_test[training_variables], df_test['label'], df_test['weight']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    plot_feature_distributions(X_train.values, y_train, X_train_scaled, training_variables)

    np.save(f'/eos/user/t/tcritchl/DNN/training20/X_train_{args.label}.npy', X_train_scaled)
    np.save(f'/eos/user/t/tcritchl/DNN/testing20/X_test_{args.label}.npy', X_test_scaled)
    np.save(f'/eos/user/t/tcritchl/DNN/training20/y_train_{args.label}.npy', y_train)
    np.save(f'/eos/user/t/tcritchl/DNN/testing20/y_test_{args.label}.npy', y_test)
    np.save(f'/eos/user/t/tcritchl/DNN/testing20/weights_test_{args.label}.npy', weights_test)
    np.save(f'/eos/user/t/tcritchl/DNN/testing20/weights_train_{args.label}.npy', weights_train)

    print(f"Data preparation complete for label: {args.label}")

if __name__ == "__main__":
    prepare_datasets()
