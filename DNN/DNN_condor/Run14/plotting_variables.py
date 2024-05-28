import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_signal_vs_background(variable, label, variable_index):
    # Load the training data
    X_train = np.load(f'/eos/user/t/tcritchl/DNN/training14/X_train_{label}.npy')
    y_train = np.load(f'/eos/user/t/tcritchl/DNN/training14/y_train_{label}.npy')
    weights_train = np.load(f'/eos/user/t/tcritchl/DNN/testing14/weights_train_{label}.npy')
    
    # Separate the signal and background data
    signal = X_train[y_train == 1][:, variable_index]
    background = X_train[y_train == 0][:, variable_index]
    signal_weights = weights_train[y_train == 1]
    background_weights = weights_train[y_train == 0]

    print("Mean and standard deviation of the scaled training data:")
    print(f"Mean of signal: {np.mean(signal, axis=0)}")
    print(f"Mean of background: {np.mean(background, axis=0)}")
    print(f"Standard Deviation of signal: {np.std(signal, axis=0)}")
    print(f"Standard Deviation of background: {np.std(background, axis=0)}")

    
    # Plotting the signal vs background
    plt.figure(figsize=(10, 6))
    plt.hist(signal, bins=50, alpha=0.5, label='Signal', weights=signal_weights, density=True)
    plt.hist(background, bins=50, alpha=0.5, label='Background', weights=background_weights, density=True)
    
    plt.xlabel(variable)
    plt.ylabel('Normalized Frequency')
    plt.title(f'Signal vs Background: {variable}')
    plt.legend(loc='best')
    
    # Save the plot
    plt.savefig(f'/eos/user/t/tcritchl/DNN/DNN_plots_variables/signal_vs_background_{variable}_{label}.pdf')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot signal vs background for a chosen variable')
    parser.add_argument('--label', required=True, help='Signal label, e.g., 10GeV_1e-2')
    parser.add_argument('--variable', required=True, help='Variable name to plot')
    args = parser.parse_args()
    
    # Dictionary to map variable names to their respective column indices
    variable_indices = {
        "RecoDiJet_delta_R": 0,
        "RecoElectron_DiJet_delta_R": 1,
        "RecoElectronTrack_absD0": 2,
        "RecoDiJet_phi": 3,
        "RecoMissingEnergy_theta": 4,
        "RecoMissingEnergy_e": 5,
        "RecoElectron_lead_e": 6,
        "Vertex_chi2": 7,
        "n_primt": 8,
        "ntracks": 9
    }
    
    if args.variable not in variable_indices:
        raise ValueError(f"Variable {args.variable} not found in the dataset.")
    
    plot_signal_vs_background(args.variable, args.label, variable_indices[args.variable])

if __name__ == "__main__":
    main()
