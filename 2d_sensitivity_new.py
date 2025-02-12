import ROOT
from ROOT import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import math
import os
import json
from scipy.interpolate import griddata
import json

uncertainty_count_factor = 0.1

angles_list = []
masses_list = []

variable_list = [
    ["RecoElectron_lead_e", "Reco lead electron energy [GeV]"], #variable name in histo[0], axis title[1]
    ["RecoDiJet_delta_R", "Reco di-jet #Delta R [Rad.]"],
    ["RecoDiJet_angle", "Reco di-jet #Psi [Rad.]"],
    ["RecoElectron_LeadJet_delta_R", "Reco lead jet #DeltaR [Rad.]"],
    ["RecoElectron_SecondJet_delta_R", "Reco second jet #Delta R [Rad.]"],
    ["RecoElectron_DiJet_delta_R","Reco electron di-jet #Delta R [Rad.]"],
    ["RecoLeadElectron_Pmiss_delta_theta", "Reconstructed electron missing momentum angle #theta [Rad.]"],
    ["RecoElectronTrack_absD0sig", "Reco electron |d_{0}| [mm] sig"],
    ["RecoElectronTrack_absD0cov", "Reco electron |d_{0}| [mm] cov"],
    ["RecoElectronTrack_absD0", "Reco electron |d_{0}| [mm]"],
    ["RecoDiJet_phi", "Reco DiJet #phi [Rad.]"],
    ["RecoMissingEnergy_theta", "Reco Missing Energy #theta [Rad.]"],
    ["RecoMissingEnergy_e", "Reco missing energy [GeV]"],
    ["RecoDiJetElectron_invMass", "Mass [GeV]"] #for invmass of the HNL
]
chosen_variable = variable_list[13] 

luminosity = 10000 #10 fb^-1 as 1e4 pb^-1
#luminosity = 150000000 #150 ab^-1 as 1.5e8 pb^-1

#set these to true or false
normalisation = True 
log_scale = True

#pick your selection
selection = "selMissingEGt12_EleEGt35_AngleLt24_DiJetDRLt3" #all selections

input_dir_bkg = "/afs/cern.ch/work/t/tcritchl/full_background_21Nov_2023/" #bb cc and 4body samples
input_dir_sgl = "/eos/user/t/tcritchl/HNLs/final/" #signals 

output_dir =  "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/HNL_sample_creation/signal_HNLS/SignalvsBackground/testfinalcuts/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

file_Zbb = input_dir_bkg + 'p8_ee_Zbb_ecm91_'+selection+'_histo'+'.root'
file_Zcc = input_dir_bkg + 'p8_ee_Zcc_ecm91_' + selection+'_histo'+'.root'
file_4body = input_dir_bkg + 'ejjnu_' + selection+'_histo'+'.root'

with open("/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/HNL_sample_creation/event_info.json", "r") as json_file:
    events_info = json.load(json_file)

# Define the list of signal events
files_list_signal = []

for file_name in os.listdir(input_dir_sgl):
    if file_name.endswith(f"{selection}_histo.root"):
        print(f"file name is {file_name}")
        mass = file_name.split('_')[3]
        angle = file_name.split('_')[4]
        print(mass)
        print(angle)
        # Find the corresponding entry in events_info
        for event_info in events_info:
            if event_info[1] == mass and event_info[0] == angle:
                cross_section = event_info[2]
                selection_scale_sgl = event_info[3]
                total_events = 1 
                break
        else:
            # Continue to the next file if there's no matching entry in events_info
            continue

        # Check if the selection_scale_sgl is not null
        if selection_scale_sgl is not None:
            file_path = os.path.join(input_dir_sgl, file_name)
            signal_info = [file_path, chosen_variable[0], f"{mass} HNL", cross_section, total_events, selection_scale_sgl]
            files_list_signal.append(signal_info)

###background
cross_sections_bg = [5215.46, 6654.46,0.01399651855102697] #pb
total_events_bg = [2.640333103799864e-05,6645.46, 0.00036679999999999975] #typically normalised to 1 pb of luminosity
selection_scale_bg = [2/(499786495),1/(438738637),2620/100000]
files_list_bg = [
    [file_Zcc, chosen_variable[0], "Z #rightarrow cc", cross_sections_bg[0], total_events_bg[0], selection_scale_bg[0]],
    [file_Zbb, chosen_variable[0], "Z #rightarrow bb", cross_sections_bg[1], total_events_bg[1], selection_scale_bg[1]],
    [file_4body, chosen_variable[0], "Z #rightarrow e #nu qq", cross_sections_bg[2], total_events_bg[2], selection_scale_bg[2]]
]

legend_list_bg = [f[2] for f in files_list_bg]
ratio_list_bg = [f[2] for f in files_list_bg]

legend_list_signal = [f[2] for f in files_list_signal]
ratio_list_signal = [f[2] for f in files_list_signal]

colors_signal = [ROOT.kMagenta - 7, ROOT.kMagenta - 2, ROOT.kMagenta + 3]
colors_bg = [856, 410, 801, 629, 879, 602, 921, 622]


def make_hist(files_list):
    h_list = []
    for f in files_list:
        print("Looking at file", f[2])
        with ROOT.TFile.Open(f[0]) as my_file:
            print("Getting histogram for variable", f[1])
            hist = my_file.Get(f[1])  # Select the chosen variable from the histo root file

            if normalisation:
                # Apply normalization based on cross section, total events, and luminosity, and selection fraction surviving
                cross_section = f[3]  # Cross section in pb
                selection_scale = f[5]
                scaling_factor = selection_scale * ((cross_section * luminosity) / (hist.Integral()))
                hist.Scale(scaling_factor)

            hist.SetDirectory(0)  # Make the chosen histogram independent of the directory

            h_list.append((f[0], hist))  # Include the histogram and file name in the list
            print("Histogram added to h_list")
        
        print("-----------------------")
    return h_list


#significance of the most significant bin for each signal
def max_significance(files_list, n_bins, h_list_bg):
    max_sig_list = []

    for file_info in files_list:
        file_name, hist = file_info

        max_sig_value = 0

        for bin_idx in range(1, n_bins + 1):
            s = hist.Integral(bin_idx, bin_idx)
            print(f"signal integral {s}")
            b = sum(bg_hist[1].Integral(bin_idx, bin_idx) for bg_hist in h_list_bg)
            print(f"b intergal {b}")
            sigma = b * uncertainty_count_factor

            if s + b > 0 and b > 0 and s != 0 and sigma != 0:
                n = s + b
                current_significance = math.sqrt(abs(
                    2 * (n * math.log((n * (b + sigma**2)) / (b**2 + n * sigma**2)) - (b**2 / sigma**2) * math.log((1 + (sigma**2 * (n - b)) / (b * (b + sigma**2))))
                )))

                if current_significance > max_sig_value:
                    max_sig_value = current_significance

        max_sig_list.append((max_sig_value, file_name))  # keep the significance value and file name

    return max_sig_list

#averaged significance of the most significant bin \pm 5 Gev

def max_significance_weighted(files_list, n_bins, h_list_bg):
    max_sig_list = []

    for file_info in files_list:
        file_name, hist = file_info

        max_sig_value = 0

        for bin_idx in range(1, n_bins + 1):
            s = hist.Integral(bin_idx, bin_idx)
            print(f"signal integral {s}")
            b = sum(bg_hist[1].Integral(bin_idx, bin_idx) for bg_hist in h_list_bg)
            print(f"b integral {b}")
            sigma = b * uncertainty_count_factor

            if s + b > 0 and b > 0 and s != 0 and sigma != 0:
                n = s + b
                current_significance = math.sqrt(abs(
                    2 * (n * math.log((n * (b + sigma**2)) / (b**2 + n * sigma**2)) - (b**2 / sigma**2) * math.log((1 + (sigma**2 * (n - b)) / (b * (b + sigma**2))))
                )))

                if current_significance > max_sig_value:
                    max_sig_value = current_significance
                    peak_bin_idx = bin_idx

        # Calculate average significance in the region around the peak
        region_width = 2
        start_bin = max(1, peak_bin_idx - region_width)
        end_bin = min(n_bins, peak_bin_idx + region_width)

        region_significances = []
        for idx in range(start_bin, end_bin + 1):
            s = hist.Integral(idx, idx)
            b = sum(bg_hist[1].Integral(idx, idx) for bg_hist in h_list_bg)
            sigma = b * uncertainty_count_factor

            if s + b > 0 and b > 0 and s != 0 and sigma != 0:
                n = s + b
                current_significance = math.sqrt(abs(
                    2 * (n * math.log((n * (b + sigma**2)) / (b**2 + n * sigma**2)) - (b**2 / sigma**2) * math.log((1 + (sigma**2 * (n - b)) / (b * (b + sigma**2))))
                )))
                region_significances.append(current_significance)

        avg_significance = sum(region_significances) / len(region_significances) if region_significances else 0
        max_sig_list.append((avg_significance, file_name))

    return max_sig_list


h_list_signal = make_hist(files_list_signal)
h_list_bg = make_hist(files_list_bg)
n_bins = h_list_bg[0][1].GetNbinsX()
x_min = h_list_bg[0][1].GetXaxis().GetXmin()
x_max = h_list_bg[0][1].GetXaxis().GetXmax()

max_sig_list = max_significance(h_list_signal, n_bins, h_list_bg)

def make_hist_2D(max_sig_list):
    
    print("Building 2D histogram..")
    mass_list = []
    coupling_list = []
    significance_list = []

    for info in max_sig_list:
        
        files = info[1]
        print(f"files = {files}")
        angle_string = files.split('_')[4]
        print(angle_string)
        angle1 = angle_string.strip('Ve')
        angle2 = angle1.replace('p', '.')
        print(angle2)
        angle_exponent = float(angle2.strip('1e'))
        print(angle_exponent)
        angle = 10**(angle_exponent)
        print(angle)
        massesGeV = files.split('_')[3]
        masses = massesGeV.strip('GeV')
        print(f"masses = {masses}")
        max_sig = info[0]
        x = masses
        y = np.log10(angle*angle)
        z = max_sig
        mass_list.append(x)
        coupling_list.append(y)
        significance_list.append(z)
    
    data_points = list(zip(mass_list, coupling_list, significance_list))
    
    # Save the list of data points to a JSON file
    json_filename = "data_points.json"
    with open(json_filename, "w") as json_file:
        json.dump(data_points, json_file)

    print(f"Data points saved to {json_filename}")

    print(data_points)
    print(f"masses {mass_list}")
    print(f"couplings {coupling_list}")
    print(f"signifiancees {significance_list}")
    
    masses = [float(data[0]) for data in data_points]
    angles = [float(data[1]) for data in data_points]  # Angle squared is 10^(2 * exponent)
    significances = [data[2] for data in data_points]

    x_range = (0, 90)
    y_range = (-11, -3)
    
    x_bins = [4, 6, 14, 16, 24, 26, 34, 36, 44, 46, 54, 56, 64, 66, 74, 76, 84, 86]
    y_bins = np.arange(-10.75, -2.25, 0.5)

    n_bins = [x_bins, y_bins]

    norm = LogNorm(vmin=1e-5, vmax=5)

    cmap = plt.cm.RdPu

    plt.hist2d(masses, angles, bins=n_bins, range=[x_range, y_range], weights=significances, cmap=cmap, norm=norm)

    hist, edges = np.histogramdd((masses, angles), bins=n_bins, range=[x_range, y_range], weights=significances)
    x_edges, y_edges = edges
    
    plt.colorbar()
    
    ###delimiting line
    threshold = 2.0
    x_grid, y_grid = np.meshgrid(x_bins, y_bins)

    # Perform the interpolation
    z_interp = griddata((masses, angles), significances, (x_grid, y_grid), method='linear')
    plt.contour(x_grid, y_grid, z_interp, levels=[threshold], colors='red', linestyles='dashed')

    plt.xlabel("HNL mass [GeV]", fontsize=12)
    plt.ylabel("log($U^2$)", fontsize=12)
    plt.title("Z-significance at $150 \, \mathrm{ab}^{-1}$", fontsize=14)
    
    
    for i in range(len(x_edges) -1):
        for j in range(len(y_edges) -1):
            if hist[i, j] > 0:
                plt.text(x_edges[i] +0.2, y_edges[j]+0.1, f'{hist[i, j]:.1f}', color='white', fontsize = 7)

    
    plt.savefig('exclusionplot.pdf', format='pdf')
    plt.show()

make_hist_2D(max_sig_list)