import ROOT
from ROOT import *
import numpy as np
import math
import os
import json
import json

"""
/eos/user/t/tcritchl/new_variables_HNL_test_March24/final --> location of the histograms for the HNLs
/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/final --> location of the histograms for 4body, cc and bb

background format:

ejjnu_selMissingEGt12_EleEGt35_AngleLt24_DiJetDRLt3_histo.root 
p8_ee_Zbb_ecm91_selMissingEGt12_EleEGt35_AngleLt24_DiJetDRLt3_histo.root
p8_ee_Zcc_ecm91_selMissingEGt12_EleEGt35_AngleLt24_DiJetDRLt3_histo.root

HNL format for 20 gev |Ven|^2 = 5:

HNL_Dirac_ejj_20GeV_1e-2p5Ve_selMissingEGt12_EleEGt35_AngleLt24_DiJetDRLt3_histo.root

"""

### variable list, complete with the latest variables including the number of primary tracks and chi2 etc ###
variable_list = [
    ["RecoElectron_lead_e", "Reco lead electron energy [GeV]"], #variable name in histo[0], axis title[1]
    ["RecoDiJet_delta_R", "Reco di-jet #Delta R"], #1
    ["RecoDiJet_angle", "Reco di-jet #Psi [Rad.]"], #2
    ["RecoElectron_LeadJet_delta_R", "Reco lead jet #DeltaR"], #3
    ["RecoElectron_SecondJet_delta_R", "Reco second jet #Delta R"], #4
    ["RecoElectron_DiJet_delta_R","Reco electron di-jet #Delta R"], #5
    ["RecoLeadElectron_Pmiss_delta_theta", "Reconstructed electron missing momentum #theta [Rad.]"], #6
    ["RecoElectronTrack_absD0sig", "Reco electron |d_{0}| [mm] sig"], #7
    ["RecoElectronTrack_absD0cov", "Reco electron |d_{0}| [mm] cov"], #8
    ["RecoElectronTrack_absD0", "Reco electron |d_{0}| [mm]"], #9
    ["RecoDiJet_phi", "Reco DiJet #phi [Rad.]"], #10
    ["RecoMissingEnergy_theta", "Reco Missing Energy   #theta [Rad.]"], #11
    ["RecoMissingEnergy_e", "Reco missing energy [GeV]"], #12
    ["RecoDiJetElectron_invMass", "Mass [GeV]"], #for invmass of the HNL #13
    ["ntracks", "Number of tracks"], #14
    ["n_primt", "Number of primary tracks"], #15
    ["Vertex_chi2", "Chi^{2} of the primary vertex"], #16
]

chosen_variable = variable_list[13] # invariant mass

angles_list = []
masses_list = []

#luminosity = 10000 #10 fb^-1 as 1e4 pb^-1
luminosity = 205000000 #205 ab^-1 as 1.5e8 pb^-1
lumi_label = "205ab"

# normalise to the lumi chosen 
normalisation = True

#pick your selection
selection = "selMissingEGt12_EleEGt35_AngleLt24_DiJetDRLt3" #all selections

input_dir_bkg = "/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/final/" #bb cc and 4body samples
input_dir_sgl = "/eos/user/t/tcritchl/new_variables_HNL_test_March24/final/" #signals 

output_dir =  "/afs/cern.ch/user/t/tcritchl/testfinal/FCCAnalyses_local/cut_and_count_205ab/signficance/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")

### HNL associated madgraph cross section info ###

json_file = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"
with open(json_file, "r") as json_file:
    events_info = json.load(json_file)

# Define the list of signal events
files_list_signal = []

# Loop through files in input directory
for file_name in os.listdir(input_dir_sgl):
    if file_name.endswith(f"{selection}_histo.root"):
        print(f"File name is {file_name}")

        # Extract mass and angle from filename
        parts = file_name.split('_')
        mass = parts[3]  # "20GeV"
        angle = parts[4]  # "1e-2p5Ve"

        # Reconstruct the JSON key
        json_key = f"HNL_Dirac_ejj_{mass}_{angle}"
        print(f"Looking for JSON key: {json_key}")

        # Check if the key exists in JSON
        if json_key in events_info:
            cross_section = events_info[json_key]["cross_section_pb"]
            print(f"Found cross-section: {cross_section}")
        else:
            print(f"No signal cross-section information found for {mass}, {angle}")
            continue  # Skip this file if there's no matching entry

        # Store the relevant information
        file_path = os.path.join(input_dir_sgl, file_name)
        signal_info = [file_path, chosen_variable[0], f"{mass} HNL", cross_section]
        files_list_signal.append(signal_info)

### background processing ###
cross_sections_bg = [5215.46, 6654.46,0.01399651855102697] #pb

file_Zbb = input_dir_bkg + 'p8_ee_Zbb_ecm91_'+selection+'_histo'+'.root'
file_Zcc = input_dir_bkg + 'p8_ee_Zcc_ecm91_' + selection+'_histo'+'.root'
file_4body = input_dir_bkg + 'ejjnu_' + selection+'_histo'+'.root'

files_list_bg = [
    [file_Zcc, chosen_variable[0], "Z #rightarrow cc", cross_sections_bg[0]],
    [file_Zbb, chosen_variable[0], "Z #rightarrow bb", cross_sections_bg[1]],
    [file_4body, chosen_variable[0], "Z #rightarrow e #nu qq", cross_sections_bg[2]]
]
#10% background uncertainty for the significance
uncertainty_count_factor = 0.1 

def make_hist(files_list):
    h_list = []
    for f in files_list:
        print("Looking at file", f[2])
        with ROOT.TFile.Open(f[0]) as my_file:
            print("Getting histogram for variable", f[1])
            hist = my_file.Get(f[1])  # Select the chosen variable from the histo root file

            if normalisation:
                
                # Apply normalization based on cross section, total events, and luminosity
                cross_section = f[3]  # Cross section in pb
                
                # normalisation as N' = N_sel * S --> S = (x-sec * L' / N)
                ### new estimation of the zpole lumi --> 205 ab^-1 x 0.49 (4 body background does NOT require this factor) ###
                
                ## for 4 body ##
                if f[2] == "Z #rightarrow e #nu qq":
                    print(f'processing four body, no factor of 0.49 applied...')
                    scaling_factor =  (cross_section * luminosity) / (hist.Integral())
                ## for z->bb, z->cc ##
                else:
                    scaling_factor =  (cross_section * luminosity * 0.49) / (hist.Integral())

                hist.Scale(scaling_factor)

            # Make the chosen histogram independent of the directory
            hist.SetDirectory(0)

            # Include the histogram and file name in the list
            h_list.append((f[0], hist))
            
            print("Histogram sucessfully added!")
        
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

h_list_signal = make_hist(files_list_signal) ## list of signals!
h_list_bg = make_hist(files_list_bg)
n_bins = h_list_bg[0][1].GetNbinsX()
x_min = h_list_bg[0][1].GetXaxis().GetXmin()
x_max = h_list_bg[0][1].GetXaxis().GetXmax()

max_sig_list = max_significance(h_list_signal, n_bins, h_list_bg)

def make_sig(max_sig_list):

    print("Building significance")
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
    json_filename = f"cut_count_{lumi_label}.json"
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

make_sig(max_sig_list)

"""
#averaged significance of the most significant bin \pm 2 Gev

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

        #number of bins +/- the peak
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
"""