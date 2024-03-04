import json
import uproot
import os

# Load configuration from JSON file
with open('/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/xgboost_batch/configuration.json') as config_file:
    config = json.load(config_file)

base_path = config["preparation_input_path_HNLs"]

# Define variables
variables = [
    "RecoDiJet_delta_R",
    "RecoElectron_DiJet_delta_R",
    "RecoElectronTrack_absD0",
    "RecoDiJet_phi",
    "RecoMissingEnergy_theta",
    "RecoMissingEnergy_e",
    "RecoElectron_lead_e"
]

# Load BDT cut data from JSON file
with open('test_xgboost_results7_10fb.json', 'r') as file:
    data = json.load(file)

# Extract BDT cuts for each mass-angle combination
bdt_cuts_dict = {}
for key, value in data.items():
    parts = key.split('_')
    mass = float(parts[1].replace('GeV', ''))
    angle = parts[2].replace('1e-', '-').replace('p', '.')
    peak_significance_entry = max(value['significance_list'], key=lambda x: x[0])
    bdt_cut = peak_significance_entry[2]  # Third entry of the significance_list
    if mass not in bdt_cuts_dict:
        bdt_cuts_dict[mass] = {}
    bdt_cuts_dict[mass][angle] = bdt_cut

# Dictionary to store minimum values of variables for each signal point
variable_min_values = {mass: {angle: {variable: float('inf') for variable in variables} for angle in bdt_cuts_dict[mass]} for mass in bdt_cuts_dict}

# Process signal files
for mass, angles_dict in bdt_cuts_dict.items():
    for angle, bdt_cut in angles_dict.items():
        signal_file = os.path.join(base_path, f"test_signal_{mass}GeV_{angle}.root")
        if not os.path.exists(signal_file):
            print(f"File {signal_file} does not exist")
            continue
        
        # Open ROOT file
        file = uproot.open(signal_file)
        tree = file["events_modified_signal"]
        
        # Get BDT output values
        bdt_output_values = tree["bdt_output_signal"].array()
        
        # Iterate over events
        for i, bdt_output_value in enumerate(bdt_output_values):
            if bdt_output_value >= bdt_cut:
                # Extract variable values for the event
                event_variables = {variable: tree[variable].array()[i] for variable in variables}
                
                # Update minimum values for each variable
                for variable, value in event_variables.items():
                    variable_min_values[mass][angle][variable] = min(variable_min_values[mass][angle][variable], value)

# Store minimum values of variables for each signal point in a JSON file
with open('variable_min_values.json', 'w') as outfile:
    json.dump(variable_min_values, outfile, indent=4)
