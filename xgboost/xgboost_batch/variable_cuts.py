import json
import uproot
import os

# Define the base path
base_path = "/eos/user/t/tcritchl/xgBOOST/testing7/"

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

# Dictionary to store BDT cuts for each signal point
bdt_cuts_dict = {}
for signal_point, value in data.items():
    significance_list = value['significance_list']
    max_significance_entry = max(significance_list, key=lambda x: x[0])
    bdt_cuts_dict[signal_point] = max_significance_entry[2]  # Third entry of the significance_list

# Dictionary to store minimum values of variables for each signal point
variable_min_values = {}

# Process signal files
for signal_point, bdt_cut in bdt_cuts_dict.items():
    signal_file = os.path.join(base_path, f"test_{signal_point}.root")
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
            if signal_point not in variable_min_values:
                variable_min_values[signal_point] = {variable: float('inf') for variable in variables}
            for variable, value in event_variables.items():
                variable_min_values[signal_point][variable] = min(variable_min_values[signal_point][variable], value)

# Store minimum values of variables for each signal point in a JSON file
with open('variable_min_values.json', 'w') as outfile:
    json.dump(variable_min_values, outfile, indent=4)
