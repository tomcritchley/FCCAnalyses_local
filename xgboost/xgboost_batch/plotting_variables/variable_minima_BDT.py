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

print("Loading BDT cut data from JSON file...")
with open('test_xgboost_results7_10fb.json', 'r') as file:
    data = json.load(file)

bdt_cuts_dict = {}
for signal_point, value in data.items():
    print(f"signal point: {signal_point}")
    significance_list = value['significance_list']
    max_significance_entry = max(significance_list, key=lambda x: x[0])
    bdt_cuts_dict[signal_point] = max_significance_entry[2]  

variable_min_values = {}

print("Processing signal files...")
for signal_point, bdt_cut in bdt_cuts_dict.items():
    print(f"Processing signal point: {signal_point}...")
    signal_file = os.path.join(base_path, f"test_{signal_point}.root")
    if not os.path.exists(signal_file):
        print(f"File {signal_file} does not exist")
        continue
    
    _, mass, angle = signal_point.split('_')
    
    file = uproot.open(signal_file)
    tree = file["events_modified_signal"]
    
    bdt_output_values = tree["bdt_output_signal"].array()
    
    for i, bdt_output_value in enumerate(bdt_output_values):
        if bdt_output_value >= bdt_cut:
            print("Event passed BDT cut:")
            print(f"  - Signal Point: {signal_point}")
            print(f"  - Mass: {mass} GeV")
            print(f"  - Angle: {angle}")
            print(f"  - BDT Output Value: {bdt_output_value}")
            
            # Extract variable values for the event
            event_variables = {variable: tree[variable].array()[i] for variable in variables}
            
            print("  - Variable Values:")
            for variable, value in event_variables.items():
                print(f"    - {variable}: {value}")
                
            # Append variable values to the list for the current signal point
            if signal_point not in variable_min_values:
                variable_min_values[signal_point] = {variable: [] for variable in variables}
            for variable, value in event_variables.items():
                variable_min_values[signal_point][variable].append(value)
                
# Calculate the smallest value for each variable for each signal point
for signal_point, values_dict in variable_min_values.items():
    print(f"\nCalculating smallest variable values for signal point: {signal_point}")
    min_values = {variable: min(values) for variable, values in values_dict.items()}
    print("Smallest variable values:")
    for variable, min_value in min_values.items():
        print(f"  - {variable}: {min_value}")

# Convert array values to float
variable_min_values = {signal_point: {variable: min(map(float, value)) for variable, value in values.items()} for signal_point, values in variable_min_values.items()}
print(f"all done, printing values")
print(variable_min_values)
# Store minimum values of variables for each signal point in a JSON file
print("Storing minimum values of variables in a JSON file...")
try:
    with open('variable_min_values.json', 'w') as outfile:
        json.dump(variable_min_values, outfile, indent=4)
except Exception as e:
    print("An error occurred while dumping data to JSON file. Printing data for inspection:")
    print(variable_min_values)
    print("Error:", e)