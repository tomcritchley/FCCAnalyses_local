import json
import os
from glob import glob

# Define the path where all your individual JSON files are stored
json_files_path = '/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/Run6_opt/*.json'  # Use *.json to match all JSON files

# Initialize an empty dictionary to store combined results
combined_results = {}

# Loop through each JSON file in the directory
for json_file in glob(json_files_path):
    with open(json_file, 'r') as file:
        # Load the content of the current JSON file
        current_results = json.load(file)
        
        # Merge the current results into the combined_results dictionary
        # This assumes each JSON file's top-level keys are unique across all files
        combined_results.update(current_results)

# Define the path for the combined JSON file
combined_json_path = '/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_results_run6_opt.json'

# Save the combined results to a new JSON file
with open(combined_json_path, 'w') as combined_file:
    json.dump(combined_results, combined_file, indent=2)

print(f'Combined JSON saved to {combined_json_path}')
