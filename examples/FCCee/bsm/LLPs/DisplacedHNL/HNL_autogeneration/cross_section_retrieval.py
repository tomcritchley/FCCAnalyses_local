import os
import json

# Define the base directory
base_dir = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3"

# Define the output JSON file path
output_json_file = "HNL_cross_sections_Feb24.json"

# Initialize a dictionary to store cross-section information
cross_sections = {}

# Function to extract integrated weight from banner.txt file
def extract_cross_section(banner_file):
    with open(banner_file, 'r') as f:
        for line in f:
            if line.startswith('#  Integrated weight (pb)  :'):
                return float(line.split(':')[1].strip())
    return None

# Iterate through each directory
for dir_name in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dir_name)
    if os.path.isdir(dir_path) and dir_name.startswith("HNL_Dirac_ejj_"):
        # Extract mass and angle from directory name
        parts = dir_name.split('_')
        if len(parts) == 5:  # Ensure directory name has correct format
            mass = parts[3].replace('GeV', '')
            angle_str = parts[4].replace('Ve', '').replace('p', '.')
        
            # Read banner.txt file
            banner_file = os.path.join(dir_path, "Events", "run_01", "run_01_tag_1_banner.txt")
        
            # Extract cross-section
            cross_section = extract_cross_section(banner_file)
        
            # Store cross-section information in the dictionary
            cross_sections[dir_name] = {
                "mass": mass,
                "angle": angle_str,
                "cross_section_pb": cross_section
            }

# Write cross-section information to JSON file
with open(output_json_file, 'w') as json_file:
    json.dump(cross_sections, json_file, indent=4)

print("Cross-section information stored in", output_json_file)
