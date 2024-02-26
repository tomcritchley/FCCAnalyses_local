# stage 3

import os

# Define the masses and angles
masses = [10, 20, 30, 40, 50, 60, 70, 80]
angles_scientific = ['1e-2','1e-2.5','1e-3','1e-3.5','1e-4','1e-4.5','1e-5']
angle_decimal = [0.01, 0.0031622776601683795, 0.001, 0.00031622776601683794, 0.0001, 3.1622776601683795e-5, 1e-5]

def convert_to_filename(angle):
    return str(angle).replace('.', 'p')

def run_command(command):
    os.system(command)

# Function to unzip and rename the file
def unzip_and_rename(source, destination):
    run_command(f"gunzip {source}")
    # Get the filename without the extension
    filename = os.path.splitext(destination)[0]
    os.rename(os.path.splitext(source)[0], filename)

# Directory where the generated directories are located
base_dir = "./"

# Iterate through each process card file
for mass in masses:
    for angle in angles_scientific:
        filename = f"mg5_proc_card_HNL_Dirac_ejj_{mass}GeV_{convert_to_filename(angle)}Ve.dat"
        run_command(f"./bin/mg5_aMC {filename}")

# Iterate through each directory
for dir_name in os.listdir(base_dir):
    # Check if it's a directory and matches the pattern
    if os.path.isdir(os.path.join(base_dir, dir_name)) and dir_name.startswith("HNL_Dirac_ejj_"):
        # Navigate to the Events directory
        events_dir = os.path.join(base_dir, dir_name, "Events")
        if os.path.exists(events_dir):
            # Navigate to the run_01 directory
            run_dir = os.path.join(events_dir, "run_01")
            if os.path.exists(run_dir):
                # Find the unweighted_events.lhe.gz file
                for file in os.listdir(run_dir):
                    if file.startswith("unweighted_events") and file.endswith(".lhe.gz"):
                        # Unzip and rename the file
                        source_file = os.path.join(run_dir, file)
                        destination_file = os.path.join(run_dir, f"{dir_name}.lhe")
                        unzip_and_rename(source_file, destination_file)
                        print(f"Processed: {dir_name}")
                        break