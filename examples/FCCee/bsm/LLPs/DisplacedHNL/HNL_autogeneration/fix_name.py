# fixed mis named and mis located lhe files to the correct dirs (4)

import os

# Define the masses and angles
masses = [10, 20, 30, 40, 50, 60, 70, 80]
angles_scientific = ['1e-2','1e-2.5','1e-3','1e-3.5','1e-4','1e-4.5','1e-5']

# Define the base directory where the LHEF files are located
base_lhef_dir = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/"

def convert_to_filename(angle):
    return str(angle).replace('.', 'p')

for mass in masses:
    for angle in angles_scientific:
        # Construct source and destination paths
        destination_dir = os.path.join(base_lhef_dir, f"HNL_Dirac_ejj_{mass}GeV_{convert_to_filename(angle)}Ve", "Events", "run_01")
        source_file =  os.path.join(destination_dir, f"HNL_Dirac_ejj_{mass}GeV_{convert_to_filename(angle)}Ve")
        destination_file = os.path.join(destination_dir, f"HNL_Dirac_ejj_{mass}GeV_{convert_to_filename(angle)}Ve.lhe")

        # Rename the file
        os.rename(source_file, destination_file)
        print(f"Renamed: {source_file} -> {destination_file}")
