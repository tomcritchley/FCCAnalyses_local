import uproot
import os
import awkward as ak

# Define the directory containing the root files
root_dir = "/eos/user/t/tcritchl/xgBOOST/fullstats/withvertex/p8_ee_Zcc_ecm91/"

# Define the new cross section value
new_cross_section = 5215.46

# Function to update cross section in a root file
def update_cross_section(root_file):
    try:
        # Open the root file
        with uproot.open(root_file) as file:
            # Get the events tree
            tree = file["events"]
            # Update the cross section value
            cross_section_array = tree["cross_section"].array()
            print(f"cross section array: {cross_section_array}")
            updated_cross_section_array = ak.mask(cross_section_array, cross_section_array == 6654.46, new_cross_section)
            print(f"updated cross section array: {updated_cross_section_array}")
            tree["cross_section"].array()[:] = updated_cross_section_array
            # Save the changes
            file["events"] = tree
            print(f"Cross section updated in {root_file}")
            root_file.close()
    except Exception as e:
        print(f"Error updating cross section in {root_file}: {e}")

# Iterate over all root files in the directory
for root_file in os.listdir(root_dir):
    if root_file.endswith(".root"):
        update_cross_section(os.path.join(root_dir, root_file))
