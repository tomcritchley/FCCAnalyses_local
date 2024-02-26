import os
# stage 5, get the stage0 root files to pass to the FCC_analyses
# Iterate through each .cmd file
for filename in os.listdir("."):
    if filename.endswith(".cmnd"):
        # Extracting filename without extension
        filename_without_extension = os.path.splitext(filename)[0]
        
        # Constructing the command
        command = f"source /cvmfs/sw.hsf.org/spackages6/key4hep-stack/2022-12-23/x86_64-centos7-gcc11.2.0-opt/ll3gi/setup.sh \
                   && DelphesPythia8_EDM4HEP \
                   ../../../../../../FCC-config/FCCee/Delphes/card_IDEA.tcl \
                   ../../../../../../FCC-config/FCCee/Delphes/edm4hep_IDEA.tcl \
                   {filename} {filename_without_extension}.root"
                   
        # Check if corresponding .root file already exists
        if not os.path.exists(f"{filename_without_extension}.root"):
            # Running the command only if the .root file does not exist
            os.system(command)
            print(f"Executed command for {filename}")
        else:
            print(f"Skipping {filename}, corresponding .root file already exists")
    