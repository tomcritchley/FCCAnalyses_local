# stage 2

# Define the masses and angles
masses = [10, 20, 30, 40, 50, 60, 70, 80]
angles_scientific = ['1e-2','1e-2.5','1e-3','1e-3.5','1e-4','1e-4.5','1e-5']

# Define the base directory where the LHEF files are located
base_lhef_dir = "/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/"

def convert_to_filename(angle):
    return str(angle).replace('.', 'p')

# Generate .cmd files
for mass in masses:
    for angle in angles_scientific:
        filename = f"HNL_Dirac_ejj_{mass}GeV_{convert_to_filename(angle)}Ve"
        lhef_path = f"{base_lhef_dir}{filename}/Events/run_01/{filename}.lhe"
        content = f"""\
! File: {filename}.cmnd
Random:setSeed = on
Main:timesAllowErrors = 1000         ! how many aborts before run stops
Main:numberOfEvents = 100000         ! number of events to generate


! 2) Settings related to output in init(), next() and stat().
Next:numberCount = 100             ! print message every n events
!Beams:idA = 11                     ! first beam, e+ = 11
!Beams:idB = -11                    ! second beam, e- = -11

Beams:frameType = 4                ! read info from a LHEF
! Change the LHE file here
Beams:LHEF = {lhef_path}

! 3) Settings for the event generation process in the Pythia8 library.
PartonLevel:ISR = on               ! initial-state radiation
PartonLevel:FSR = on               ! final-state radiation

LesHouches:setLifetime = 2                           
"""
        with open(f"{filename}.cmnd", "w") as f:
            f.write(content)
        print(f"Generated: {filename}.cmnd")
