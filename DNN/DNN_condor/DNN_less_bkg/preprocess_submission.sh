#!/bin/bash

# Define the JSON file path
json_file="/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"

# Extract signal points, masses, and couplings from the JSON file
signal_points=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join([key.split('_')[-2] + '_' + key.split('_')[-1] for key in data]))")

masses=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-2] for key in data])))")
echo "Masses: $masses"

couplings=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-1] for key in data])))")
echo "Couplings: $couplings"

base_path="/eos/user/t/tcritchl/DNN/testing11"

labels=()

for mass in $masses; do
    for coupling in $couplings; do
        labels+=("${mass}_${coupling//Ve/}")
    done
done

echo "labels: ${labels[@]}"
# Loop over each label for the current signal point
for label in "${labels[@]}"; do
    echo "label: $label" 
    # Create a unique shell script for the current signal point
    script_file="RunAnSt1_HTC_${label}_preprocess.sh"
    echo "#!/bin/bash" > "$script_file"
    echo "source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate" >> "$script_file"
    echo "python3.11 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/DNN_less_bkg/Data_Preparation.py --label \"$label\"" >> "$script_file"
    chmod +x "$script_file"

    # Create a unique Condor submission script for the current signal point
    cat <<EOF > "RunAnSt1_HTC_${label}_preprocess.condor"
#!/bin/bash
executable     = ./$script_file
universe       = vanilla
arguments      = \$(ClusterId) \$(ProcId)
output         = DNN_preprocess_${label}.\$(ClusterId).\$(ProcId).out
error          = DNN_preprocess_${label}.\$(ClusterId).\$(ProcId).error
log            = DNN_preprocess_${label}.\$(ClusterId).\$(ProcId).log
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
environment    = "TESTVAR1=1 TESTVAR2='2' TESTVAR3='spacey ''quoted'' value'"
requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobFlavour    = "testmatch"
queue
EOF

    # Submit a Condor job for the current signal point and label
    condor_submit "RunAnSt1_HTC_${label}_preprocess.condor"
    
done
