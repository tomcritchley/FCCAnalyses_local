#!/bin/bash

# Define the JSON file path
json_file="/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"

# Extract signal points, masses, and couplings from the JSON file
signal_points=$(python -c "import json; data = json.load(open('$json_file')); print(' '.join([key.split('_')[-2] + '_' + key.split('_')[-1] for key in data]))")

masses=$(python -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-2] for key in data])))")
echo "Masses: $masses"

couplings=$(python -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-1] for key in data])))")
echo "Couplings: $couplings"

base_path="/eos/user/t/tcritchl/xgBOOST/training13"
# Generate labels for the current signal point
labels=()
for mass in $masses; do
    for coupling in $couplings; do
        base_file="train_signal_${mass}_${coupling//Ve/}.root"
        signal_file="${base_path}/${base_file}"
        if [ -f "$signal_file" ]; then
            labels+=("signal_${mass}_${coupling//Ve/}")
            echo "File $signal_file added to label as signal_${mass}_${coupling//Ve/}"
        else
            echo "File $signal_file does not exist, moving to next file"
        fi
    done
done
echo "labels: ${labels[@]}"
# Loop over each label for the current signal point
for label in "${labels[@]}"; do
    echo "label: $label" 
    # Create a unique shell script for the current signal point
    script_file="RunAnSt1_HTC_${label}.sh"
    echo "#!/bin/bash" > "$script_file"
    echo "source /cvmfs/sft.cern.ch/lcg/views/dev3/Mon/x86_64-centos7-gcc11-opt/setup.sh" >> "$script_file"
    echo "python3 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/new_variables_training/LLP_study/PromptDecay/training_macro.py --label \"$label\"" >> "$script_file"
    chmod +x "$script_file"

    # Create a unique Condor submission script for the current signal point
    cat <<EOF > "RunAnSt1_HTC_${label}.condor"
#!/bin/bash
executable     = ./$script_file
universe       = vanilla
arguments      = \$(ClusterId) \$(ProcId)
output         = bdt_training_${label}.\$(ClusterId).\$(ProcId).out
error          = bdt_training_${label}.\$(ClusterId).\$(ProcId).error
log            = bdt_training_${label}.\$(ClusterId).\$(ProcId).log
should_transfer_files   = Yes
when_to_transfer_output = ON_EXIT
environment    = "TESTVAR1=1 TESTVAR2='2' TESTVAR3='spacey ''quoted'' value'"
requirements   = (OpSysAndVer =?= "CentOS7")
+JobFlavour    = "testmatch"
queue
EOF

    # Submit a Condor job for the current signal point and label
    condor_submit "RunAnSt1_HTC_${label}.condor"

sleep 5

done
