#!/bin/bash

# Define the JSON file path
json_file="/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"

# Extract signal points, masses, and couplings from the JSON file
signal_points=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join([key.split('_')[-2] + '_' + key.split('_')[-1] for key in data]))")

masses=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-2] for key in data])))")
echo "Masses: $masses"

couplings=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-1] for key in data])))")
echo "Couplings: $couplings"

base_path="/eos/user/t/tcritchl/DNN/testing10"

labels=()
for mass in $masses; do
    for coupling in $couplings; do
        x_test_file="X_test_${mass}_${coupling//Ve/}.npy"
        y_test_file="y_test_${mass}_${coupling//Ve/}.npy"
        x_test_path="${base_path}/${x_test_file}"
        y_test_path="${base_path}/${y_test_file}"

        if [ -f "$x_test_path" ] && [ -f "$y_test_path" ]; then
            labels+=("${mass}_${coupling//Ve/}")
            echo "Testing files for $mass and ${coupling//Ve/} exist, added to labels"
        else
            echo "One or both testing files for $mass and ${coupling//Ve/} do not exist, moving to next file"
        fi
    done
done

echo "labels: ${labels[@]}"
# Loop over each label for the current signal point
for label in "${labels[@]}"; do
    echo "label: $label" 
    # Create a unique shell script for the current signal point
    script_file="RunAnSt1_HTC_${label}_testing.sh"
    echo "#!/bin/bash" > "$script_file"
    echo "source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate" >> "$script_file"
    echo "python3.11 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/DNN_less_bkg/test_evaluate_model.py --label \"$label\"" >> "$script_file"
    chmod +x "$script_file"

    # Create a unique Condor submission script for the current signal point
    cat <<EOF > "RunAnSt1_HTC_${label}_testing.condor"
#!/bin/bash
executable     = ./$script_file
universe       = vanilla
arguments      = \$(ClusterId) \$(ProcId)
output         = DNN12_testing_${label}.\$(ClusterId).\$(ProcId).out
error          = DNN12_testing_${label}.\$(ClusterId).\$(ProcId).error
log            = DNN12_testing_${label}.\$(ClusterId).\$(ProcId).log
should_transfer_files   = Yes
when_to_transfer_output = ON_EXIT
environment    = "TESTVAR1=1 TESTVAR2='2' TESTVAR3='spacey ''quoted'' value'"
requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobFlavour    = "testmatch"
queue
EOF

    # Submit a Condor job for the current signal point and label
    condor_submit "RunAnSt1_HTC_${label}_testing.condor"

done
