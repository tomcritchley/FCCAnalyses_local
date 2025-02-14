#!/bin/bash

# Define the JSON file path
json_file="/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"

# Extract signal points, masses, and couplings from the JSON file
signal_points=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join([key.split('_')[-2] + '_' + key.split('_')[-1] for key in data]))")

masses=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-2] for key in data])))")
echo "Masses: $masses"

couplings=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-1] for key in data])))")
echo "Couplings: $couplings"

base_path="/eos/user/t/tcritchl/DNN/testing_205ab"
output_path="/afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/Run16/Run16_205ab"

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

# Run each job sequentially, only if the output file does NOT already exist
for label in "${labels[@]}"; do
    output_file="${output_path}/DNN_RunFINAL_205ab_${label}.json"

    if [ -f "$output_file" ]; then
        echo "Skipping $label - Output file already exists: $output_file"
        continue
    fi

    echo "Running evaluation for label: $label"
    source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate
    python3.11 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/Run16/Run16_205ab/test_evaluate_model.py --label "$label"
done

