#!/bin/bash

json_file="/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"

signal_points=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join([key.split('_')[-2] + '_' + key.split('_')[-1] for key in data]))")

masses=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-2] for key in data])))")
echo "Masses: $masses"

couplings=$(python3 -c "import json; data = json.load(open('$json_file')); print(' '.join(set([key.split('_')[-1] for key in data])))")
echo "Couplings: $couplings"

base_path="/eos/user/t/tcritchl/DNN/training6"

labels=()
for mass in $masses; do
    for coupling in $couplings; do
        x_train_file="X_train_${mass}_${coupling//Ve/}.npy"
        y_train_file="y_train_${mass}_${coupling//Ve/}.npy"
        x_train_path="${base_path}/${x_train_file}"
        y_train_path="${base_path}/${y_train_file}"

        if [ -f "$x_train_path" ] && [ -f "$y_train_path" ]; then
            labels+=("${mass}_${coupling//Ve/}")
            echo "Training files for $mass and ${coupling//`Ve/} exist, added to labels"
        else
            echo "One or both training files for $mass and ${coupling//Ve/} do not exist, moving to next file"
        fi
    done
done

echo "labels: ${labels[@]}"

for label in "${labels[@]}"; do
    echo "label: $label" 
    script_file="RunAnSt1_HTC_${label}.sh"
    echo "#!/bin/bash" > "$script_file"
    echo "source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate" >> "$script_file"
    echo "python3.11 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/Run6_opt/hyperparameter.py --label \"$label\"" >> "$script_file"
    chmod +x "$script_file"

    cat <<EOF > "RunAnSt1_HTC_${label}.condor"
#!/bin/bash
executable     = ./$script_file
universe       = vanilla
arguments      = \$(ClusterId) \$(ProcId)
output         = DNN_training_${label}.\$(ClusterId).\$(ProcId).out
error          = DNN_training_${label}.\$(ClusterId).\$(ProcId).error
log            = DNN_training_${label}.\$(ClusterId).\$(ProcId).log
should_transfer_files   = Yes
when_to_transfer_output = ON_EXIT
environment    = "TESTVAR1=1 TESTVAR2='2' TESTVAR3='spacey ''quoted'' value'"
requirements = (OpSysAndVer =?= "AlmaLinux9")
+JobFlavour    = "testmatch"
queue
EOF

    condor_submit "RunAnSt1_HTC_${label}.condor"

sleep 5

done
