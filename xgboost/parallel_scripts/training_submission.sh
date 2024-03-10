#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/dev3/latest/x86_64-centos7-gcc11-opt/setup.sh

# Define the JSON file path
json_file="/afs/cern.ch/work/t/tcritchl/MG5_aMC_v3_5_3/HNL_cross_sections_Feb24.json"

# Convert signal points, masses, and couplings to sets to ensure uniqueness
signal_points=$(python -c "import json; data = json.load(open('$json_file')); print(' '.join(set([f'{key.split('_')[-2]}_{key.split('_')[-1]}' for key in data])))")
masses=$(python -c "import json; data = json.load(open('$json_file')); print(' '.join(set([f'{key.split('_')[-2]}' for key in data])))")
couplings=$(python -c "import json; data = json.load(open('$json_file')); print(' '.join(set([f'{key.split('_')[-1]}' for key in data])))")

# Loop over each signal point
for signal_point in $signal_points; do
    
    base_path="/eos/user/t/tcritchl/xgBOOST/training8"
    # Generate labels for the current signal point
    labels=()
    for mass in $masses; do
        for coupling in $couplings; do
            base_file="train_signal_${mass}_${coupling}.root"
            signal_file="${base_path}/${base_file}"
            if [ -f "$signal_file" ]; then
                labels+=("signal_${mass}_${coupling}")
            else
                echo "File $signal_file does not exist, moving to next file"
            fi
        done
    done

    # Loop over each label for the current signal point
    for label in "${labels[@]}"; do
        # Run the training script for the current signal point
        python3 /afs/cern.ch/t/tcritchl/FCCAnalyses_local/xgboost/parallel_script/training_macro.py --label "$label" --json_file "$json_file"
        
        # Create a unique Condor submission script for the current signal point
        echo "#!/bin/bash" > "RunAnSt1_HTC_${signal_point}.condor"
        echo "executable     = bdt_training.sh" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "universe       = vanilla" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "arguments    = $(ClusterId) $(ProcId)" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "output         = bdt_training_${signal_point}.$(ClusterId).$(ProcId).out" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "error          = bdt_training_${signal_point}.$(ClusterId).$(ProcId).error" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "log            = bdt_training_${signal_point}.$(ClusterId).$(ProcId).log" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "should_transfer_files   = Yes" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "when_to_transfer_output = ON_EXIT" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "environment    = \"TESTVAR1=1 TESTVAR2='2' TESTVAR3='spacey ''quoted'' value'\"" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "requirements   = (OpSysAndVer =?= \"CentOS7\")" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "+JobFlavour    = workday" >> "RunAnSt1_HTC_${signal_point}.condor"
        echo "queue" >> "RunAnSt1_HTC_${signal_point}.condor"
        
        # Submit a Condor job for the current signal point
        condor_submit "RunAnSt1_HTC_${signal_point}.condor"
    done
done
