#!/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/dev3/Mon/x86_64-centos7-gcc11-opt/setup.sh
python3 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/xgboost/new_variables_training/LLP_study/PromptDecay/training_macro.py --label "signal_20GeV_1e-5"
