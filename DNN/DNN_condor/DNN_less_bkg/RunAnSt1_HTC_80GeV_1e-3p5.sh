#!/bin/bash
source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate
python3.11 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/DNN_less_bkg/hyperparameter.py --label "80GeV_1e-3p5"
