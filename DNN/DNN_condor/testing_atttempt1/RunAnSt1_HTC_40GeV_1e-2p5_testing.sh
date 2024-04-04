#!/bin/bash
source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate
python3 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/test_evaluate_model.py --label "40GeV_1e-2p5"
