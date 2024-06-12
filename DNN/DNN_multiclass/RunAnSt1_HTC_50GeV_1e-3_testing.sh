#!/bin/bash
source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate
python3.11 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_multiclass/test_evaluate_model.py --label "50GeV_1e-3"
