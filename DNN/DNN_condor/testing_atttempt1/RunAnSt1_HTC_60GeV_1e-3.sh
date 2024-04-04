#!/bin/bash
source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate
python3 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/train_model.py --label "60GeV_1e-3"
