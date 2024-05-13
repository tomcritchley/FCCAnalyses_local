#!/bin/bash
source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate
python3.11 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/4body_only/Data_Preparation.py --label "40GeV_1e-2p5"
