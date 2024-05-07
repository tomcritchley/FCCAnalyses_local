#!/bin/bash
source /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/venv/bin/activate
python3.11 /afs/cern.ch/work/t/tcritchl/FCCAnalyses_local/DNN/DNN_condor/Run6_opt/test_evaluate_model.py --label "60GeV_1e-3"
