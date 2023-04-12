import ROOT
import os

#input_dir = "/afs/cern.ch/user/d/dimoulin/FCCAnalyses_new/FCC-LLP/FCCAnalyses_local/examples/FCCee/bsm/LLPs/DisplacedHNL/Analysis/outputs/"
input_dir = "/eos/user/d/dimoulin/analysis_outputs/"

output_dir = "selected_hist/"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Directory ",output_dir," Created ")
else:
    print("Directory ",output_dir," already exists")


selection_list = [
  "selNone",
  "selRecoEleGt0",
  "selRecoEleEGt11",
  "selMissingEGt10",
  "selEleDiJetDRLt27",
  "selEleSecondJetDRGt2",
  "selEleEGt13_MissingEGt10",
  "selEleEGt13_MissingEGt10_EleSecondJetDRGt06",
  "selEleEGt13_MissingEGt10_EleSecondJetDRGt06_EleLeadJetDRGt06",
  "selEleEGt13_MissingEGt10_EleSecondJetDRGt06_EleLeadJetDRGt06Lt32",
]

variables_list = [
     "RecoDiJetElectron_invMass", 
     "RecoElectron_LeadJet_delta_R",
     "RecoElectron_SecondJet_delta_R",
     "RecoElectron_DiJet_delta_R",
     "RecoDiJet_delta_R",
     "RecoElectron_lead_e",
     "RecoMissingEnergy_e",
]

for selection in selection_list:
    files_list = []    

    file_4body = [input_dir + "4body_W2023/output_finalSel/4body_W2023_" + selection + '_histo.root', output_dir + "/4body_W2023_" + selection + '.root']
    file_Zbb = [input_dir + 'p8_ee_Zbb_ecm91_W2023/output_finalSel/p8_ee_Zbb_ecm91_' + selection + '_histo.root', output_dir + "/p8_ee_Zbb_ecm91_W2023_" + selection + ".root"]
    file_Zcc = [input_dir + 'p8_ee_Zcc_ecm91_W2023/output_finalSel/p8_ee_Zcc_ecm91_' + selection + '_histo.root', output_dir + "/p8_ee_Zcc_ecm91_W2023_" + selection + ".root"]
    file_HNL_20GeV = [input_dir + 'HNL_Dirac_ejj_20GeV_1e-3Ve_W2023/output_finalSel/HNL_Dirac_ejj_20GeV_1e-3Ve_W2023_'+selection+'_histo.root', output_dir + "/HNL_Dirac_ejj_20GeV_1e-3Ve_W2023_"+selection+".root"]
    file_HNL_50GeV = [input_dir + 'HNL_Dirac_ejj_50GeV_1e-3Ve_W2023/output_finalSel/HNL_Dirac_ejj_50GeV_1e-3Ve_W2023_'+selection+'_histo.root', output_dir + "/HNL_Dirac_ejj_50GeV_1e-3Ve_W2023_"+selection+".root"]
    file_HNL_70GeV = [input_dir + 'HNL_Dirac_ejj_70GeV_1e-3Ve_W2023/output_finalSel/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023_'+selection+'_histo.root', output_dir + "/HNL_Dirac_ejj_70GeV_1e-3Ve_W2023_"+selection+".root"]

    files_list.append(file_4body)
    files_list.append(file_Zbb)
    files_list.append(file_Zcc)
    files_list.append(file_HNL_20GeV)
    files_list.append(file_HNL_50GeV)
    files_list.append(file_HNL_70GeV)

    for ifile, f in enumerate(files_list):
        hist_file = ROOT.TFile.Open(f[0])
        histSelect = ROOT.TFile.Open(f[1], "RECREATE")
        for var in variables_list:
            histSelect.WriteObject(hist_file.Get(var), var)

print("done")
